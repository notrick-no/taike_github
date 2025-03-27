# -*- coding: utf-8 -*-
import threading
import queue

import numpy as np              # 数据处理的库numpy
import cv2                      # 图像处理的库OpenCv
import wx                       # 构造显示界面的GUI
import wx.xrc
import wx.adv
import argparse
import datetime,time
import math
import os
import torch
import torch.backends.cudnn as cudnn
import torch.hub
import traceback


class VideoCanvas(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        self.current_frame = None  # 当前视频帧
        self.buffer = wx.Bitmap(1, 1)  # 初始化缓冲位图

        # 绑定绘制事件
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)  # 处理窗口尺寸变化

    def on_paint(self, event):
        # 创建双缓冲设备上下文
        dc = wx.BufferedPaintDC(self, self.buffer)
        if self.buffer.IsOk():
            # 获取画布尺寸
            width, height = self.GetClientSize()
            # 计算居中绘制的位置
            x = (width - self.buffer.GetWidth()) // 2
            y = (height - self.buffer.GetHeight()) // 2
            # 绘制缓冲位图
            dc.DrawBitmap(self.buffer, x, y, True)

    def update_frame(self, bitmap):
        """ 安全更新帧的线程方法 """
        if not bitmap.IsOk():
            return

        # 计算保持宽高比的缩放尺寸
        img_w, img_h = bitmap.GetWidth(), bitmap.GetHeight()
        panel_w, panel_h = self.GetClientSize()

        # 计算缩放比例
        scale = min(panel_w / img_w, panel_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # 创建缩放后的位图
        scaled_bmp = bitmap.ConvertToImage()
        scaled_bmp = scaled_bmp.Scale(new_w, new_h, wx.IMAGE_QUALITY_BILINEAR)
        scaled_bmp = wx.Bitmap(scaled_bmp)

        # 更新缓冲位图
        self.buffer = scaled_bmp
        self.Refresh()  # 触发重绘

    def on_size(self, event):
        """ 窗口尺寸变化时重置缓冲 """
        self.buffer = wx.Bitmap(*self.GetClientSize())
        self.Refresh()
        event.Skip()


class DefectDetectorFrame(wx.Frame):
    def __init__(self, parent, title):
        # 初始化窗口基础设置
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=title,
                          size=(1200, 800),  # 更适合工业检测的窗口尺寸
                          style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        # ---------------------------
        # 界面初始化
        # ---------------------------
        self._init_ui()
        self._bind_events()

        # ---------------------------
        # 系统参数初始化
        # ---------------------------
        self._init_parameters()

        # 渲染任务队列
        self.render_queue = queue.Queue()
        self.render_thread = threading.Thread(target=self._render_worker)
        self.render_thread.daemon = True
        self.render_thread.start()

        print("工业缺陷检测系统界面初始化完成")

    def _init_ui(self):
        """界面布局初始化"""
        # 主布局容器
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # ================= 左侧视频显示区域 =================
        # 视频显示面板（占窗口70%宽度）
        video_panel = wx.Panel(self, size=(840, 800))
        video_panel.SetBackgroundColour(wx.Colour(30, 30, 30))  # 工业深色背景

        # # 视频显示控件（后续用于显示实时视频流）
        # self. = wx.StaticBitmap(video_panel)
        # video_sizer = wx.BoxSizer(wx.VERTICAL)
        # video_sizer.Add(self., 1, wx.EXPAND | wx.ALL, 5)
        # video_panel.SetSizer(video_sizer)

        video_panel = wx.Panel(self)
        self.video_canvas = VideoCanvas(video_panel)
        video_sizer = wx.BoxSizer(wx.VERTICAL)
        video_sizer.Add(self.video_canvas, 1, wx.EXPAND | wx.ALL,5)
        video_panel.SetSizer(video_sizer)

        main_sizer.Add(video_panel, 7, wx.EXPAND)  # 7:3的宽度比例

        # ================= 右侧控制面板区域 =================
        control_panel = wx.Panel(self, size=(360, 800))
        control_sizer = wx.BoxSizer(wx.VERTICAL)

        # ----------------- 设备控制区块 -----------------
        device_box = wx.StaticBox(control_panel, label="设备控制")
        device_sizer = wx.StaticBoxSizer(device_box, wx.VERTICAL)

        # 视频源选择
        source_choice_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.cam_choices = wx.Choice(device_box, choices=[
            "摄像头 0 (主)",
            "摄像头 1 (辅)",
            "RTSP流"
        ])
        self.cam_choices.SetSelection(0)
        source_choice_sizer.Add(wx.StaticText(device_box, label="视频源:"), 0, wx.ALIGN_CENTER)
        source_choice_sizer.Add(self.cam_choices, 1, wx.EXPAND | wx.LEFT, 10)

        # 控制按钮组
        btn_grid = wx.GridSizer(2, 2, 5, 5)  # 2行2列，间距5px
        self.btn_start = wx.Button(device_box, label="▶ 启动检测")
        self.btn_stop = wx.Button(device_box, label="⏹ 停止检测")
        self.btn_export = wx.Button(device_box, label="📁 导出报告")
        self.btn_config = wx.Button(device_box, label="⚙ 系统设置")
        [btn_grid.Add(btn, 0, wx.EXPAND) for btn in (
            self.btn_start, self.btn_stop,
            self.btn_export, self.btn_config
        )]

        # 组装设备控制区块
        device_sizer.Add(source_choice_sizer, 0, wx.EXPAND | wx.ALL, 5)
        device_sizer.Add(btn_grid, 1, wx.EXPAND | wx.ALL, 5)

        # ----------------- 检测参数区块 -----------------
        param_box = wx.StaticBox(control_panel, label="检测参数")
        param_sizer = wx.StaticBoxSizer(param_box, wx.VERTICAL)

        # 缺陷类型选择
        defect_type_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.check_raise = wx.CheckBox(param_box, label="凸起检测")
        self.check_missing = wx.CheckBox(param_box, label="缺失检测")
        self.check_deform = wx.CheckBox(param_box, label="变形检测")
        [defect_type_sizer.Add(cb, 0, wx.RIGHT, 5) for cb in (
            self.check_raise, self.check_missing, self.check_deform
        )]

        # 灵敏度调节滑动条
        sensitivity_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.slider_sensitivity = wx.Slider(param_box, value=75, minValue=0, maxValue=100,
                                            style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        sensitivity_sizer.Add(wx.StaticText(param_box, label="灵敏度:"), 0, wx.ALIGN_CENTER)
        sensitivity_sizer.Add(self.slider_sensitivity, 1, wx.EXPAND)

        # 组装参数区块
        param_sizer.Add(defect_type_sizer, 0, wx.EXPAND | wx.BOTTOM, 10)
        param_sizer.Add(sensitivity_sizer, 0, wx.EXPAND)

        # ----------------- 状态输出区块 -----------------
        status_box = wx.StaticBox(control_panel, label="检测状态")
        status_sizer = wx.StaticBoxSizer(status_box, wx.VERTICAL)

        # 实时统计信息
        self.lbl_defect_count = wx.StaticText(status_box, label="缺陷总数: 0")
        self.lbl_current_type = wx.StaticText(status_box, label="当前缺陷: 无")

        # 日志输出区域
        self.log_output = wx.TextCtrl(status_box, style=wx.TE_MULTILINE | wx.TE_READONLY)

        # 组装状态区块
        status_sizer.Add(self.lbl_defect_count, 0, wx.EXPAND | wx.BOTTOM, 5)
        status_sizer.Add(self.lbl_current_type, 0, wx.EXPAND | wx.BOTTOM, 10)
        status_sizer.Add(self.log_output, 1, wx.EXPAND)

        # 整合所有区块到右侧面板
        control_sizer.Add(device_sizer, 3, wx.EXPAND | wx.ALL, 5)  # 设备控制占3份高度
        control_sizer.Add(param_sizer, 2, wx.EXPAND | wx.ALL, 5)  # 参数设置占2份
        control_sizer.Add(status_sizer, 5, wx.EXPAND | wx.ALL, 5)  # 状态输出占5份
        control_panel.SetSizer(control_sizer)

        main_sizer.Add(control_panel, 3, wx.EXPAND)
        self.SetSizer(main_sizer)

        # 设置窗口图标（示例路径，需替换实际图标文件）
        self.SetIcon(wx.Icon('./images/shangda2.ico', wx.BITMAP_TYPE_ICO))

    def _bind_events(self):
        """事件绑定"""
        # 按钮事件
        self.btn_start.Bind(wx.EVT_BUTTON, self.on_start_detection)
        self.btn_stop.Bind(wx.EVT_BUTTON, self.on_stop_detection)
        self.btn_export.Bind(wx.EVT_BUTTON, self.on_export_report)

        # 窗口关闭事件
        self.Bind(wx.EVT_CLOSE, self.on_close_window)

    def _init_parameters(self):
        """系统参数初始化"""
        # 视频流相关
        self.is_detecting = False  # 检测状态标志
        self.current_stream = None  # 当前视频流对象

        # 模拟检测参数
        self.defect_types = {
            'raise': True,  # 凸起检测启用
            'missing': True,  # 缺失检测启用
            'deform': False  # 变形检测默认关闭
        }

        # 状态跟踪
        self.total_defects = 0  # 累计缺陷数量
        self.current_defect = None  # 当前检测到的缺陷类型

    # ---------------------------
    # 事件处理函数（伪代码示例）
    # ---------------------------
    def on_start_detection(self, event):
        """启动检测"""
        if not self.is_detecting:
            # 获取选择的视频源
            source_type = self.cam_choices.GetStringSelection()

            # 初始化视频流（示例伪代码）
            try:
                if source_type == "RTSP流":
                    self.current_stream = RTSPStream(address="192.168.1.100")  # 伪代码类
                else:
                    cam_id = int(source_type.split()[1])  # 提取摄像头ID
                    self.current_stream = CameraStream(cam_id)  # 伪代码类

                # 启动检测线程
                self.detection_thread = DetectionThread(
                    stream=self.current_stream,
                    callback=self.update_ui  # 回调函数，用于更新UI
                )
                self.detection_thread.start()

                # 更新界面状态
                self.is_detecting = True
                self.btn_start.Disable()
                self.btn_stop.Enable()
                self.log_output.AppendText(f"[系统] 已启动{source_type}检测\n")

            except Exception as e:
                wx.MessageBox(f"启动失败: {str(e)}", "错误", wx.OK | wx.ICON_ERROR)

    def update_ui(self, result):
        """更新界面显示（伪代码示例）"""
        # 在主线程更新UI
        wx.CallAfter(self._process_detection_result, result)

    def _render_worker(self):
        """渲染线程的工作函数"""
        while True:
            try:
                # 从队列中获取渲染任务
                frame, boxes = self.render_queue.get(timeout=1)
                if frame is None:
                    break

                # 创建副本帧用于绘制锚框
                overlay_frame = frame.copy()

                # 在副本帧上绘制锚框
                for box, label, color in boxes:
                    overlay_frame = draw_defect_rect(overlay_frame, box, label, color)

                # 在主线程中更新UI
                wx.CallAfter(self.show_frame, overlay_frame)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"渲染线程出错: {str(e)}")
                traceback.print_exc()

    def _process_detection_result(self, result):
        """处理检测结果，添加红色大框、蓝色小框，并控制渲染持续不短于1秒"""
        try:
            import random
            import time

            # 初始化最后一次框处理时间和缓存框架
            if not hasattr(self, "_last_detection_time"):
                self._last_detection_time = 0
                self._cached_boxes = []  # 缓存红色大框和蓝色小框
                self._cached_frame = None

            # 判断时间间隔是否满足1秒展示需求
            current_time = time.time()
            if current_time - self._last_detection_time >= 3:  # 每3秒生成新框数据
                self._last_detection_time = current_time

            # 使用Module1检测并获取锚框位置
            if not hasattr(self, 'module1'):
                from model_interface import Module1
                self.module1 = Module1()

            # 调用detect_and_crop方法获取锚框
            boxes, _ = self.module1.detect_and_crop(result['frame'])

            # 转换为与当前代码兼容的格式
            self._cached_boxes = []
            for box in boxes:
                x1, y1, x2, y2, confidence = box
                self._cached_boxes.append(((x1, y1, x2, y2), f"Part {confidence:.2f}", (0, 0, 255)))

            # 更新当前帧
            self._cached_frame = result['frame'].copy()

            # 将渲染任务放入队列
            if self._cached_frame is not None:
                self.render_queue.put((self._cached_frame.copy(), self._cached_boxes))

        except Exception as e:
            print("处理检测渲染的逻辑出现异常:", e)
        except Exception as e:
            print(f"处理检测结果时出错：{str(e)}")
            import traceback
            traceback.print_exc()

    def show_frame(self, frame):
        """显示视频帧"""
        try:
            if frame is None:
                print("警告：接收到空帧")
                return
            
            # 调整图像大小以适应显示区域
            height, width = frame.shape[:2]
            # 假设显示区域大小为840x600
            target_width = 840
            target_height = 600
            
            # 计算缩放比例
            scale = min(target_width/width, target_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # 缩放图像
            frame = cv2.resize(frame, (new_width, new_height))
            print(f"缩放后帧尺寸: {frame.shape}")
            
            # 确保图像为RGB格式
            if len(frame.shape) == 2:  # 灰度图像
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:  # RGBA图像
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            print(f"转换后帧尺寸: {frame.shape}, 通道数: {frame.shape[2]}")
            
            # 转换为wx格式
            wx_image = wx.Bitmap.FromBuffer(new_width, new_height, frame)
            print(f"wx.Bitmap创建成功: {wx_image.IsOk()}")
            
            # 使用VideoCanvas更新显示
            self.video_canvas.update_frame(wx_image)
            print("视频帧已成功更新到画布")
            
        except Exception as e:
            print(f"显示帧时出错：{str(e)}")
        traceback.print_exc()

    def on_stop_detection(self, event):
        """停止检测"""
        if self.is_detecting:
            try:
                if self.current_stream:
                    self.current_stream.stop()  # 停止视频流
                if hasattr(self, 'detection_thread'):
                    self.detection_thread.stop()  # 停止检测线程
                    self.detection_thread.join()  # 等待线程结束

                # 重置状态
                self.is_detecting = False
                self.btn_stop.Disable()
                self.btn_start.Enable()
                self.log_output.AppendText("[系统] 检测已停止\n")
            except Exception as e:
                self.log_output.AppendText(f"[错误] 停止检测时出错: {str(e)}\n")

    def on_export_report(self, event):
        """导出检测报告（伪代码）"""
        # 示例保存路径选择对话框
        with wx.FileDialog(self, "保存报告", wildcard="CSV文件 (*.csv)|*.csv") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                save_path = dlg.GetPath()
                generate_report(self.total_defects, save_path)  # 伪代码函数
                self.log_output.AppendText(f"[系统] 报告已导出至 {save_path}\n")

    def on_close_window(self, event):
        """安全关闭窗口"""
        if self.is_detecting:
            self.on_stop_detection(None)
        self.Destroy()


# ================= 伪代码类/函数示例 =================
def RTSPStream(address):
    pass

def draw_defect_rect(frame, box, label, color):
    """在图像上绘制缺陷检测框,label为缺陷类型,box为坐标(x1,y1,x2,y2),frame为图像"""
    # 确保操作的是输入帧，不会修改原始图像
    x1, y1, x2, y2 = map(int, box)  # 确保坐标是整数
    
    # 根据颜色绘制矩形框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
    
    # 在矩形框上方添加标签文字
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    top_left = (x1, y1 - 10)
    bottom_right = (x1 + label_size[0], y1)
    cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)  # 背景
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), thickness=1)
    
    return frame  # 返回修改后的帧

def generate_report(total_defects, save_path):
    pass

class CameraStream:
    """模拟摄像头流类"""
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.is_running = True
        print(f"正在尝试打开摄像头 {cam_id}")  # 添加调试信息
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            print(f"摄像头 {cam_id} 打开失败")  # 添加调试信息
            raise Exception(f"无法打开摄像头 {cam_id}")
        else:
            print(f"摄像头 {cam_id} 打开成功")  # 添加调试信息
            
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def read(self):
        if not self.is_running:
            print("摄像头流已停止")
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("警告：无法从摄像头读取帧")
                return None
                
            # 打印帧信息
            print(f"成功读取帧，尺寸: {frame.shape}, 类型: {frame.dtype}")
            
            # 将BGR转换为RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"转换后帧信息，尺寸: {frame.shape}, 类型: {frame.dtype}")
            
            return {
                'frame': frame,
                'defect_found': np.random.choice([True, False]),
                'defect_type': np.random.choice(["凸起", "缺失", "变形"]),
                'position': (100, 100, 200, 200)
            }
        except Exception as e:
            print(f"读取摄像头帧时发生错误: {str(e)}")
            traceback.print_exc()
            return None
        
    def stop(self):
        """停止摄像头"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()

class DetectionThread(threading.Thread):
    """检测线程"""
    def __init__(self, stream, callback):
        super().__init__()
        self.stream = stream
        self.callback = callback
        self.running = True  # 添加运行状态标志
        
    def run(self):
        """线程运行函数"""
        while self.running:
            try:
                result = self.stream.read()
                if result is None:  # 如果流已停止
                    break
                self.callback(result)
            except Exception as e:
                print(f"检测线程错误: {str(e)}")
                break
                
    def stop(self):
        """停止线程"""
        self.running = False

if __name__ == "__main__":
    app = wx.App()
    frame = DefectDetectorFrame(None, "工业缺陷检测系统")
    frame.Show()
    app.MainLoop()