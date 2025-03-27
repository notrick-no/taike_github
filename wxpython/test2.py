import wx
import wx.lib.agw.aui as aui
import random
import os
import cv2
import threading
import numpy as np
from datetime import datetime
from wx.lib.pubsub import pub


class DefectDetectorFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="工业缺陷检测系统", size=(900, 600))

        # 初始化界面
        self._init_ui()
        self._bind_events()

        # 模拟数据
        self.demo_images = ["img.png"]  # 准备2张示例图片
        self.current_image = None

        # 视频流相关
        self.camera_thread = None
        self.is_live = False
        self.current_frame = None

    def _init_ui(self):
        """界面组件初始化"""
        self.SetMinSize((800, 600))

        # 创建管理器
        self._mgr = aui.AuiManager(self)

        # 主图像显示面板
        self.image_panel = wx.Panel(self)
        self.image_ctrl = wx.StaticBitmap(self.image_panel)
        self._mgr.AddPane(self.image_panel, aui.AuiPaneInfo().CenterPane())

        # 状态栏
        self.status_bar = self.CreateStatusBar(2)
        self.status_bar.SetStatusWidths([-2, -1])
        self.status_bar.SetStatusText("就绪", 0)

        # 新增视频控制按钮
        control_panel = self._create_control_panel()
        video_btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.video_start_btn = wx.Button(control_panel, label="启动摄像头")
        self.video_stop_btn = wx.Button(control_panel, label="停止视频流")
        video_btn_sizer.Add(self.video_start_btn, 0, wx.ALL, 5)
        video_btn_sizer.Add(self.video_stop_btn, 0, wx.ALL, 5)

        # 将视频按钮添加到原有控制面板
        control_panel.GetSizer().Insert(3, video_btn_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # 订阅消息
        pub.subscribe(self.update_video_display, "video_frame")
        pub.subscribe(self.show_video_error, "video_error")

        # 右侧控制面板
        control_panel = self._create_control_panel()
        self._mgr.AddPane(control_panel, aui.AuiPaneInfo().Right().Layer(1).BestSize(300, -1))

        self._mgr.Update()

    def _create_control_panel(self):
        """创建右侧控制面板"""
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # 1. 连接设置
        conn_box = wx.StaticBox(panel, label="设备连接")
        conn_sizer = wx.StaticBoxSizer(conn_box, wx.VERTICAL)
        self.cam_ip = wx.TextCtrl(panel, value="192.168.1.100")
        conn_sizer.Add(self.cam_ip, 0, wx.EXPAND | wx.ALL, 5)

        # 2. 检测参数
        param_box = wx.StaticBox(panel, label="检测参数")
        param_sizer = wx.StaticBoxSizer(param_box, wx.VERTICAL)

        self.confidence = wx.Slider(panel, value=70, minValue=0, maxValue=100)
        param_sizer.Add(wx.StaticText(panel, label="置信度阈值 (70%)"), 0, wx.ALL, 5)
        param_sizer.Add(self.confidence, 0, wx.EXPAND | wx.ALL, 5)

        # 3. 统计信息
        stats_box = wx.StaticBox(panel, label="实时统计")
        stats_sizer = wx.StaticBoxSizer(stats_box, wx.VERTICAL)
        self.defect_count = wx.StaticText(panel, label="缺陷数量: 0")
        self.defect_type = wx.StaticText(panel, label="缺陷类型: 无")
        stats_sizer.Add(self.defect_count, 0, wx.ALL, 5)
        stats_sizer.Add(self.defect_type, 0, wx.ALL, 5)

        # 4. 操作按钮
        btn_sizer = wx.GridSizer(2, 2, 5, 5)
        self.start_btn = wx.Button(panel, label="开始检测")
        self.stop_btn = wx.Button(panel, label="停止检测")
        self.load_btn = wx.Button(panel, label="加载图像")
        self.export_btn = wx.Button(panel, label="导出报告")
        btn_sizer.AddMany([(self.start_btn), (self.stop_btn),
                           (self.load_btn), (self.export_btn)])

        # 整合所有组件
        vbox.Add(conn_sizer, 0, wx.EXPAND | wx.ALL, 5)
        vbox.Add(param_sizer, 0, wx.EXPAND | wx.ALL, 5)
        vbox.Add(stats_sizer, 0, wx.EXPAND | wx.ALL, 5)
        vbox.Add(btn_sizer, 1, wx.EXPAND | wx.ALL, 5)

        panel.SetSizer(vbox)
        return panel

    def _bind_events(self):
        """事件绑定"""
        self.Bind(wx.EVT_BUTTON, self.on_start, self.start_btn)
        self.Bind(wx.EVT_BUTTON, self.on_stop, self.stop_btn)
        self.Bind(wx.EVT_BUTTON, self.on_load_image, self.load_btn)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_BUTTON, self.on_video_start, self.video_start_btn)
        self.Bind(wx.EVT_BUTTON, self.on_video_stop, self.video_stop_btn)

    def _update_ui(self, img_path):
        """更新图像显示"""
        img = wx.Image(img_path, wx.BITMAP_TYPE_ANY)
        w, h = self.image_panel.GetSize()
        img = img.Scale(w, h, wx.IMAGE_QUALITY_HIGH)
        self.image_ctrl.SetBitmap(wx.Bitmap(img))
        self.image_panel.Layout()

    def _random_defects(self):
        """生成模拟缺陷数据"""
        return {
            "count": random.randint(0, 5),
            "types": random.choice(["凸起", "缺失", "变形", "混合"]),
            "confidence": round(random.uniform(0.6, 0.95), 2)
        }

    # ---------- 事件处理 ----------
    def on_start(self, event):
        """开始检测"""
        if not self.current_image:
            wx.MessageBox("请先加载图像!", "警告", wx.OK | wx.ICON_WARNING)
            return

        self.status_bar.SetStatusText("检测中...", 0)
        self.start_btn.Disable()

        # 模拟检测过程
        wx.CallLater(2000, self._show_result)

    def _show_result(self):
        """显示检测结果"""
        result = self._random_defects()
        status = f"发现 {result['count']} 处缺陷" if result["count"] > 0 else "无缺陷"
        self.status_bar.SetStatusText(status, 0)

        # 更新统计
        self.defect_count.SetLabel(f"缺陷数量: {result['count']}")
        self.defect_type.SetLabel(f"缺陷类型: {result['types']}")
        self.start_btn.Enable()

        # 显示带缺陷框的图片（此处可替换为实际标记图）
        self._update_ui(random.choice(self.demo_images))

    def on_stop(self, event):
        """停止检测"""
        self.status_bar.SetStatusText("检测已停止", 0)
        self.start_btn.Enable()

    def on_load_image(self, event):
        """加载测试图像"""
        dlg = wx.FileDialog(self, "选择检测图像", "", "",
                            "图像文件 (*.jpg;*.png)|*.jpg;*.png", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.current_image = path
            self._update_ui(path)
            self.status_bar.SetStatusText(f"已加载: {os.path.basename(path)}", 0)
        dlg.Destroy()

    def on_video_start(self, event):
        """启动摄像头"""
        if not self.is_live:
            self.camera_thread = VideoThread()
            self.camera_thread.start()
            self.is_live = True
            self.status_bar.SetStatusText("摄像头已启动", 0)
            self.video_start_btn.Disable()
            self.video_stop_btn.Enable()

    def on_video_stop(self, event):
        """停止视频流"""
        if self.is_live:
            self.camera_thread.stop()
            self.is_live = False
            self.status_bar.SetStatusText("视频流已停止", 0)
            self.video_start_btn.Enable()
            self.video_stop_btn.Disable()
    def show_video_error(self, msg):
        """显示视频错误"""
        wx.MessageBox(msg, "视频流错误", wx.OK | wx.ICON_ERROR)
        self.on_video_stop(None)

    def update_video_display(self, frame):
        """更新视频显示（集成原有图像显示功能）"""
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 在此处添加实际缺陷检测调用
        # results = model.predict(frame)
        # 以下是模拟检测结果
        h, w = frame.shape[:2]
        fake_boxes = [
            (np.random.randint(0, w - 100), np.random.randint(0, h - 100),
             100, 100, '凸起', 0.85),
            (np.random.randint(0, w - 100), np.random.randint(0, h - 100),
             100, 100, '缺失', 0.92)
        ]

        # 绘制检测结果
        for (x, y, w, h, label, conf) in fake_boxes:
            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame_rgb, f"{label} {conf:.2f}",
                        (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)

        # 更新显示
        h, w = frame_rgb.shape[:2]
        img = wx.Image(w, h, frame_rgb.tobytes())
        img = img.Scale(self.image_panel.Size.width,
                        self.image_panel.Size.height,
                        wx.IMAGE_QUALITY_BICUBIC)
        self.image_ctrl.SetBitmap(wx.Bitmap(img))
        self.image_panel.Refresh()

        # 更新统计信息（模拟）
        self.defect_count.SetLabel(f"缺陷数量: {len(fake_boxes)}")
        self.defect_type.SetLabel("缺陷类型: 凸起, 缺失")

    def on_close(self, event):
        """安全关闭"""
        if self.is_live:
            self.on_video_stop(None)
        self._mgr.UnInit()
        self.Destroy()


class VideoThread(threading.Thread):
    """视频采集线程（新增类）"""

    def __init__(self, source=0):
        super().__init__()
        self.source = source  # 可以是摄像头索引或视频文件路径
        self.running = False
        self.cap = None

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            pub.sendMessage("video_error", msg=f"无法打开视频源: {self.source}")
            return

        # 设置工业相机参数（示例）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                pub.sendMessage("video_frame", frame=frame)
            else:
                pub.sendMessage("video_error", msg="视频流中断")
                break

        self.cap.release()

    def stop(self):
        self.running = False


if __name__ == "__main__":
    app = wx.App()
    frame = DefectDetectorFrame()
    frame.Show()
    app.MainLoop()