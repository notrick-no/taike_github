import wx
import argparse
import wx.lib.agw.aui as aui
import random
import os
import cv2
import threading
import numpy as np
from datetime import datetime
from wx.lib.pubsub import pub

COVER = './images/shangda8.png'

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='weights\mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or slim or RFB')

parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--long_side', default=640, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


class Product_detecting(wx.Frame):
    # 使用 wxPython 创建图形界面应用程序的代码
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=title, pos=wx.DefaultPosition, size=wx.Size(873, 535),
                          style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENU))

        bSizer1 = wx.BoxSizer(wx.VERTICAL)
        bSizer2 = wx.BoxSizer(wx.HORIZONTAL)
        bSizer3 = wx.BoxSizer(wx.VERTICAL)

        self.m_animCtrl1 = wx.adv.AnimationCtrl(self, wx.ID_ANY, wx.adv.NullAnimation, wx.DefaultPosition,
                                                wx.DefaultSize, wx.adv.AC_DEFAULT_STYLE)
        bSizer3.Add(self.m_animCtrl1, 1, wx.ALL | wx.EXPAND, 5)
        bSizer2.Add(bSizer3, 9, wx.EXPAND, 5)
        bSizer4 = wx.BoxSizer(wx.VERTICAL)
        sbSizer1 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"参数设置"), wx.VERTICAL)
        sbSizer2 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"视频源"), wx.VERTICAL)
        gSizer1 = wx.GridSizer(0, 2, 0, 8)
        m_choice1Choices = [u"摄像头ID_0", u"摄像头ID_1", u"摄像头ID_2"]
        self.m_choice1 = wx.Choice(sbSizer2.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size(90, 25),
                                   m_choice1Choices, 0)
        self.m_choice1.SetSelection(0)
        gSizer1.Add(self.m_choice1, 0, wx.ALL, 5)
        self.camera_button1 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"开始检测", wx.DefaultPosition,
                                        wx.Size(90, 25), 0)
        gSizer1.Add(self.camera_button1, 0, wx.ALL, 5)
        self.vedio_button2 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"打开视频文件", wx.DefaultPosition,
                                       wx.Size(90, 25), 0)
        gSizer1.Add(self.vedio_button2, 0, wx.ALL, 5)

        self.off_button3 = wx.Button(sbSizer2.GetStaticBox(), wx.ID_ANY, u"暂停", wx.DefaultPosition, wx.Size(90, 25),
                                     0)
        gSizer1.Add(self.off_button3, 0, wx.ALL, 5)
        sbSizer2.Add(gSizer1, 1, wx.EXPAND, 5)
        sbSizer1.Add(sbSizer2, 2, wx.EXPAND, 5)
        sbSizer3 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"疲劳检测"), wx.VERTICAL)
        bSizer5 = wx.BoxSizer(wx.HORIZONTAL)
        self.yawn_checkBox1 = wx.CheckBox(sbSizer3.GetStaticBox(), wx.ID_ANY, u"打哈欠检测", wx.Point(-1, -1),
                                          wx.Size(-1, 15), 0)
        self.yawn_checkBox1.SetValue(True)
        bSizer5.Add(self.yawn_checkBox1, 0, wx.ALL, 5)
        self.blink_checkBox2 = wx.CheckBox(sbSizer3.GetStaticBox(), wx.ID_ANY, u"闭眼检测", wx.Point(-1, -1),
                                           wx.Size(-1, 15), 0)
        self.blink_checkBox2.SetValue(True)
        bSizer5.Add(self.blink_checkBox2, 0, wx.ALL, 5)
        sbSizer3.Add(bSizer5, 1, wx.EXPAND, 5)

        bSizer6 = wx.BoxSizer(wx.HORIZONTAL)
        self.face_checkBox4 = wx.CheckBox(sbSizer3.GetStaticBox(), wx.ID_ANY, u"离位检测", wx.DefaultPosition,
                                          wx.Size(-1, 15), 0)
        self.face_checkBox4.SetValue(True)
        bSizer6.Add(self.face_checkBox4, 0, wx.ALL, 5)

        self.reset_button4 = wx.Button(sbSizer3.GetStaticBox(), wx.ID_ANY, u"重置", wx.DefaultPosition, wx.Size(-1, 22),
                                       0)
        bSizer6.Add(self.reset_button4, 0, wx.ALL, 5)
        sbSizer3.Add(bSizer6, 1, wx.EXPAND, 5)
        sbSizer1.Add(sbSizer3, 2, 0, 5)

        sbSizer6 = wx.StaticBoxSizer(wx.StaticBox(sbSizer1.GetStaticBox(), wx.ID_ANY, u"状态输出"), wx.VERTICAL)
        self.m_textCtrl3 = wx.TextCtrl(sbSizer6.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition,
                                       wx.DefaultSize, wx.TE_MULTILINE | wx.TE_READONLY)
        sbSizer6.Add(self.m_textCtrl3, 1, wx.ALL | wx.EXPAND, 5)
        sbSizer1.Add(sbSizer6, 5, wx.EXPAND, 5)
        bSizer4.Add(sbSizer1, 1, wx.EXPAND, 5)
        bSizer2.Add(bSizer4, 3, wx.EXPAND, 5)
        bSizer1.Add(bSizer2, 1, wx.EXPAND, 5)

        self.SetSizer(bSizer1)
        self.Layout()
        self.Centre(wx.BOTH)

        # Connect Events
        self.m_choice1.Bind(wx.EVT_CHOICE, self.cameraid_choice)  # 绑定事件
        self.camera_button1.Bind(wx.EVT_BUTTON, self.camera_on)  # 开
        self.vedio_button2.Bind(wx.EVT_BUTTON, self.vedio_on)
        self.off_button3.Bind(wx.EVT_BUTTON, self.off)  # 关
        self.reset_button4.Bind(wx.EVT_BUTTON, self.reset)

        # 封面图片
        self.image_cover = wx.Image(COVER, wx.BITMAP_TYPE_ANY)
        # 显示图片在m_animCtrl1上
        self.bmp = wx.StaticBitmap(self.m_animCtrl1, -1, wx.Bitmap(self.image_cover))

        # 设置窗口标题的图标
        self.icon = wx.Icon('./images/shangda2.ico', wx.BITMAP_TYPE_ICO)
        self.SetIcon(self.icon)
        # 系统事件
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        print("wxpython界面初始化加载完成！")

        """参数"""
        # 默认为摄像头0
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False  # False未打开摄像头，True摄像头已打

        self.EYE_AR_CONSEC_FRAMES = 3
        self.MOUTH_AR_CONSEC_FRAMES = 5
        self.LONG_EYE_AR_CONSEC_FRAMES = 15

        self.long_EyeClose = 0
        self.EyeClose = 0
        self.blink_count = 0
        self.MouthOpen = 0
        self.eyetimes = 2
        self.yawm_count = 0
        self.alltimes = 23
        self.mask = False
        self.k = 0

class main_app(wx.App):
    """
     在OnInit() 里边申请Frame类，这样能保证一定是在app后调用，
     这个函数是app执行完自己的__init__函数后就会执行
    """
    # OnInit 方法在主事件循环开始前被wxPython系统调用，是wxpython独有的
    def OnInit(self):
        self.frame = Product_detecting(parent=None,title="Product_Detector")
        self.frame.Show(True)
        return True

if __name__ == "__main__":
    app = main_app()
    app.MainLoop()