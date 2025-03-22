import cv2
import os  # 新增导入

# 全局变量，用于存储鼠标选择的区域
roi_selected = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)

# 鼠标回调函数
def select_roi(event, x, y, flags, param):
    global roi_selected, top_left_pt, bottom_right_pt

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下，记录起点
        top_left_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，记录终点
        bottom_right_pt = (x, y)
        roi_selected = True

# 读取图像
image_folder = '../2418566-3-/'  # 指定图像文件夹
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.bmp', '.jpg', '.png'))]  # 获取所有图像文件
output_folder = './images2/'  # 指定输出文件夹
for image_file in image_files:  # 遍历每个图像文件
    image = cv2.imread(os.path.join(image_folder, image_file))  # 读取图像

    # 创建窗口并绑定鼠标回调函数
    cv2.namedWindow('Select ROI',cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cropped ROI', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select ROI', select_roi)
    while True:
        # 显示图像
        display_image = image.copy()

        # 如果选择了区域，绘制矩形
        if top_left_pt != (-1, -1) and bottom_right_pt != (-1, -1):
            cv2.rectangle(display_image, top_left_pt, bottom_right_pt, (0, 255, 0), 10)

        cv2.imshow('Select ROI', display_image)

        # 按下 'c' 键切割选中区域
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and roi_selected:
            break

    # 切割选中区域
    if roi_selected:
        x1, y1 = top_left_pt
        x2, y2 = bottom_right_pt
        roi = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]  # 确保坐标顺序正确

        # 显示切割后的区域
        # cv2.imshow('Cropped ROI', roi)
        # cv2.waitKey(0)

        # 可选：保存切割后的区域
        output_file_path = os.path.join(output_folder, f'{image_files.index(image_file) + 1}.png')  # 生成文件名
        cv2.imwrite(output_file_path, roi)  # 保存图像

# 关闭所有窗口
cv2.destroyAllWindows()