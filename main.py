# This is a sample Python script.
import cv2
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


from PIL import Image, ImageDraw

import YoloRecognition


def crop_circle_and_square(image_path, output_path):
    # 加载图像
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 获取图像的宽度和高度
    height, width = img.shape[:2]

    # 定义圆的中心和半径
    center = (width // 2, height // 2)
    radius = min(width, height) // 2

    # 创建一个黑色遮罩
    mask = np.zeros((height, width), dtype=np.uint8)

    # 在遮罩上绘制白色的圆形区域
    cv2.circle(mask, center, radius, 255, -1)

    # 创建一个全白色背景
    white_background = np.full_like(img, 255)

    # 将圆形区域覆盖到白色背景上
    circle_img = cv2.bitwise_and(img, img, mask=mask)

    # 将白色背景应用到原图像中
    background = cv2.bitwise_not(mask)  # 反转遮罩，获取非圆形区域
    white_bg_img = cv2.bitwise_and(white_background, white_background, mask=background)

    # 合并白色背景和圆形图像
    final_img = cv2.add(white_bg_img, circle_img)

    # 裁剪到最小正方形区域
    x, y, w, h = center[0] - radius, center[1] - radius, 2 * radius, 2 * radius
    cropped_img = final_img[y:y + h, x:x + w]

    # 保存结果
    cv2.imwrite(output_path, cropped_img)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # crop_circle_and_square("/Users/xiaotijun/Pictures/insect/3.jpg", "new_tailor.png")
    # YoloRecognition.crop_to_circle('/Users/xiaotijun/Pictures/insect/3.jpg', 'tailor_image.png')
    YoloRecognition.testActionPredict('/Users/xiaotijun/Pictures/insect/3.jpg')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
