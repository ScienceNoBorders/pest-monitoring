import cv2
from ultralytics import YOLO

from flask import Flask, jsonify, request
import json

from PIL import Image, ImageDraw, ImageFont
import numpy as np

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = "application/json;charset=utf-8"

model = YOLO('best.pt')
# model = YOLO('/Users/xiaotijun/Documents/workspace/xinjiang/document/smart-agriculture/code/ultralytics/runs/detect/train2/weights/best.pt')

@app.route('/')
def index():
    return '欢迎使用虫情识别工具！'

def crop_to_circle(image_path, output_path, circle_diameter=None):
    """
    从图像中截取中间的圆形区域并保存。

    :param image_path: 输入图像的路径
    :param output_path: 输出图像的路径
    :param circle_diameter: 圆形的直径。如果为None，则默认为最小边的一半
    """
    # 打开图片
    image = Image.open(image_path).convert("RGBA")

    # 计算圆心和半径
    center = (image.width // 2, image.height // 2)
    if circle_diameter is None:
        radius = min(image.width, image.height) // 2  # 默认圆的直径为最小边的一半
    else:
        radius = circle_diameter // 2

    # 创建一个与原图大小相同的遮罩图像，初始化为全透明
    mask = Image.new('L', image.size, 0)

    # 创建一个绘制对象
    draw = ImageDraw.Draw(mask)

    # 绘制白色填充的圆形
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)

    # 使用遮罩将原图中的圆形区域保留，其余部分变为透明
    # result = ImageOps.fit(image, mask.size, centering=(0.5, 0.5))
    result = Image.new("RGBA", image.size)
    result.paste(image, (0, 0), mask=mask)
    result.putalpha(mask)

    # 保存结果图像
    result.save(output_path, format="PNG")
    print("保存成功！")

def make_circle(image_path, output_path):
    # 打开图片
    img = Image.open(image_path).convert("RGBA")

    # 计算圆心和半径
    width, height = img.size
    radius = min(width, height) // 2
    center = (width // 2, height // 2)

    # 创建一个圆形遮罩
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)

    # 应用遮罩
    result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    result.paste(img, (0, 0), mask)

    # 裁剪图片到最小的正方形区域
    bbox = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
    cropped_img = result.crop(bbox)

    # 保存结果
    cropped_img.save(output_path, format="PNG")
    print("保存成功！")

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

@app.route('/predictImage/<param>')
def predictImage(path):
    result = {"code": 200}
    names = ['草蛉科', '九香虫', '小飞虫', '甲虫', '昆虫', '蝗虫', '飞蛾', '蚊子']
    # 设置字体，这里需要你有中文字体文件，如果没有则需要下载或使用支持中文的字体
    font = ImageFont.truetype("Arial Unicode.ttf", 18)
    pairs_tuple = ()
    insect_detail = []
    output_image_path = 'pic/predict1.png'
    try:
        output_path = 'tailor_image.png'

        # (3000, 3000)裁剪尺寸
        make_circle(path, output_path)
        # crop_to_circle(path, output_path, 3000)
        print("开始识别！")
        image = cv2.imread(output_path)
        # 进行预测
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用模型进行推理
        results = model(image_rgb)

        # 获取详细的结果信息进行进一步处理
        boxes = results[0].boxes  # 获取第一个结果的boxes信息

        # 在图像上绘制标注框和标签
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取标注框坐标并转换为整数
            conf = box.conf[0]  # 获取置信度
            cls = box.cls[0]  # 获取类别
            label = f'{model.names[int(cls)]}: {conf:.2f}'  # 创建标签

            # 绘制标注框
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制标签背景
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(y1, label_size[1])
            cv2.rectangle(image_rgb, (x1, top - label_size[1]), (x1 + label_size[0], top + base_line), (0, 255, 0),
                          cv2.FILLED)
            # 绘制标签文本
            # cv2.putText(image_rgb, label, (x1, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            pair = (f'{names[int(cls)]}: {conf:.2f}', (x1, top-17), names[int(cls)], int(cls), conf)
            pairs_tuple += (pair,)
            insect_detail.append({
                'name_en': model.names[int(cls)],
                'name_cn': names[int(cls)],
                'type': int(cls),
                'confidence': f'{conf:.2f}'
            })

        # 将OpenCV图像转换成PIL图像
        image_pil = Image.fromarray(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))

        # 准备绘画对象
        draw = ImageDraw.Draw(image_pil)

        for key, value in pairs_tuple:
            # 绘制中文文本
            draw.text(value, key, font=font, fill=(0, 0, 0))
        # 将PIL图像转换回OpenCV图像
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_image_path, image_bgr)
    except Exception as ex:
        result['msg'] = ex

    result['msg'] = '扫描成功！'
    result['data'] = {
        'insectNumber': len(pairs_tuple),
        'indentifyPicUrl': output_image_path,
        'insectDetail': insect_detail,
    }
    insectJson = json.dumps(result, ensure_ascii=False)
    return jsonify(json.loads(insectJson))

@app.route('/actionPredict')
def actionPredict():
    path = request.args.get('path', '')
    return predictImage(path)

def testActionPredict(path):
    return predictImage(path)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )



