from PIL import Image, ImageDraw, ImageFilter
import random


def generate_handwritten_digit(digit, save_path=None):
    if not isinstance(digit, int) or digit < 0 or digit > 9:
        raise ValueError("digit 必须是 0 到 9 的整数")

    # 创建 28x28 的 8-bit 灰度图，黑底白字
    img = Image.new("L", (28, 28), 0)
    draw = ImageDraw.Draw(img)

    # 为了让图像更像手写体，给每个数字设置不同的基本笔画
    if digit == 0:
        draw.ellipse((7, 4, 20, 24), outline=255, width=2)

    elif digit == 1:
        draw.line((14, 5, 14, 23), fill=255, width=2)
        draw.line((11, 8, 14, 5), fill=255, width=2)
        draw.line((11, 23, 17, 23), fill=255, width=2)

    elif digit == 2:
        draw.arc((6, 4, 21, 14), start=0, end=180, fill=255, width=2)
        draw.line((20, 12, 8, 23), fill=255, width=2)
        draw.line((8, 23, 21, 23), fill=255, width=2)

    elif digit == 3:
        draw.arc((7, 4, 20, 14), start=270, end=90, fill=255, width=2)
        draw.arc((7, 13, 20, 24), start=270, end=90, fill=255, width=2)

    elif digit == 4:
        draw.line((18, 5, 18, 23), fill=255, width=2)
        draw.line((8, 16, 21, 16), fill=255, width=2)
        draw.line((8, 16, 17, 5), fill=255, width=2)

    elif digit == 5:
        draw.line((8, 5, 20, 5), fill=255, width=2)
        draw.line((8, 5, 8, 14), fill=255, width=2)
        draw.line((8, 14, 18, 14), fill=255, width=2)
        draw.arc((7, 13, 20, 24), start=270, end=120, fill=255, width=2)

    elif digit == 6:
        draw.arc((7, 4, 20, 24), start=45, end=340, fill=255, width=2)
        draw.line((18, 15, 10, 15), fill=255, width=2)

    elif digit == 7:
        draw.line((7, 5, 21, 5), fill=255, width=2)
        draw.line((21, 5, 11, 23), fill=255, width=2)

    elif digit == 8:
        draw.ellipse((8, 3, 20, 13), outline=255, width=2)
        draw.ellipse((7, 12, 21, 25), outline=255, width=2)
        draw.line((10, 12, 18, 12), fill=255, width=2)

    elif digit == 9:
        draw.arc((7, 4, 20, 16), start=0, end=360, fill=255, width=2)
        draw.line((19, 10, 12, 23), fill=255, width=2)

    # 轻微模糊，让边缘更像手写数字
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    if save_path is None:
        save_path = f"digit_{digit}_28x28.png"

    img.save(save_path)
    return save_path


if __name__ == "__main__":
    n = int(input("请输入一个数字(0-9): "))
    path = generate_handwritten_digit(n)
    print(f"已生成图片: {path}")