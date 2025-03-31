from PIL import Image, ImageOps

# 打开掩码图像
image_mask = Image.open('mask_image.png')

# 反转颜色
inverted_image_mask = ImageOps.invert(image_mask)

# 保存或显示结果
inverted_image_mask.save('inverted_image_mask.png')
inverted_image_mask.show()