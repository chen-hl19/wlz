import cv2
import numpy as np
import os

# 输入和输出文件夹路径
input_folder = 'butterfly_mask'  # 包含原始图片的文件夹
output_folder = 'butterfly_mask'  # 保存掩码的文件夹

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 只处理PNG图片且不是掩码文件
    if filename.lower().endswith('.png') and not filename.lower().endswith('_mask.png'):
        image_path = os.path.join(input_folder, filename)
        
        # 读取图像（包含Alpha通道）
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if image is not None and image.shape[2] == 4:
            alpha_channel = image[:, :, 3]
            mask = cv2.bitwise_not(alpha_channel)  # 反转alpha通道作为掩码
            
            # 生成掩码文件名
            base_name = os.path.splitext(filename)[0]
            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(output_folder, mask_filename)
            
            # 保存掩码
            cv2.imwrite(mask_path, mask)
            print(f"已生成掩码: {mask_filename}")
            
            # 删除原图
            os.remove(image_path)
            print(f"已删除原图: {filename}")
        else:
            print(f"跳过 {filename} - 没有Alpha通道或不是4通道图像")

print("所有图片处理完毕！原图已删除，只保留掩码。")