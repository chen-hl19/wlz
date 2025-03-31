from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import os
from tqdm import tqdm

# 初始化模型
pipe = StableDiffusionInpaintPipeline.from_single_file(
    "512-inpainting-ema.ckpt",  # .ckpt 文件路径
    torch_dtype=torch.float16,
)
pipe.to("cuda")

# 定义路径
input_folder = "butterfly_true"  # 原始图片文件夹
mask_folder = "butterfly_mask"   # 掩码图片文件夹（假设与原始图片同目录）
output_folder = "result"         # 输出文件夹

# 你的10个prompt
prompts = [
    "Dead-leaf butterfly on autumn leaves, camouflaged",
    "Butterfly blends into tree bark, natural",
    "Butterfly hidden in fallen leaves, subtle",
    "Butterfly mimics broken twigs, seamless",
    "Butterfly camouflaged in forest floor",
    "Butterfly among moss and dead leaves",
    "Butterfly on rotting wood, invisible",
    "Butterfly in shadows, barely visible",
    "Butterfly in ground cracks, hidden",
    "Butterfly blends into leaf litter"
]

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取所有原始图片文件（排除掩码文件）
image_files = [f for f in os.listdir(input_folder) 
               if f.endswith('.png') and not f.endswith('_mask.png')]

# 处理每张图片
for image_file in tqdm(image_files, desc="Processing images"):
    # 构建文件路径
    base_name = os.path.splitext(image_file)[0]
    image_path = os.path.join(input_folder, image_file)
    mask_path = os.path.join(mask_folder, f"{base_name}_mask.png")
    
    # 检查掩码是否存在
    if not os.path.exists(mask_path):
        print(f"Warning: Mask not found for {image_file}, skipping...")
        continue
    
    # 加载图像和掩码
    try:
        image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {image_file}: {e}")
        continue
    
    # 为每个prompt生成图像
    for i, prompt in enumerate(prompts, 1):
        # 生成图像
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
        ).images[0]
        
        # 保存结果
        output_filename = f"{base_name}_result_{i}.png"
        output_path = os.path.join(output_folder, output_filename)
        result.save(output_path)

print("All images processed successfully!")