import lpips
import torch
from PIL import Image
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_ssim(image1_path, image2_path):
    image1 = np.array(Image.open(image1_path).convert('RGB'))
    image2 = np.array(Image.open(image2_path).convert('RGB'))
    
    # 確保兩個圖像具有相同的尺寸
    assert image1.shape == image2.shape, "Images must have the same dimensions for SSIM."
    
    # 計算SSIM
    ssim_value = ssim(image1, image2, multichannel=True)
    return ssim_value

def compute_lpips_distance(image1_path, image2_path, net='alex'):
    # 初始化LPIPS模型
    lpips_model = lpips.LPIPS(net=net)
    image1_tensor = lpips.im2tensor(lpips.load_image(image1_path))
    image2_tensor = lpips.im2tensor(lpips.load_image(image2_path))
    
    # 計算LPIPS距離
    with torch.no_grad():
        distance = lpips_model.forward(image1_tensor, image2_tensor)
    
    return distance.item()

def compare_handwritings(folder_path, my_handwriting_path):
    results = []  # 用於儲存每個文件的LPIPS和SSIM值
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        if image_path != my_handwriting_path:
            lpips_distance = compute_lpips_distance(my_handwriting_path, image_path)
            ssim_value = compute_ssim(my_handwriting_path, image_path)
            results.append((image_name, lpips_distance, ssim_value))
    
    # 將結果轉換為DataFrame並按LPIPS距離排序
    df = pd.DataFrame(results, columns=['Student', 'LPIPS', 'SSIM'])
    df.sort_values(by='LPIPS', inplace=True)
    
    return df

# 使用您提供的路徑
folder_path = 'D:/the/folder/path/of/the/word'
my_handwriting_path = 'D:/the/image/path/of/yours'

# 比較手寫字圖片並生成表格
df = compare_handwritings(folder_path, my_handwriting_path)
print(df)

output_csv_path = 'D:/the/place/to/save/excel'
df.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")