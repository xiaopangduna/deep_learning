import os
import time
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import hashlib
import shutil

def get_image_hash(img_path):
    """计算图片内容的MD5哈希值，用于生成唯一的npy文件名"""
    with open(img_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:16]

def prepare_npy_files(image_paths, cache_dir='npy_cache'):
    """将所有图片解码后保存为npy文件"""
    os.makedirs(cache_dir, exist_ok=True)
    npy_paths = []
    
    print("正在准备npy缓存文件...")
    for img_path in tqdm(image_paths, desc="转换图片为npy"):
        # 读取并解码图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片 {img_path}，已跳过")
            continue
            
        # 生成唯一的npy文件名
        img_hash = get_image_hash(img_path)
        npy_path = os.path.join(cache_dir, f"{img_hash}.npy")
        
        # 保存为npy文件
        np.save(npy_path, img)
        npy_paths.append(npy_path)
    
    return npy_paths

def test_image_reading(image_paths, repeat=10):
    """测试读取并解码图片的速度"""
    total_time = 0.0
    
    print("\n开始测试图片读取速度...")
    for i in range(repeat):
        start_time = time.perf_counter()
        
        # 读取并解码所有图片
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图片 {img_path}")
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        total_time += elapsed
        print(f"图片测试轮次 {i+1}/{repeat}: {elapsed:.4f} 秒")
    
    avg_time = total_time / repeat
    avg_per_image = (avg_time / len(image_paths)) * 1000  # 转换为毫秒/张
    return avg_time, avg_per_image

def test_npy_reading(npy_paths, repeat=10):
    """测试读取npy文件的速度"""
    total_time = 0.0
    
    print("\n开始测试npy文件读取速度...")
    for i in range(repeat):
        start_time = time.perf_counter()
        
        # 读取所有npy文件
        for npy_path in npy_paths:
            img = np.load(npy_path)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        total_time += elapsed
        print(f"npy测试轮次 {i+1}/{repeat}: {elapsed:.4f} 秒")
    
    avg_time = total_time / repeat
    avg_per_image = (avg_time / len(npy_paths)) * 1000  # 转换为毫秒/张
    return avg_time, avg_per_image

def main(image_dir, repeat=10, cleanup=False):
    # 获取所有图片路径
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(image_dir, ext)))
    
    if not image_paths:
        print(f"错误: 在目录 {image_dir} 中未找到任何图片文件")
        return
    
    print(f"找到 {len(image_paths)} 张图片，准备开始测试...")
    
    # 准备npy文件
    npy_paths = prepare_npy_files(image_paths)
    
    if not npy_paths:
        print("错误: 未能生成任何npy文件，无法进行测试")
        return
    
    # 运行测试
    img_total, img_per = test_image_reading(image_paths, repeat)
    npy_total, npy_per = test_npy_reading(npy_paths, repeat)
    
    # 显示结果
    print("\n" + "="*50)
    print(f"测试结果 (重复 {repeat} 次，共 {len(image_paths)} 张图片):")
    print(f"图片读取总平均时间: {img_total:.4f} 秒")
    print(f"单张图片平均读取+解码时间: {img_per:.4f} 毫秒")
    print(f"npy文件读取总平均时间: {npy_total:.4f} 秒")
    print(f"单张npy平均读取时间: {npy_per:.4f} 毫秒")
    print(f"npy比图片快: {img_per/npy_per:.2f} 倍")
    print("="*50 + "\n")
    
    # 清理缓存文件（如果需要）
    if cleanup:
        print("正在清理临时npy文件...")
        shutil.rmtree('npy_cache', ignore_errors=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试图片与npy文件读取速度差异')
    parser.add_argument('image_dir', help='存放图片的目录路径')
    parser.add_argument('--repeat', type=int, default=10, help='测试重复次数，默认10次')
    parser.add_argument('--cleanup', action='store_true', help='测试结束后清理生成的npy文件')
    
    args = parser.parse_args()
    main(args.image_dir, args.repeat, args.cleanup)
    