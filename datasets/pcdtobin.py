import os
import numpy as np
import argparse
import open3d as o3d
from datetime import datetime

# PCD to array
def read_pcd(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    if not pcd.has_points():
        raise ValueError(f"No points found in {pcd_file}")
    
    point_cloud = np.asarray(pcd.points, dtype=np.float32)
    
    return point_cloud  # 仅返回 x, y, z


# Save pointcloud to KITTI bin
def save_kitti_bin(point_cloud, bin_file):
    # 只保存 x, y, z
    kitti_point_cloud = point_cloud[:, :3]
    kitti_point_cloud.tofile(bin_file)


# 生成纯数字时间戳
def generate_numeric_timestamp(format_type='unix'):
    if format_type == 'unix':
        # Unix 时间戳，整数形式
        return int(datetime.now().timestamp())
    elif format_type == 'formatted':
        # 格式化时间戳，例如：20240427154530
        return int(datetime.now().strftime('%Y%m%d%H%M%S'))
    else:
        raise ValueError("Unsupported timestamp format type. Use 'unix' or 'formatted'.")

def main():
    parser = argparse.ArgumentParser(description="Convert PCD files to KITTI BIN format with grouping and numeric timestamps.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input PCD folder.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output BIN folder.')
    parser.add_argument('--group_size', type=int, default=300, help='Number of BIN files per group.')
    parser.add_argument('--timestamp_format', type=str, default='unix', choices=['unix', 'formatted'], help="Timestamp format: 'unix' for Unix timestamp, 'formatted' for YYYYMMDDHHMMSS.")
    
    args = parser.parse_args()
    
    pcd_folder = args.input
    kitti_bin_folder = args.output
    group_size = args.group_size
    timestamp_format = args.timestamp_format
    
    if not os.path.exists(kitti_bin_folder):
        os.makedirs(kitti_bin_folder)
        print(f"Created output directory: {kitti_bin_folder}")
    
    # 初始化分组变量
    group_index = 1
    current_group_count = 0
    current_group_info = []
    
    # 创建第一个分组的子文件夹及 PointCloud 文件夹
    current_group_folder = os.path.join(kitti_bin_folder, f"group_{group_index}")
    pointcloud_folder = os.path.join(current_group_folder, "PointCloud")
    os.makedirs(pointcloud_folder, exist_ok=True)
    print(f"Processing files into folder: {pointcloud_folder}")
    
    # 转换所有 PCD 文件
    for filename in sorted(os.listdir(pcd_folder)):
        if filename.endswith('.pcd'):
            pcd_file = os.path.join(pcd_folder, filename)
            bin_filename = os.path.splitext(filename)[0] + '.bin'
            bin_file = os.path.join(pointcloud_folder, bin_filename)
            
            try:
                # 读取 PCD 文件
                point_cloud = read_pcd(pcd_file)
                
                # 保存为 BIN 文件
                save_kitti_bin(point_cloud, bin_file)
                
                # 生成纯数字时间戳
                timestamp = generate_numeric_timestamp(format_type=timestamp_format)
                
                # 记录 BIN 文件名和时间戳
                current_group_info.append({'bin_file': bin_filename, 'timestamp': timestamp})
                
                # 打印转换信息
                print(f"Converted {filename} to {bin_filename} with timestamp {timestamp}")
                
                # 更新计数器
                current_group_count += 1
                
                # 检查是否达到分组大小
                if current_group_count >= group_size:
                    # 生成 CSV 文件
                    csv_filename = f"group_{group_index}_timestamps.csv"
                    csv_file_path = os.path.join(current_group_folder, csv_filename)
                    
                    with open(csv_file_path, 'w') as csv_file:
                        csv_file.write("bin_file,timestamp\n")
                        for info in current_group_info:
                            csv_file.write(f"{info['bin_file']},{info['timestamp']}\n")
                    
                    print(f"Created CSV file: {csv_file_path}")
                    
                    # 重置分组变量
                    group_index += 1
                    current_group_count = 0
                    current_group_info = []
                    
                    # 创建新的分组文件夹及 PointCloud 文件夹
                    current_group_folder = os.path.join(kitti_bin_folder, f"group_{group_index}")
                    pointcloud_folder = os.path.join(current_group_folder, "PointCloud")
                    os.makedirs(pointcloud_folder, exist_ok=True)
                    print(f"Processing files into folder: {pointcloud_folder}")
            
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")
    
    # 处理最后一组不足 group_size 的文件
    if current_group_info:
        csv_filename = f"group_{group_index}_timestamps.csv"
        csv_file_path = os.path.join(current_group_folder, csv_filename)
        
        with open(csv_file_path, 'w') as csv_file:
            csv_file.write("bin_file,timestamp\n")
            for info in current_group_info:
                csv_file.write(f"{info['bin_file']},{info['timestamp']}\n")
        
        print(f"Created CSV file for the last group: {csv_file_path}")
    
    print("Conversion complete.")

if __name__ == "__main__":
    main()
