#!/bin/bash

# 设置路径
PROGRAM_PATH="/home/cartin/BEVPlace-main/datasets/build/pointcloud_to_bev"  
BIN_ROOT_PATH="/home/cartin/BEVPlace-main/datasets/PSA_PCD"              
OUTPUT_ROOT_PATH="/home/cartin/BEVPlace-main/datasets/PSA_IMG"        

# 检查可执行文件是否存在
if [ ! -f "$PROGRAM_PATH" ]; then
    echo "可执行文件 $PROGRAM_PATH 不存在，请检查路径。"
    exit 1
fi

# 创建输出根目录（如果不存在）
mkdir -p "$OUTPUT_ROOT_PATH"

# # 遍历所有 group 目录
# for group_dir in "${BIN_ROOT_PATH}"/group_*; do
#     # 检查是否为目录
#     if [ ! -d "$group_dir" ]; then
#         continue
#     fi

# group_name=$(basename "$group_dir")
# echo "${BIN_ROOT_PATH}"
# pointcloud_dir= "${BIN_ROOT_PATH}"
# echo "${pointcloud_dir}"
time_csv_path="${BIN_ROOT_PATH}/time.csv"  # 生成 CSV 文件路径
gt_csv_path="${BIN_ROOT_PATH}/path_PSA_0325_9to11.csv"  # 生成 CSV 文件路径
# output_dir="${OUTPUT_ROOT_PATH}/${group_name}"

# 检查 PointCloud 目录是否存在
if [ ! -d "$BIN_ROOT_PATH" ]; then
    echo "警告: PointCloud 目录 $BIN_ROOT_PATH 不存在，跳过 $group_name。"
    continue
fi

# 检查 CSV 文件是否存在
if [ ! -f "$time_csv_path" ]; then
    echo "Wrong Time Path"
    continue
fi

if [ ! -f "$gt_csv_path" ]; then
    echo "Wrong Ground Truth Path"
    continue
fi

# 创建对应的输出子目录
# mkdir -p "$OUTPUT_ROOT_PATH"

# echo "正在处理 $group_name ..."

# 运行转换程序
"$PROGRAM_PATH" "$BIN_ROOT_PATH" "$time_csv_path" "$gt_csv_path" "$OUTPUT_ROOT_PATH"

# 检查程序是否成功执行
if [ $? -ne 0 ]; then
    echo "错误: 处理 $group_name 失败。"
else
    echo "成功: 处理 $group_name 完成。"
fi
# done

echo "所有组别处理完成。"
