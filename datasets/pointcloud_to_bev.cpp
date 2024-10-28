#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <filesystem>
#include <chrono>
#include <stdexcept>
#include <cfloat>
#include <cmath>
#include <sstream>  // 添加此头文件用于格式化时间戳

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

// 定义pad_size
const int pad_size = 32;

// 结构体用于存储BVFT描述符（示例，具体实现根据需求调整）
struct BVFT {
    Mat descriptors;
    // 可以添加其他成员变量
};

// 函数声明
int readPointCloud(pcl::PointCloud<pcl::PointXYZ> &point_cloud, const std::string &filename);
map<string, string> readCSV(const string &csv_path);
int imagePadding(Mat& img, int &cor_x, int &cor_y);
void generateImage(pcl::PointCloud<pcl::PointXYZ> &point_cloud, float resolution, Mat &mat_local_image);
BVFT detectBVFT(const Mat &img); // 假设此函数已实现
void writeMatToImage(const cv::Mat &mat, const std::string &filename);

// 函数实现

// 读取点云数据
int readPointCloud(pcl::PointCloud<pcl::PointXYZ> &point_cloud, const std::string &filename)
{
    point_cloud.clear();
    std::ifstream binfile(filename, std::ios::binary);
    if (!binfile)
    {
        throw std::runtime_error("无法打开文件: " + filename);
        return -1;
    }

    // 读取BIN文件，每个点由三个float组成
    while (true)
    {
        float x, y, z;

        // 读取x
        binfile.read(reinterpret_cast<char*>(&x), sizeof(float));
        if (binfile.eof()) break;

        // 读取y
        binfile.read(reinterpret_cast<char*>(&y), sizeof(float));
        if (binfile.eof()) break;

        // 读取z
        binfile.read(reinterpret_cast<char*>(&z), sizeof(float));
        if (binfile.eof()) break;

        pcl::PointXYZ point;
        point.x = x;
        point.y = y;
        point.z = z;
        point_cloud.push_back(point);
    }

    binfile.close();

    // 打印点云信息
    std::cout << "读取 " << filename << " 完成，点数: " << point_cloud.size() << std::endl;
    if (!point_cloud.empty())
    {
        std::cout << "示例点: (" 
                  << point_cloud.points[0].x << ", " 
                  << point_cloud.points[0].y << ", " 
                  << point_cloud.points[0].z << ")" << std::endl;
    }

    return 1;
}

// 读取CSV文件，将时间戳映射到BIN文件名
map<string, string> readCSV(const string &csv_path)
{
    map<string, string> timestamp_to_bin;
    std::ifstream csv_file(csv_path);
    if (!csv_file.is_open())
    {
        throw std::runtime_error("无法打开CSV文件: " + csv_path);
    }

    string line;
    // 如果CSV文件没有表头，可以注释掉以下代码
    // if (!getline(csv_file, line))
    //     throw std::runtime_error("CSV文件为空: " + csv_path);

    // 读取内容
    while (getline(csv_file, line))
    {
        if (line.empty())
            continue;
        size_t comma_pos = line.find(',');
        if (comma_pos == string::npos)
            continue;
        string timestamp = line.substr(0, comma_pos);
        string bin_file = line.substr(comma_pos + 1);
        timestamp_to_bin[timestamp] = bin_file;
    }

    csv_file.close();
    return timestamp_to_bin;
}

// 图像填充函数
int imagePadding(Mat& img, int &cor_x, int &cor_y)
{
    // 初始边界填充
    copyMakeBorder(img, img, pad_size/2, pad_size/2, pad_size/2, pad_size/2, BORDER_CONSTANT, Scalar(0));

    // 扩展图像到最佳DFT大小
    int m = getOptimalDFTSize(img.rows);
    int n = getOptimalDFTSize(img.cols);
    int row_pad = (m - img.rows) / 2;
    int col_pad = (n - img.cols) / 2;

    copyMakeBorder(img, img, row_pad, (m - img.rows) % 2 ? row_pad + 1 : row_pad,
                  col_pad, (n - img.cols) % 2 ? col_pad + 1 : col_pad, BORDER_CONSTANT, Scalar(0));

    cor_x += col_pad + pad_size / 2;
    cor_y += row_pad + pad_size / 2;

    return 0;
}

// 生成BEV图像
void generateImage(pcl::PointCloud<pcl::PointXYZ> &point_cloud, float resolution, Mat &mat_local_image)
{
    pcl::VoxelGrid<pcl::PointXYZ> down_size_filter;
    down_size_filter.setLeafSize(resolution, resolution, resolution / 2);
    down_size_filter.setInputCloud(point_cloud.makeShared());
    down_size_filter.filter(point_cloud);

    float x_min = FLT_MAX, y_min = FLT_MAX, x_max = -FLT_MAX, y_max = -FLT_MAX;
    for (const auto &point : point_cloud.points)
    {
        if (point.y < y_min) y_min = point.y;
        if (point.y > y_max) y_max = point.y;
        if (point.x < x_min) x_min = point.x;
        if (point.x > x_max) x_max = point.x;
    }

    float x_range = x_max - x_min;
    float y_range = y_max - y_min;
    // float max_range = max(x_range, y_range);
    float max_range = 80;

    // 设定目标图像尺寸
    int target_size = 201;
    // 计算分辨率以适应目标图像尺寸
    float computed_resolution = max_range / static_cast<float>(target_size);
    // 为确保覆盖范围，重新计算分辨率
    float final_resolution = computed_resolution;

    cout << "计算得到的分辨率: " << final_resolution << " 米/像素" << endl;

    // 重新过滤点云数据以匹配新的分辨率
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(final_resolution, final_resolution, final_resolution / 2);
    voxel_filter.setInputCloud(point_cloud.makeShared());
    voxel_filter.filter(point_cloud);

    // 重新计算范围
    x_min = FLT_MAX;
    y_min = FLT_MAX;
    x_max = -FLT_MAX;
    y_max = -FLT_MAX;
    for (const auto &point : point_cloud.points)
    {
        if (point.y < y_min) y_min = point.y;
        if (point.y > y_max) y_max = point.y;
        if (point.x < x_min) x_min = point.x;
        if (point.x > x_max) x_max = point.x;
    }

    // 创建固定大小的图像
    mat_local_image = Mat::zeros(target_size, target_size, CV_32F);

    for (const auto &point : point_cloud.points)
    {
        int ind_i = static_cast<int>(round(target_size/2.0 - point.x/final_resolution));
        int ind_j = static_cast<int>(round(target_size/2.0 - point.y/final_resolution));
        if (ind_i >= target_size || ind_j >= target_size || ind_i < 0 || ind_j < 0)
            continue;
        mat_local_image.at<float>(ind_j, ind_i) += 0.12f;
    }

    // 打印BEV图像信息
    int non_zero = countNonZero(mat_local_image);
    std::cout << "BEV图像大小: " << mat_local_image.rows << "x" << mat_local_image.cols 
              << ", 非零像素: " << non_zero << std::endl;

    // 可选：显示图像用于调试（仅处理单个 group 时使用）
    imshow("BEV Image", mat_local_image);
    waitKey(0);
}

// 假设的BVFT检测函数，实现根据需求调整
BVFT detectBVFT(const Mat &img)
{
    BVFT bvft;
    // TODO: 实现BVFT检测逻辑，这里仅作为示例
    bvft.descriptors = img.clone();
    return bvft;
}

// 将Mat保存为PNG图像
void writeMatToImage(const cv::Mat &mat, const std::string &filename)
{
    // 假设mat是单通道浮点数矩阵，需要转换为8位并归一化
    Mat mat_8u;
    mat.convertTo(mat_8u, CV_8U, 255.0);
    if (!imwrite(filename, mat_8u))
    {
        throw std::runtime_error("无法保存图像: " + filename);
    }
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "使用方法: " << argv[0] << " <bin_path> <csv_path> <output_dir>" << std::endl;
        return -1;
    }

    string bin_path = argv[1];
    string csv_path = argv[2];
    string output_dir = argv[3];

    // 创建输出目录
    if (!fs::exists(output_dir))
    {
        fs::create_directories(output_dir);
    }

    // 读取CSV文件
    map<string, string> timestamp_to_bin;
    try
    {
        timestamp_to_bin = readCSV(csv_path);
    }
    catch (const std::exception &e)
    {
        cerr << "读取CSV文件时出错: " << e.what() << endl;
        return -1;
    }

    // 将时间戳提取为vector，并进行排序
    vector<string> timestamps;
    for (const auto &pair : timestamp_to_bin)
    {
        timestamps.push_back(pair.first);
    }
    // 按照时间戳排序（需要将字符串转换为浮点数进行排序）
    sort(timestamps.begin(), timestamps.end(), [](const string &a, const string &b) {
        return std::stod(a) < std::stod(b);
    });

    cout << "BIN文件数量: " << timestamps.size() << endl;

    auto t_start = chrono::steady_clock::now();

    // 定义每组的BIN文件数量
    const int bins_per_group = 20;

    // 计算总组数
    int total_groups = (timestamps.size() + bins_per_group - 1) / bins_per_group;

    for (int group_idx = 0; group_idx < total_groups; group_idx++)
    {
        // 定义当前组的起始和结束索引
        int start_idx = group_idx * bins_per_group;
        int end_idx = min(start_idx + bins_per_group, (int)timestamps.size());

        // 合并点云数据
        pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        string last_timestamp = "unknown";

        for (int i = start_idx; i < end_idx; i++)
        {
            string timestamp = timestamps[i];
            string bin_file = timestamp_to_bin[timestamp];
            string bin_file_full_path = fs::path(bin_path) / bin_file;

            try
            {
                // 读取点云数据
                pcl::PointCloud<pcl::PointXYZ> point_cloud;
                readPointCloud(point_cloud, bin_file_full_path);

                // 检查点云是否为空
                if (point_cloud.empty())
                {
                    std::cerr << "警告: 点云数据为空，文件: " << bin_file_full_path << std::endl;
                    continue;
                }

                // 合并点云
                *merged_cloud += point_cloud;

                // 获取时间戳（使用最后一个BIN文件的时间戳）
                last_timestamp = timestamp;

            }
            catch (const std::exception &e)
            {
                cerr << "读取BIN文件失败 " << bin_file_full_path << ": " << e.what() << endl;
                continue;
            }
        }

        // 检查是否有有效点云
        if (merged_cloud->empty())
        {
            std::cerr << "警告: 组 " << group_idx + 1 << " 没有有效的点云数据。" << std::endl;
            continue;
        }

        // 计算分辨率并生成BEV图像
        // 首先，计算点云的物理范围
        float x_min = FLT_MAX, y_min = FLT_MAX, x_max = -FLT_MAX, y_max = -FLT_MAX;
        for (const auto &point : merged_cloud->points)
        {
            if (point.y < y_min) y_min = point.y;
            if (point.y > y_max) y_max = point.y;
            if (point.x < x_min) x_min = point.x;
            if (point.x > x_max) x_max = point.x;
        }

        float x_range = x_max - x_min;
        float y_range = y_max - y_min;
        // float max_range = max(x_range, y_range);
        float max_range = 100;

        // 设定目标图像尺寸
        int target_size = 201;
        // 计算分辨率
        float resolution = max_range / static_cast<float>(target_size);
        cout << "Group " << group_idx + 1 << ": x_range=" << x_range << ", y_range=" << y_range 
             << ", max_range=" << max_range << ", resolution=" << resolution << " 米/像素" << endl;

        // 生成BEV图像
        Mat mat_local_image;
        generateImage(*merged_cloud, resolution, mat_local_image);

        // 填充图像
        int cor_x = 0, cor_y = 0;
        imagePadding(mat_local_image, cor_x, cor_y);

        // 检测BVFT特征（示例）
        BVFT bvft = detectBVFT(mat_local_image);

        // 生成输出文件名
        // 格式化时间戳
        std::ostringstream oss;
        oss.precision(6);
        oss << std::fixed << std::stod(last_timestamp);
        string formatted_timestamp = oss.str();

        string output_filename = "bev_group_" + to_string(group_idx + 1) + "_timestamp_" + formatted_timestamp + ".png";
        string output_full_path = fs::path(output_dir) / output_filename;

        // 保存描述符为PNG图像
        try
        {
            writeMatToImage(bvft.descriptors, output_full_path);

            cout << "成功保存 BEV 图像: " << output_full_path << endl;
        }
        catch (const std::exception &e)
        {
            cerr << "保存图像失败 " << output_full_path << ": " << e.what() << endl;
            continue;
        }
    }

    auto t_end = chrono::steady_clock::now();
    chrono::duration<float> time_used = chrono::duration_cast<chrono::duration<float>>(t_end - t_start);
    cout << "转换完成，耗时 " << time_used.count() << " 秒" << endl;

    return 0;
}
