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
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

// 定义pad_size
const int pad_size = 32;
Eigen::Affine3d B_T_L = Eigen::Affine3d::Identity();

// 结构体用于存储BVFT描述符（示例，具体实现根据需求调整）
struct BVFT {
    Mat descriptors;
    // 可以添加其他成员变量
};

// 函数声明
std::vector<double> readTimeCSV(const string &csv_path);
std::vector<std::vector<double>> readGTCSV(const string &gt_csv_path);
int imagePadding(Mat& img, int &cor_x, int &cor_y);
void generateImage(pcl::PointCloud<pcl::PointXYZ> &point_cloud, float resolution, Mat &mat_local_image);
BVFT detectBVFT(const Mat &img); // 假设此函数已实现
void writeMatToImage(const cv::Mat &mat, const std::string &filename);

// 读取CSV文件
std::vector<double> readTimeCSV(const string &csv_path) {
    std::string line;
    std::ifstream file(csv_path, std::ios::in);

    std::vector<double> timestamp;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            double time = std::stod(line); // Convert the line to double
            timestamp.push_back(time);
        }
        file.close();
    } else std::cout << "Wrong Path: TimeStamp" << std::endl;

    return timestamp;
}

std::vector<std::vector<double>> readGTCSV(const string &gt_csv_path) {
    std::string line;
    std::ifstream file(gt_csv_path, std::ios::in);

    std::vector<std::vector<double>> gt_path;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::vector<double> row; // store data in row
            double value;
            // get data in each row
            while (iss >> value) {
                row.push_back(value);
                if (iss.peek() == ',') iss.ignore();
            }
            gt_path.push_back(row);
        }
        file.close();
    } else std::cout << "Wrong Path: Ground Truth" << std::endl;
    return gt_path;
}

// 在任意时间点进行插值
Eigen::Affine3d interpolate_at_time(const std::vector<std::vector<double>>& poses_with_time, double target_time) {
    for (size_t i = 0; i < poses_with_time.size()-1; ++i) {
        double current_time = poses_with_time[i][0]; // 假设时间戳递增
        double next_time = poses_with_time[i+1][0];

        if (target_time >= current_time && target_time <= next_time) {
            double alpha = (target_time - current_time) / (next_time - current_time);
            Eigen::Vector3d current_pos(poses_with_time[i][1], poses_with_time[i][2], poses_with_time[i][3]);
            Eigen::Vector3d next_pos(poses_with_time[i+1][1], poses_with_time[i][2], poses_with_time[i][3]);            
            Eigen::Vector3d targ_pos = current_pos + alpha * (next_pos - current_pos);
            Eigen::Quaterniond current_quat(poses_with_time[i][4], poses_with_time[i][5], poses_with_time[i][6], poses_with_time[i][7]); //w,x,y,z
            Eigen::Quaterniond next_quat(poses_with_time[i+1][4], poses_with_time[i+1][5], poses_with_time[i+1][6], poses_with_time[i+1][7]); //w,x,y,z
            
            auto targ_quat = current_quat.slerp(alpha, next_quat);
            Eigen::Affine3d pose = Eigen::Affine3d::Identity();
            pose.translate(targ_pos);
            pose.rotate(targ_quat);
            return pose;
        }
    }
    return Eigen::Affine3d::Identity();
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

    // cor_x += col_pad + pad_size / 2;
    // cor_y += row_pad + pad_size / 2;

    return 0;
}

// 生成BEV图像
template <typename PointT>
void generateImage(pcl::PointCloud<PointT> &point_cloud, float resolution, Mat &mat_local_image)
{
    pcl::VoxelGrid<PointT> down_size_filter;
    down_size_filter.setLeafSize(0.2, 0.2, 0.2);
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

    // // 重新过滤点云数据以匹配新的分辨率
    // pcl::VoxelGrid<PointT> voxel_filter;
    // voxel_filter.setLeafSize(final_resolution, final_resolution, final_resolution / 2);
    // voxel_filter.setInputCloud(point_cloud.makeShared());
    // voxel_filter.filter(point_cloud);

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
    mat_local_image = Mat::zeros(target_size, target_size, CV_8UC1);

    for (const auto &point : point_cloud.points)
    {
        int ind_i = static_cast<int>(round(target_size/2.0 - point.x/final_resolution));
        int ind_j = static_cast<int>(round(target_size/2.0 - point.y/final_resolution));
        if (ind_i >= target_size || ind_j >= target_size || ind_i < 0 || ind_j < 0)
            continue;
        mat_local_image.at<uint8_t>(ind_j, ind_i) += 1;
        // int x_ind = x_max_ind-int((point_cloud.points[i].y)/resolution);
        // int y_ind = y_max_ind-int((point_cloud.points[i].x)/resolution);
        // if(x_ind>=x_num || y_ind>=y_num ) continue;
        // mat_local_image.at<uint8_t>( y_ind,x_ind) += 1;
    }

    uint8_t max_pixel = 0;
    for(int i=0; i<target_size; i++)
        for(int j=0; j<target_size; j++)
        {
            if (mat_local_image.at<uint8_t>(j, i)>max_pixel) max_pixel=mat_local_image.at<uint8_t>(j, i);
        }
    for(int i=0; i<target_size; i++) {
        for(int j=0; j<target_size; j++) {
            if(uint8_t(mat_local_image.at<uint8_t>(j, i)*10)>122) {mat_local_image.at<uint8_t>(j, i)=122;continue;}
            mat_local_image.at<uint8_t>(j, i)=uint8_t(mat_local_image.at<uint8_t>(j, i)*10);//1.0/max_pixel*255);
            if(uint8_t(mat_local_image.at<uint8_t>(j, i))==0) {mat_local_image.at<uint8_t>(j, i)=10;continue;}
        }
    }

    // 打印BEV图像信息
    // int non_zero = countNonZero(mat_local_image);
    // std::cout << "BEV图像大小: " << mat_local_image.rows << "x" << mat_local_image.cols 
    //           << ", 非零像素: " << non_zero << std::endl;

    // 可选：显示图像用于调试（仅处理单个 group 时使用）
    imshow("BEV Image", mat_local_image);
    waitKey(0);
}
template <typename PointT>
void generateImage(pcl::PointCloud<PointT> &point_cloud, int &x_max_ind, int &y_max_ind, Mat &mat_local_image)
{
    float resolution = 0.2;
    pcl::VoxelGrid<PointT> down_size_filter;
    down_size_filter.setLeafSize(resolution, resolution, resolution/2);
    down_size_filter.setInputCloud(point_cloud.makeShared());
    down_size_filter.filter(point_cloud);

    float x_min = -100, y_min = -100, x_max=100,y_max=100;
    // float x_min1, y_min1, x_max1, y_max1;
    // for(int i=0; i<point_cloud.size(); i++)
    // {
    //     if(point_cloud.points[i].y< x_min1) x_min1=point_cloud.points[i].y;
    //     if(point_cloud.points[i].y> x_max1) x_max1=point_cloud.points[i].y;
    //     if(point_cloud.points[i].x< y_min1) y_min1=point_cloud.points[i].x;
    //     if(point_cloud.points[i].x> y_max1) y_max1=point_cloud.points[i].x;
    // }

    
    int x_min_ind = int(x_min/resolution);
    x_max_ind = int(x_max/resolution);
    int y_min_ind = int(y_min/resolution);
    y_max_ind = int(y_max/resolution);
    // std::cout << x_min << ' ' << x_max << ' ' <<  y_min << ' ' << y_max << std::endl;
    // std::cout << x_min_ind << ' ' << x_max_ind << ' ' <<  y_min_ind << ' ' << y_max_ind << std::endl;
    // std::cout << int(x_min1/resolution) << ' ' << int(x_max1/resolution) << ' ' <<  int(y_min1/resolution) << ' ' << int(y_max1/resolution) << std::endl;

    int x_num = x_max_ind-x_min_ind+1;
    int y_num = y_max_ind-y_min_ind+1;
    // std::cout << x_num << y_num << std::endl;
    mat_local_image=Mat(y_num, x_num, CV_8UC1, cv::Scalar::all(0));

    for(int i=0; i<point_cloud.size(); i++)
    {
        // int x_ind = x_max_ind-int((point_cloud.points[i].y)/resolution);
        // int y_ind = y_max_ind-int((point_cloud.points[i].x)/resolution);
        // if(x_ind>=x_num || y_ind>=y_num ) continue;
        // mat_local_image.at<uint8_t>(y_ind, x_ind) += 1;
        int ind_i = static_cast<int>(round(y_num/2.0 - point_cloud.points[i].y/resolution));
        int ind_j = static_cast<int>(round(x_num/2.0 - point_cloud.points[i].x/resolution));
        if (ind_i >= y_num || ind_j >= x_num || ind_i < 0 || ind_j < 0)
            continue;
        mat_local_image.at<uint8_t>(ind_i, ind_j) += 1;
        //std::cout << int(mat_local_image.at<uint8_t>(x_ind, y_ind)) << std::endl;
    }
    // uint8_t max_pixel = 0;
    // for(int i=0; i<x_num; i++)
    //     for(int j=0; j<y_num; j++)
    //     {
    //         if (mat_local_image.at<uint8_t>(j, i)>max_pixel) max_pixel=mat_local_image.at<uint8_t>(j, i);
    //     }
    for(int i=0; i<x_num; i++) {
        for(int j=0; j<y_num; j++) {
            if(uint8_t(mat_local_image.at<uint8_t>(j, i)*10)>122) {mat_local_image.at<uint8_t>(j, i)=122;continue;}
            mat_local_image.at<uint8_t>(j, i)=uint8_t(mat_local_image.at<uint8_t>(j, i)*10);//1.0/max_pixel*255);
            if(uint8_t(mat_local_image.at<uint8_t>(j, i))==0) {mat_local_image.at<uint8_t>(j, i)=10;continue;}
        }
    }
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
    // Mat mat_8u;
    // mat.convertTo(mat_8u, CV_8U, 255.0);
    imwrite(filename, mat);
    // if (!imwrite(filename, mat_8u))
    // {
    //     throw std::runtime_error("无法保存图像: " + filename);
    // }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "使用方法: " << argv[0] << " <bin_path> <time_csv_path> <gt_csv_path> <output_dir>" << std::endl;
        return -1;
    }

    B_T_L.translate(Eigen::Vector3d(-0.006253, 0.011775, 0.03055));
    B_T_L.rotate(Eigen::Quaterniond(0,0,0,1)); //w,x,y,z

    string bin_path = argv[1];
    string time_csv_path = argv[2];
    string gt_csv_path = argv[3];
    string output_dir = argv[4];
    std::string output_pose_path = fs::path(output_dir) / "pose.txt";
    std::ofstream pose_path(output_pose_path);
    

    // 创建输出目录
    if (!fs::exists(output_dir))
    {
        fs::create_directories(output_dir);
    }

    // 读取CSV文件
    std::vector<double> timestamp = readTimeCSV(time_csv_path);
    std::vector<std::vector<double>> gt_path = readGTCSV(gt_csv_path);
    std::sort(gt_path.begin(), gt_path.end(), [](const std::vector<double> &a, const std::vector<double> &b) {
        return a[0] < b[0];
    });
    // std::cout << gt_path.size() << std::endl;
    // printf("%.6f\n",gt_path[182873][0]-timestamp[26982]);

    // 定义每组的BIN文件数量
    const int bins_per_group = 10;

    // 计算总组数
    int total_groups = (timestamp.size() + bins_per_group - 1) / bins_per_group;

    for (int group_idx = 0; group_idx < total_groups; group_idx++)
    {
        // 定义当前组的起始和结束索引
        int start_idx = group_idx * bins_per_group;
        int end_idx = min(start_idx + bins_per_group, (int)timestamp.size());

        // 合并点云数据
        Eigen::Affine3d pose_last = Eigen::Affine3d::Identity();
        pcl::PointCloud<pcl::PointXYZI>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        // std::cout << "Hell0" << std::endl;
        for (int i = start_idx; i < end_idx; i++) {
            // std::cout << "Hell-1" << std::endl;
            std::string pcd_file = bin_path + "/scans_" + std::to_string(i+1) + ".pcd";
            pcl::PointCloud<pcl::PointXYZI>::Ptr scan_pcd(new pcl::PointCloud<pcl::PointXYZI>);
            // std::cout << pcd_file << std::endl;
            if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *scan_pcd) == -1) {
                std::cout << "Couldn't read PCD file" << std::endl;
                continue;
            }
            // std::cout << "Hell-2" << std::endl;
            // std::cout << "Loaded " << scan_pcd->width * scan_pcd->height << " data points from PCD file." << std::endl;
            // 检查点云是否为空
            if (scan_pcd->empty()) {
                std::cerr <<  pcd_file << " is empty!" << std::endl;
                continue;
            }
            // std::cout << "Hell1" << std::endl;
            pose_last = interpolate_at_time(gt_path, timestamp[i]);
            // std::cout << "Hell2" << std::endl;
            // Eigen::Quaterniond quaternion(pose_last.rotation());
            // std::cout << pose_last.translation().transpose() << " " << quaternion.w() << std::endl;
            if (pose_last.translation() == Eigen::Vector3d(0,0,0)) continue;
            // std::cout << "Hell3" << std::endl;
            // pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud_body(new pcl::PointCloud<pcl::PointXYZI>);
            // // std::cout << "Hell4" << std::endl;
            // pcl::transformPointCloud(*scan_pcd, *current_cloud_body, B_T_L);
            // std::cout << "Hell5" << std::endl;

            pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud_world(new pcl::PointCloud<pcl::PointXYZI>);
            // std::cout << "Hell6" << std::endl;
            pcl::transformPointCloud(*scan_pcd, *current_cloud_world, pose_last*B_T_L);
            // std::cout << "Hell7" << std::endl;
            *merged_cloud += *current_cloud_world;
        }
        // std::cout << merged_cloud->size() << std::endl;

        // 检查是否有有效点云
        if (merged_cloud->empty()) {
            std::cerr << "警告: 组 " << group_idx + 1 << " 没有有效的点云数据。" << std::endl;
            continue;
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr merged_cloud_body(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*merged_cloud, *merged_cloud_body, pose_last.inverse());
        *merged_cloud = *merged_cloud_body;

        int max_local_x_ind,max_local_y_ind;
        // 生成BEV图像
        Mat mat_local_image;
        generateImage(*merged_cloud, max_local_x_ind, max_local_y_ind, mat_local_image);
        // 填充图像
        // int cor_x = 0, cor_y = 0;
        // imagePadding(mat_local_image, max_local_x_ind, max_local_y_ind);

        // 检测BVFT特征（示例）
        // BVFT bvft = detectBVFT(mat_local_image);

        std::string output_filename = std::to_string(group_idx + 1) + ".png";
        std::string output_bev_path = fs::path(output_dir) / output_filename;
        
        // 保存描述符为PNG图像
        try {
            writeMatToImage(mat_local_image, output_bev_path);
            cout << "Save BEV Image as: " << output_bev_path << endl;
            Eigen::Quaterniond quaternion(pose_last.rotation());
            if(pose_path.is_open()) {
                pose_path << pose_last.translation().x() << " " << pose_last.translation().y() << " " <<pose_last.translation().z()
                    << " " << quaternion.w() << " " << quaternion.x() << " " << quaternion.y() << " " << quaternion.z() << "\n";
            }

        }
        catch (const std::exception &e) {
            cerr << "Save BEV Image Failed" << output_bev_path << ": " << e.what() << endl;
            continue;
        }
    }
    pose_path.close();
    // auto t_end = chrono::steady_clock::now();
    // chrono::duration<float> time_used = chrono::duration_cast<chrono::duration<float>>(t_end - t_start);
    // cout << "转换完成，耗时 " << time_used.count() << " 秒" << endl;

    return 0;
}
