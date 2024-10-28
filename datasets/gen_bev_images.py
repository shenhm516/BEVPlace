import numpy as np
import matplotlib.pyplot as plt
# import pcl
import open3d as o3d
import cv2
import os
import argparse
from tqdm import trange
from scipy.spatial.transform import Slerp, Rotation as R
parser = argparse.ArgumentParser(description='BEVPlace-Gen-BEV-Images')
parser.add_argument('--vel_path', type=str, default="/mnt/share_disk/KITTI/dataset/sequences/00/velodyne/", help='path to data')
parser.add_argument('--bev_save_path', type=str, default="./KITTI_new_imgs/00/imgs/", help='path to data')

def getBEV(all_points): #N*3
    
    all_points_pc = o3d.geometry.PointCloud()# pcl.PointCloud()
    all_points_pc.points = o3d.utility.Vector3dVector(all_points)#all_points_pc.from_array(all_points)
    all_points_pc = all_points_pc.voxel_down_sample(voxel_size=0.4) #f = all_points_pc.make_voxel_grid_filter()
    

    all_points = np.asarray(all_points_pc.points)# np.array(all_points_pc.to_list())


    x_min = -40
    y_min = -40
    x_max = 40 
    y_max = 40

    x_min_ind = np.floor(x_min/0.4).astype(int)
    x_max_ind = np.floor(x_max/0.4).astype(int)
    y_min_ind = np.floor(y_min/0.4).astype(int)
    y_max_ind = np.floor(y_max/0.4).astype(int)

    x_num = x_max_ind-x_min_ind+1
    y_num = y_max_ind-y_min_ind+1

    mat_global_image = np.zeros(( y_num,x_num),dtype=np.uint8)
          
    for i in range(all_points.shape[0]):
        x_ind = x_max_ind-np.floor(all_points[i,1]/0.4).astype(int)
        y_ind = y_max_ind-np.floor(all_points[i,0]/0.4).astype(int)
        if(x_ind>=x_num or y_ind>=y_num):
            continue
        if mat_global_image[ y_ind,x_ind]<10:
            mat_global_image[ y_ind,x_ind] += 1

    max_pixel = np.max(np.max(mat_global_image))

    mat_global_image[mat_global_image<=1] = 0  
    mat_global_image = mat_global_image*10
    
    mat_global_image[np.where(mat_global_image>255)]=255
    mat_global_image = mat_global_image/np.max(mat_global_image)*255

    return mat_global_image,x_max_ind,y_max_ind

def extract_number(filename):
    numeric_part = ''.join(filter(str.isdigit, filename))
    return int(numeric_part) if numeric_part else float('inf')

if __name__ == "__main__":

    args = parser.parse_args()
    bins_path = os.listdir(args.vel_path)
    sorted_bins_path = [file for file in bins_path if file.endswith(".pcd")]
    sorted_bins_path = sorted(sorted_bins_path, key=extract_number)
    # print(sorted_bins_path)
    os.system('mkdir -p '+args.bev_save_path)
    pose_path = args.bev_save_path + "/pose.txt"
    if os.path.exists(pose_path):
        os.remove(pose_path)

    B_t_L = np.array([-0.006253, 0.011775, 0.03055])
    B_R_L = R.from_quat([0,0,1,0]) #x,y,z,w

    timestamp = np.loadtxt(args.vel_path +'/time.csv')
    gt_path = np.loadtxt(args.vel_path + '/path_loop_dense_PSA_0325_9to11.csv', delimiter=',')
    gt_path = gt_path[np.argsort(gt_path[:, 0]), :]
    dup_idx = np.where(gt_path[1:, 0] - gt_path[:-1, 0] == 0)[0]
    gt_path = np.delete(gt_path, dup_idx, axis=0)
    # build slerp
    key_rots = R.from_quat(gt_path[:, [5, 6, 7, 4]])    
    key_times = gt_path[:, 0]
    slerp = Slerp(key_times, key_rots)
    key_pos = gt_path[:, 1:4]
    # print(len(sorted_bins_path), len(timestamp), len(gt_path))

    bins_per_group = 10
    total_groups = (len(sorted_bins_path) + bins_per_group - 1) // bins_per_group
    # print(total_groups)
    last_interp_rotsT = R.from_quat([0,0,0,1])
    last_pos_T = np.zeros([1, 3])
    write_idx = 1
    for group_idx in range(0, total_groups):
        start_idx = group_idx * bins_per_group
        end_idx = min(start_idx + bins_per_group, len(sorted_bins_path))
        merged_map = np.empty([0, 3])
        for i in range(start_idx, end_idx):
            b_p = sorted_bins_path[i]
            pcd = o3d.io.read_point_cloud(args.vel_path+'/'+b_p)
            pcs = np.asarray(pcd.points)[:,:3]
            
            if timestamp[i] < gt_path[0,0] or timestamp[i] > gt_path[-1,0]: continue
            interp_rots = slerp(timestamp[i])   
            W_interp_rots_L = R.from_matrix(np.dot(interp_rots.as_matrix(),B_R_L.as_matrix()))

            interp_pos = np.empty([1, key_pos.shape[1]])
            for j in range(key_pos.shape[1]):
                interp_pos[:, j] = np.interp(timestamp[i], key_times, key_pos[:, j])
            W_interp_pos_L = interp_rots.apply(B_t_L)+interp_pos
            # transform to world frame and merge
            merged_map = np.vstack([merged_map, W_interp_rots_L.apply(pcs) + W_interp_pos_L])            
            last_interp_rotsT = interp_rots.inv()
            last_pos_T = -last_interp_rotsT.apply(interp_pos)
        if merged_map is not np.empty([0, 3]) and len(merged_map)>1000:            
            merged_map = last_interp_rotsT.apply(merged_map) + last_pos_T
            merged_map = merged_map[np.where(np.abs(merged_map[:,0])<40)[0],:]
            merged_map = merged_map[np.where(np.abs(merged_map[:,1])<40)[0],:]
            merged_map = merged_map[np.where(np.abs(merged_map[:,2])<40)[0],:]
            merged_map = merged_map.astype(np.float32)
            img, _, _ = getBEV(merged_map)
            cv2.imwrite(args.bev_save_path+'/'+str(write_idx)+".png",img)
            write_idx = write_idx+1            
            with open(pose_path, 'a') as f:
                interp_rots = last_interp_rotsT.inv()
                interp_pos = -interp_rots.apply(last_pos_T)                
                pos_str = ' '.join([f'{elem:.6f}' for elem in interp_pos[0]])
                quat_str = ' '.join([f'{elem:.6f}' for elem in interp_rots.as_quat()])
                f.write(pos_str + ' ' + quat_str + '\n')
exit()
