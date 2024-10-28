import numpy as np
# import matplotlib.pyplot as plt
# import pcl
# import open3d as o3d
import cv2
import os
import argparse
# from tqdm import trange
# from scipy.spatial.transform import Slerp, Rotation as R
from scipy.spatial import ConvexHull
import skimage
import torch
import gpytorch
# import ipdb

parser = argparse.ArgumentParser(description='BEVPlace-Gen-BEV-Images')
parser.add_argument('--vel_path', type=str, default="/mnt/share_disk/KITTI/dataset/sequences/00/velodyne/", help='path to data')
parser.add_argument('--bev_save_path', type=str, default="./KITTI_new_imgs/00/imgs/", help='path to data')


def voxel_filter(point_cloud, leaf_size, random=False):
    filtered_points = []
    # 计算边界点
    x_min, y_min, z_min = np.amin(point_cloud[:,0:3], axis=0) #计算x y z 三个维度的最值
    x_max, y_max, z_max = np.amax(point_cloud[:,0:3], axis=0)
 
    # 计算 voxel grid维度
    Dx = (x_max - x_min)//leaf_size + 1
    Dy = (y_max - y_min)//leaf_size + 1
    # Dz = (z_max - z_min)//leaf_size + 1
    # print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))
 
    # 计算每个点的voxel索引
    h = list()  #h 为保存索引的列表
    for i in range(len(point_cloud)):
        hx = (point_cloud[i][0] - x_min)//leaf_size
        hy = (point_cloud[i][1] - y_min)//leaf_size
        hz = (point_cloud[i][2] - z_min)//leaf_size
        h.append(hx + hy*Dx + hz*Dx*Dy)
    h = np.array(h)
 
    # 筛选点
    h_indice = np.argsort(h) # 返回h里面的元素按从小到大排序的索引
    h_sorted = h[h_indice]
    begin = 0
    for i in range(len(h_sorted)-1):   # 0~9999
        if h_sorted[i] == h_sorted[i + 1]:
            continue
        else:
            point_idx = h_indice[begin: i+1]
            filtered_points.append(np.mean(point_cloud[point_idx], axis=0))
            begin = i+1
 
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # covar_x = self.rbf_kernel_module(x) + self.white_noise_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = ExactGPModel(torch.from_numpy(np.column_stack([0, 0])), 1, likelihood) 

cnt = 0
# model = ExactGPModel()
def getBEV(all_points): #N*3    
    res = 0.05
    points = np.array([point for point in all_points])
    ds_points = voxel_filter(points, res)
    mags = ds_points[:,3:6]
    mean_ds_points = np.mean(ds_points[:,0:3], axis=0)
    points = ds_points[:,0:3] - mean_ds_points

    global cnt
    global likelihood
    global hyparam
    global model

    if cnt == 0:    
        model = ExactGPModel(torch.from_numpy(np.column_stack([points[:,1], points[:,0]])), torch.from_numpy(np.linalg.norm(mags, axis=1)), likelihood)   
        model.train()
        likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # this is for running the notebook in our testing framework
        training_iter = 50
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(torch.from_numpy(np.column_stack([points[:,1], points[:,0]])))
            # Calc loss and backprop gradients
            loss = -mll(output, torch.from_numpy(np.linalg.norm(mags, axis=1)))
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            optimizer.step()
        hyparam = model.state_dict()
        model.eval()
        likelihood.eval()
        # observed_pred = likelihood(model(torch.from_numpy(np.column_stack([points[:10,1], points[:100,0]]))))
    model.load_state_dict(hyparam)
    model.set_train_data(torch.from_numpy(np.column_stack([points[:,1], points[:,0]])), torch.from_numpy(np.linalg.norm(mags, axis=1)), False)
    cnt += 1


    x_min = -2.5
    y_min = -2.5
    x_max = 2.5 
    y_max = 2.5
    x_min_ind = np.floor(x_min/res).astype(int)
    x_max_ind = np.floor(x_max/res).astype(int)
    y_min_ind = np.floor(y_min/res).astype(int)
    y_max_ind = np.floor(y_max/res).astype(int)
    x_num = x_max_ind-x_min_ind+1
    y_num = y_max_ind-y_min_ind+1

    mat_global_image = np.zeros((y_num,x_num))
    mat_global_image_raw = np.zeros((y_num,x_num))
    for i in range(points.shape[0]):
        y_ind = y_max_ind-np.floor(points[i,1]/res).astype(int)
        x_ind = x_max_ind-np.floor(points[i,0]/res).astype(int)
        if(x_ind>=x_num or y_ind>=y_num or x_ind<0 or y_ind<0): continue
        mat_global_image[y_ind,x_ind] = 1#np.linalg.norm(mags[i,0:3])
        mat_global_image_raw[y_ind,x_ind] = np.linalg.norm(mags[i,0:3])

    kernel_size = 10
    kernel = skimage.morphology.disk(kernel_size)
    mat_global_image = cv2.morphologyEx(mat_global_image, cv2.MORPH_CLOSE, kernel)

    # print(mat_global_image.shape)
    # height,width = mat_global_image.shape    
    # img_idx = np.empty([0, 4])
    # for y in range(height):
    #     for x in range(width):
    #         if mat_global_image[y,x] == 1:
    #             img_idx = np.vstack([img_idx, np.array([-res*(y-y_max_ind), -res*(x-x_max_ind), y, x])]) 
    img_id = np.where(mat_global_image==1)
    img_idx = np.column_stack([-res*(img_id[0]-y_max_ind), -res*(img_id[1]-x_max_ind), img_id[0], img_id[1]])
    # print(np.where(mat_global_image==1)[0], np.where(mat_global_image==1)[1])
    # print(img_idx1-img_idx)

    observed_pred = likelihood(model(torch.from_numpy(img_idx[:,0:2])))
    intensity_pre = observed_pred.mean.detach().numpy()
    mat_global_image[img_idx[:,2].astype(int), img_idx[:,3].astype(int)] = intensity_pre
    
    # hull = ConvexHull(ds_points[:,0:2])
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # observed_pred = likelihood(model(torch.from_numpy(img_idx[:,0:2])))
    # for i in range(len(img_idx)):
    #     mat_global_image[img_idx[i,2].astype(int), img_idx[i,3].astype(int)] = intensity_pre[i]

    # max_intensity = 1000
    global max_intensity
    global min_intensity
    # print(max_intensity, min_intensity)
    # mat_global_image[np.where(mat_global_image>max_intensity)]=max_intensity
    mat_global_image[np.where(mat_global_image<min_intensity)] = min_intensity
    mat_global_image[np.where(mat_global_image>max_intensity)] = max_intensity
    mat_global_image = (mat_global_image-min_intensity)/(max_intensity-min_intensity)
    # mat_global_image[np.where(mat_global_image==-min_intensity/(max_intensity-min_intensity))] = 0 
    mat_global_image = mat_global_image*65535.0

    # mat_global_image_raw = (mat_global_image_raw-min_intensity)/(max_intensity-min_intensity)
    # mat_global_image_raw[np.where(mat_global_image_raw==-min_intensity/(max_intensity-min_intensity))] = 0 
    # mat_global_image_raw = mat_global_image_raw*65535.0

    # diff = mat_global_image - mat_global_image_raw
    # diff[mat_global_image_raw == 0] = 0
    # diff = np.abs(diff)


    return mat_global_image.astype(np.uint16), ds_points[:,0:3]

def extract_number(filename):
    numeric_part = ''.join(filter(str.isdigit, filename))
    return int(numeric_part) if numeric_part else float('inf')

if __name__ == "__main__":

    args = parser.parse_args()
    mag_data = np.loadtxt(args.vel_path +'/mag_output.txt', delimiter=',')
    mag_data = mag_data[np.logical_and(mag_data[:, 7] >=8,  mag_data[:, 7] <12)]
    # mag_data[]
    mag_data = mag_data[np.argsort(mag_data[:, 0]),:]
    # print(len(mag_data), np.linalg.norm(mag_data[:,4:7],axis=1))
    max_intensity = 60#np.max(np.linalg.norm(mag_data[:,4:7],axis=1))
    min_intensity = 25#np.min(np.linalg.norm(mag_data[:,4:7],axis=1))
    # print(min_intensity, max_intensity)
    pose_path = args.vel_path + "/hull.txt"
    if os.path.exists(pose_path):
        os.remove(pose_path)


    last_pos = np.zeros([3,1])
    last_time = 0
    mag_buffer = []
    write_idx = 0
    # last_pos_shot = np.zeros([3,1])
    # tmp_cnt = 0
    for row in range(len(mag_data)):        
        time = mag_data[row,0]
        cur_pos = mag_data[row,1:4]
        
        if np.array_equal(last_pos, np.zeros([3, 1])): 
            last_pos = cur_pos
            # last_time = time
            mag_buffer.append(mag_data[row,1:7])
        elif np.linalg.norm(cur_pos-last_pos) > 5:# and time!=last_time:
            mag_buffer_local = np.copy(mag_buffer)
            for ii in range(len(mag_buffer)):
                mag_buffer_local[ii][0:3] = mag_buffer[ii][0:3]-mag_buffer[0][0:3]
            
            
            # if tmp_cnt ==0 or tmp_cnt >250: 
            img, ds_points = getBEV(mag_buffer_local)
            ds_points = ds_points + mag_buffer[0][0:3]
            hull = ConvexHull(ds_points)
            # print(img.dtype)
            print(args.bev_save_path+'/'+str(write_idx)+".png")
            cv2.imwrite(args.bev_save_path+'/'+str(write_idx)+".png",img)     
            # print(args.bev_save_path+'/raw' + str(write_idx)+".png")
            # cv2.imwrite(args.bev_save_path+'/raw'+str(write_idx)+".png",img_raw)
            # print(args.bev_save_path+'/diff' + str(write_idx)+".png")
            # cv2.imwrite(args.bev_save_path+'/diff'+str(write_idx)+".png",img_diff)       
            with open(pose_path, 'a') as f:
                vertices_str = ','.join([f'{ds_points[v][0]},{ds_points[v][1]}' for v in hull.vertices]) 
                # vertices_str = ','.join([f'{x},{y}' for x, y in hull.vertices]) 
                    
                f.write(vertices_str + '\n')                
            write_idx = write_idx+1

            cut_index = np.floor(0.2*len(mag_buffer)).astype(int)
            # last_pos = cur_pos
            last_pos = mag_buffer[cut_index][0:3]
            # last_time = time
            mag_buffer = mag_buffer[cut_index:]
            # mag_buffer = []
            continue
        if np.linalg.norm(cur_pos-last_pos) > 0.2:
            mag_buffer.append(mag_data[row,1:7])
        
        # last_pos_shot = cur_pos
exit()
