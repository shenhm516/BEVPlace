import numpy as np
# import matplotlib.pyplot as plt
# import pcl
# import open3d as o3d
import cv2
import os
import argparse
# from tqdm import trange
from scipy.spatial.transform import Slerp, Rotation as R
from scipy.spatial import ConvexHull
import skimage
import torch
import gpytorch
import rosbag
import faiss

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
    # ds_points = np.copy(all_points)
    mags = ds_points[:,3]
    mean_ds_points = np.mean(ds_points[:,0:3], axis=0)
    points = ds_points[:,0:3] - mean_ds_points
    # print(points)   
    
    global cnt
    global likelihood
    global hyparam
    global model

    if cnt == 0:    
        model = ExactGPModel(torch.from_numpy(np.column_stack([points[:,1], points[:,0]])), torch.from_numpy(mags), likelihood)   
        model.train()
        likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # this is for running the notebook in our testing framework
        training_iter = 500
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(torch.from_numpy(np.column_stack([points[:,1], points[:,0]])))
            # Calc loss and backprop gradients
            loss = -mll(output, torch.from_numpy(mags))
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
    model.set_train_data(torch.from_numpy(np.column_stack([points[:,1], points[:,0]])), torch.from_numpy(mags), False)
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

    global max_intensity
    global min_intensity
    
    mat_global_image_raw = np.zeros((y_num,x_num))
    for i in range(points.shape[0]):        
        y_ind = y_max_ind+np.floor(points[i,1]/res).astype(int)
        x_ind = x_max_ind+np.floor(points[i,0]/res).astype(int)
        if(x_ind>=x_num or y_ind>=y_num or x_ind<0 or y_ind<0): continue
        mat_global_image_raw[y_ind,x_ind] = mags[i]
    mat_global_image_raw[np.where(mat_global_image_raw<min_intensity)] = min_intensity
    mat_global_image_raw[np.where(mat_global_image_raw>max_intensity)] = max_intensity
    mat_global_image_raw = (mat_global_image_raw-min_intensity)/(max_intensity-min_intensity)
    mat_global_image_raw = mat_global_image_raw*65535.0
    # kernel_size = 10
    # kernel = skimage.morphology.disk(kernel_size)
    # mat_global_image = cv2.morphologyEx(mat_global_image, cv2.MORPH_CLOSE, kernel)

    # print(mat_global_image.shape)
    # height,width = mat_global_image.shape 
    mat_global_image = np.zeros((y_num,x_num))   
    img_idx = np.empty([0, 4])
    for y in range(y_num):
        for x in range(x_num):
            # if mat_global_image[y,x] == 1:
            img_idx = np.vstack([img_idx, np.array([res*(y-y_max_ind), res*(x-x_max_ind), y, x])]) 
    # img_id = np.where(mat_global_image==1)
    # img_idx = np.column_stack([res*(img_id[0]-y_max_ind), res*(img_id[1]-x_max_ind), img_id[0], img_id[1]])
    # print(np.where(mat_global_image==1)[0], np.where(mat_global_image==1)[1])
    # print(img_idx1-img_idx)

    observed_pred = likelihood(model(torch.from_numpy(img_idx[:,0:2])))
    intensity_pre = observed_pred.mean.detach().numpy()
    mat_global_image[img_idx[:,2].astype(int), img_idx[:,3].astype(int)] = intensity_pre
    



    # mat_global_image_raw1 = mat_global_image_raw
    # # mat_global_image1 = mat_global_image
    # mat_global_image_raw1[np.where(mat_global_image_raw1<min_intensity)] = min_intensity
    # mat_global_image_raw1[np.where(mat_global_image_raw1>max_intensity)] = max_intensity
    # mat_global_image_raw1 = (mat_global_image_raw1-min_intensity)/(max_intensity-min_intensity)
    # kernel_size = 10
    # kernel = skimage.morphology.disk(kernel_size)
    # mat_global_image_raw1 = cv2.morphologyEx(mat_global_image_raw1, cv2.MORPH_CLOSE, kernel)
    # mat_global_image1[np.where(mat_global_image1<min_intensity)] = min_intensity
    # mat_global_image1[np.where(mat_global_image1>max_intensity)] = max_intensity
    # mat_global_image1 = (mat_global_image1-min_intensity)/(max_intensity-min_intensity)
    # cv2.imshow('raw_img',mat_global_image_raw1)
    # cv2.imshow('gp_img',mat_global_image1)
    
    
    # hull = ConvexHull(ds_points[:,0:2])
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # observed_pred = likelihood(model(torch.from_numpy(img_idx[:,0:2])))
    # for i in range(len(img_idx)):
    #     mat_global_image[img_idx[i,2].astype(int), img_idx[i,3].astype(int)] = intensity_pre[i]

    # max_intensity = 1000

    # print(max_intensity, min_intensity)
    # mat_global_image[np.where(mat_global_image>max_intensity)]=max_intensity
    mat_global_image[np.where(mat_global_image<min_intensity)] = min_intensity
    mat_global_image[np.where(mat_global_image>max_intensity)] = max_intensity
    mat_global_image = (mat_global_image-min_intensity)/(max_intensity-min_intensity)

    # mat_global_image[np.where(mat_global_image==-min_intensity/(max_intensity-min_intensity))] = 0 
    
    # cv2.imshow('gp_img1',mat_global_image)
    mat_global_image = mat_global_image*65535.0
    
    # cv2.waitKey(0)
    # mat_global_image_raw = (mat_global_image_raw-min_intensity)/(max_intensity-min_intensity)
    # mat_global_image_raw[np.where(mat_global_image_raw==-min_intensity/(max_intensity-min_intensity))] = 0 
    # mat_global_image_raw = mat_global_image_raw*65535.0

    # diff = mat_global_image - mat_global_image_raw
    # diff[mat_global_image_raw == 0] = 0
    # diff = np.abs(diff)


    return mat_global_image.astype(np.uint16), mat_global_image_raw.astype(np.uint16), ds_points[:,0:3]

def extract_number(filename):
    numeric_part = ''.join(filter(str.isdigit, filename))
    return int(numeric_part) if numeric_part else float('inf')

if __name__ == "__main__":

    args = parser.parse_args()
    bag_file = args.vel_path +'/2024-08-29-Parkinglot-map-GT.bag'
    bag = rosbag.Bag(bag_file, 'r')
    mag_txt = args.bev_save_path + "/mag_output.txt"
    if os.path.exists(mag_txt): os.remove(mag_txt)
        
    # wg_T_wl =np.array([[-4.302238853554692100e-02,-9.990497517432845864e-01,6.976218631714011138e-03,7.080125745248196267e+00],
    #                     [9.990501231129520487e-01,-4.306869158799939146e-02,-6.628673540541365865e-03,1.631233456977225771e+01],
    #                     [6.922831263764838046e-03,6.684410714340462654e-03,9.999536954582991521e-01,2.127301819800166971e-01],
    #                     [0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])
    wg_T_wl = np.eye(4)
    gt_time = np.empty([0, 1]) 
    gt_rot = np.empty([0, 4])
    gt_pos = np.empty([0, 3])
    for topic, msg, t in bag.read_messages(topics=['/Odometry']):
        gt_time = np.vstack([gt_time, t.to_sec()])
        wl_R_b = R.from_quat(np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]))
        wl_t_b = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        wl_T_b = np.eye(4)
        # print(wl_T_b,wl_R_b)
        wl_T_b[0:3,0:3] = wl_R_b.as_matrix()
        wl_T_b[0:3,3] = wl_t_b
        wg_T_b = np.dot(wg_T_wl,wl_T_b)
        wg_R_b = R.from_matrix(wg_T_b[0:3,0:3])
        wg_q_b = wg_R_b.as_quat()
        gt_rot = np.vstack([gt_rot, wg_q_b]) 
        gt_pos = np.vstack([gt_pos, wg_T_b[0:3,3]])
        # gt_rot = np.vstack([gt_rot, np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])]) 
        # gt_pos = np.vstack([gt_pos, np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])])
    slerp = Slerp(gt_time[:,0], R.from_quat(gt_rot))
    for topic, msg, t in bag.read_messages(topics=['/array_1/MagPoints']):#/array_1/MagPoints
        # print(f"Timestamp: {t}, Message: {msg}")
        mag_time = t.to_sec()
        if mag_time < np.min(gt_time[:,0]) or mag_time > np.max(gt_time[:,0]): continue  
        interp_rots = slerp(mag_time)
        interp_pos = np.empty([1, gt_pos.shape[1]])
        for j in range(gt_pos.shape[1]):
            interp_pos[:, j] = np.interp(mag_time, gt_time[:,0], gt_pos[:, j])
        w_T_b = np.eye(4)
        w_T_b[0:3,0:3] = interp_rots.as_matrix()
        w_T_b[0:3,3] = interp_pos
        # print(w_T_b)
        # W_interp_pos_L = interp_rots.apply(B_t_L)+interp_pos      
        for i in range(len(msg.mag_points)):
            b_T_m = np.eye(4)
            b_T_m[0:3,3] = np.array([msg.mag_points[i].position.x, msg.mag_points[i].position.y, msg.mag_points[i].position.z])
            b_T_m[1,1] = -1
            b_T_m[2,2] = -1
            w_T_m = np.dot(w_T_b,b_T_m)
            with open(mag_txt, 'a') as f:
                pos_str = ','.join([f'{elem:.6f}' for elem in w_T_m[0:3,3]])
                w_mag = np.dot(w_T_m[0:3,0:3],np.array([msg.mag_points[i].magnetic_field.x, msg.mag_points[i].magnetic_field.y, msg.mag_points[i].magnetic_field.z]))
                mag_str = ','.join([f'{elem:.6f}' for elem in w_mag])
                f.write(str(mag_time) + ',' + pos_str + ',' + mag_str + ',' + str(i) +'\n')


    mag_data = np.loadtxt(args.bev_save_path +'/mag_output.txt', delimiter=',')
    mag_data = mag_data[np.logical_and(mag_data[:, 7] >=8,  mag_data[:, 7] <12)]
    mag_data = mag_data[np.argsort(mag_data[:, 0]),:]

    max_intensity = 60#np.max(np.linalg.norm(mag_data[:,4:7],axis=1))
    min_intensity = 25#np.min(np.linalg.norm(mag_data[:,4:7],axis=1))

    
    # res = 0.05
    # x_min = -50
    # y_min = -50
    # x_max = 50 
    # y_max = 50
    # x_min_ind = np.floor(x_min/res).astype(int)
    # x_max_ind = np.floor(x_max/res).astype(int)
    # y_min_ind = np.floor(y_min/res).astype(int)
    # y_max_ind = np.floor(y_max/res).astype(int)
    # x_num = x_max_ind-x_min_ind+1
    # y_num = y_max_ind-y_min_ind+1
    # mat_global_image = np.zeros((y_num,x_num))
    # for mag in mag_data[:,1:7]:
    #     y_ind = y_max_ind+np.floor(mag[0]/res).astype(int)
    #     x_ind = x_max_ind+np.floor(mag[1]/res).astype(int)
    #     if(x_ind>=x_num or y_ind>=y_num or x_ind<0 or y_ind<0): continue
    #     mat_global_image[x_ind,y_ind] = np.linalg.norm(mag[3:6])

    # # # faiss_index = faiss.IndexFlatL2(mag_data[:,1:4].shape[1])
    # # # faiss_index.add(mag_data[:,1:4].astype('float32'))
    # # # _,_,I = faiss_index.range_search(mag_data[0,1:4].reshape(1,3).astype('float32'), 12.5) #(2.5*1.414)^2
    # # # # img, ds_points = getBEV(mag_data[I,1:7])
    # # # mat_local_image = np.zeros((101,101))
    # # # for mag in mag_data[I,1:7]:
    # # #     y_ind = 50+np.floor(mag[0]/0.1).astype(int)
    # # #     x_ind = 50+np.floor(mag[1]/0.1).astype(int)
    # # #     if(x_ind>=101 or y_ind>=101 or x_ind<0 or y_ind<0): continue
    # # #     mat_local_image[x_ind,y_ind] = np.linalg.norm(mag[4:7])
    # # # mat_local_image[np.where(mat_local_image<min_intensity)] = min_intensity
    # # # mat_local_image[np.where(mat_local_image>max_intensity)] = max_intensity
    # # # mat_local_image = (mat_local_image-min_intensity)/(max_intensity-min_intensity)
    # # # kernel_size = 10
    # # # kernel = skimage.morphology.disk(kernel_size)
    # # # mat_local_image = cv2.morphologyEx(mat_local_image, cv2.MORPH_CLOSE, kernel)
    # # # cv2.imshow('1',mat_local_image)
    # # # cv2.waitKey(0)
    # mat_global_image[np.where(mat_global_image<min_intensity)] = min_intensity
    # mat_global_image[np.where(mat_global_image>max_intensity)] = max_intensity
    # mat_global_image = (mat_global_image-min_intensity)/(max_intensity-min_intensity)
    # kernel_size = 10
    # kernel = skimage.morphology.disk(kernel_size)
    # mat_global_image = cv2.morphologyEx(mat_global_image, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('gp_img2',mat_global_image)
    # cv2.waitKey(0)
    # print(np.linalg.norm(mag_data[:,4:7], axis=1))
    mag_pos_intensity = np.empty([mag_data.shape[0], 4])
    mag_pos_intensity[:,0:3] = mag_data[:,1:4]
    mag_pos_intensity[:,3] = np.linalg.norm(mag_data[:,4:7], axis=1)
    filted_mag_data = voxel_filter(mag_pos_intensity, 0.5)
    # print(mag_data[0,1:4],filted_mag_data[0,0:3])
    faiss_index = faiss.IndexFlatL2(filted_mag_data[:,0:3].shape[1])
    faiss_index.add(filted_mag_data[:,0:3].astype('float32'))
    # filted_mag_data = voxel_filter(mag_data[:,1:7], 0.2)
    
    
    # faiss_index1 = faiss.IndexFlatL2(mag_data[:,1:4].shape[1])
    # faiss_index1.add(mag_data[:,1:4].astype('float32'))
    # _,_,I = faiss_index.range_search(np.array([26.26939,-1.0706773,-0.45174268]).reshape(1,3).astype('float32'), 12.5) #(2.5*1.414)^2
    # mat_local_image = np.zeros((101,101))
    # mean_ds_points = np.mean(filted_mag_data[I,0:3], axis=0)
    # for mag1 in filted_mag_data[I,:]:
    #     # mean_ds_points = np.mean(filted_mag_data[I,0:3], axis=0)
    #     y_ind = 50+np.floor((mag1[0]-mean_ds_points[0])/0.05).astype(int)
    #     x_ind = 50+np.floor((mag1[1]-mean_ds_points[1])/0.05).astype(int)
    #     if(x_ind>=101 or y_ind>=101 or x_ind<0 or y_ind<0): continue
    #     mat_local_image[x_ind,y_ind] = mag1[3]
    # mat_local_image[np.where(mat_local_image<min_intensity)] = min_intensity
    # mat_local_image[np.where(mat_local_image>max_intensity)] = max_intensity
    # mat_local_image = (mat_local_image-min_intensity)/(max_intensity-min_intensity)
    # # cv2.imshow('0',mat_local_image)
    # kernel_size = 10
    # kernel = skimage.morphology.disk(kernel_size)
    # mat_local_image = cv2.morphologyEx(mat_local_image, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('1',mat_local_image)
    # cv2.waitKey(0)
    
    write_idx = 0
    pose_path = args.bev_save_path + "/hull.txt"
    if os.path.exists(pose_path): os.remove(pose_path)
    for mag in filted_mag_data:
    # for mag in mag_data[:,1:7]:    
        # print(mag[0:3].reshape(1,3).astype('float32'), mag_data[0,1:4].reshape(1,3).astype('float32'))
        _,_,I = faiss_index.range_search(mag[0:3].reshape(1,3).astype('float32'), 30) #(2.5*1.414)^2
        # # _,_,I = faiss_index.range_search(mag_data[0,1:4].reshape(1,3).astype('float32'), 12.5) #(2.5*1.414)^2
        # mat_local_image = np.zeros((x_num,y_num))
        # mean_ds_points = np.mean(filted_mag_data[I,0:3], axis=0)
        # # print(mean_ds_points)
        # for mag1 in filted_mag_data[I,:]:
        #     # mean_ds_points = np.mean(filted_mag_data[I,0:3], axis=0)
        #     y_ind = y_max_ind+np.floor((mag1[0])/0.05).astype(int)
        #     x_ind = x_max_ind+np.floor((mag1[1])/0.05).astype(int)
        #     if(x_ind>=x_num or y_ind>=y_num or x_ind<0 or y_ind<0): continue
        #     mat_local_image[x_ind,y_ind] = mag1[3]
        # mat_local_image[np.where(mat_local_image<min_intensity)] = min_intensity
        # mat_local_image[np.where(mat_local_image>max_intensity)] = max_intensity
        # mat_local_image = (mat_local_image-min_intensity)/(max_intensity-min_intensity)
        # # cv2.imshow('0',mat_local_image)
        # kernel_size = 10
        # kernel = skimage.morphology.disk(kernel_size)
        # mat_local_image = cv2.morphologyEx(mat_local_image, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('1',mat_local_image)
        # cv2.waitKey(0)
        # print(np.linalg.norm(filted_mag_data[I,0:3] - mag[0:3].reshape(1,3),axis=1))
        # filted_mag_data[I,0:3] = filted_mag_data[I,0:3] - mag[0:3].reshape(1,3)
        img, img_raw, ds_points = getBEV(filted_mag_data[I,:])
        # ds_points[:,0:3] = ds_points[:,0:3] + mag[0:3].reshape(1,3)
        print(args.bev_save_path+'/'+str(write_idx)+".png")
        cv2.imwrite(args.bev_save_path+'/'+str(write_idx)+".png",img)  
        # cv2.imwrite(args.bev_save_path+'/'+str(write_idx)+"_raw.png",img_raw)  
        # cv2.waitKey(0)   
        with open(pose_path, 'a') as f:
            vertices_str = ','.join([f'{v[0]},{v[1]}' for v in ds_points[:,0:2]])
            f.write(vertices_str + '\n') 
        write_idx += 1 
exit()
