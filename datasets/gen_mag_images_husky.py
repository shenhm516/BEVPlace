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
import rosbag
from scipy.spatial.transform import Slerp, Rotation as R
import ipdb
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

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
    mag_pos_intensity = np.empty([points.shape[0], 4])
    mag_pos_intensity[:,0:3] = points[:,0:3]
    mag_pos_intensity[:,3] = np.linalg.norm(points[:,3:6], axis=1)

    ds_points = voxel_filter(mag_pos_intensity, res)
    mags = ds_points[:,3]
    mean_ds_points = np.mean(ds_points[:,0:3], axis=0)
    points = ds_points[:,0:3] - mean_ds_points

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
        training_iter = 50
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
    
    mat_global_image = np.zeros((y_num,x_num))
    # # mat_global_image_raw = np.zeros((y_num,x_num))
    # for i in range(points.shape[0]):
    #     y_ind = y_max_ind-np.floor(points[i,1]/res).astype(int)
    #     x_ind = x_max_ind-np.floor(points[i,0]/res).astype(int)
    #     if(x_ind>=x_num or y_ind>=y_num or x_ind<0 or y_ind<0): continue
    #     mat_global_image[y_ind,x_ind] = 1#np.linalg.norm(mags[i,0:3])
    #     # mat_global_image_raw[y_ind,x_ind] = np.linalg.norm(mags[i,0:3])
    img_idx = np.empty([0, 4])
    # print(mat_global_image.shape, y_num, x_num)
    for y in range(y_num):
        for x in range(x_num):
            img_idx = np.vstack([img_idx, np.array([res*(y-y_max_ind), res*(x-x_max_ind), y, x])]) 
            # img_idx = np.column_stack([-res*(y-y_max_ind), -res*(x-x_max_ind), y, x])
    # kernel_size = 10
    # kernel = skimage.morphology.disk(kernel_size)
    # mat_global_image = cv2.morphologyEx(mat_global_image, cv2.MORPH_CLOSE, kernel)

    # print(mat_global_image.shape)
    # height,width = mat_global_image.shape    
    
    # for y in range(height):
    #     for x in range(width):
    #         if mat_global_image[y,x] == 1:
    #             img_idx = np.vstack([img_idx, np.array([-res*(y-y_max_ind), -res*(x-x_max_ind), y, x])]) 
    # img_id = np.where(mat_global_image==1)
        
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


    return mat_global_image.astype(np.uint16), mat_global_image_raw.astype(np.uint16), ds_points[:,0:3]

def generate_custom_colormap():
    global max_intensity
    global min_intensity
    vector = np.arange(min_intensity, max_intensity, 0.000005)  #default, change if you need
    custom_cmap = np.zeros((len(vector), 3))

    for i in range(len(vector)):
        normalized_intensity = (vector[i] - min_intensity) / (max_intensity-min_intensity)
        normalized_intensity = min(max(normalized_intensity, 0.0), 1.0)

        h = normalized_intensity * 5.0 + 1.0
        j = int(np.floor(h))
        f = h - j
        if j % 2 == 0:
            f = 1 - f
        n = 1 - f

        if j <= 1:
            custom_cmap[i] = [n, 0, 1]
        elif j == 2:
            custom_cmap[i] = [0, n, 1]
        elif j == 3:
            custom_cmap[i] = [0, 1, n]
        elif j == 4:
            custom_cmap[i] = [n, 1, 0]
        elif j >= 5:
            custom_cmap[i] = [1, n, 0]

    return ListedColormap(custom_cmap)

def trans_grey_color(img):
    # global max_intensity
    # global min_intensity
    img = img/65536
    # print(img)
    img_color = np.zeros((img.shape[0],img.shape[1],3))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # if img[x,y] == 0: continue
            h = img[x,y] * 5.0 + 1.0
            j = int(np.floor(h))
            f = h - j
            if j % 2 == 0:
                f = 1 - f
            n = 1 - f

            if j <= 1:
                img_color[x,y] = [n, 0, 1]
            elif j == 2:
                img_color[x,y] = [0, n, 1]
            elif j == 3:
                img_color[x,y] = [0, 1, n]
            elif j == 4:
                img_color[x,y] = [n, 1, 0]
            elif j >= 5:
                img_color[x,y] = [1, n, 0]
    return img_color*255

def trans_grey_color_raw(img):
    # global max_intensity
    # global min_intensity
    img = img/65536
    # print(img)
    img_color = np.zeros((img.shape[0],img.shape[1],3))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x,y] == 0: 
                img_color[x,y] = [1,1,1]
                continue
            h = img[x,y] * 5.0 + 1.0
            j = int(np.floor(h))
            f = h - j
            if j % 2 == 0:
                f = 1 - f
            n = 1 - f

            if j <= 1:
                img_color[x,y] = [n, 0, 1]
            elif j == 2:
                img_color[x,y] = [0, n, 1]
            elif j == 3:
                img_color[x,y] = [0, 1, n]
            elif j == 4:
                img_color[x,y] = [n, 1, 0]
            elif j >= 5:
                img_color[x,y] = [1, n, 0]
    return img_color*255

def apply_colormap_to_images(img):

    colormap = generate_custom_colormap()

    # for filename in os.listdir(input_folder):
    #     if filename.lower().endswith('.png'):
    #         input_path = os.path.join(input_folder, filename)
    #         output_path = os.path.join(output_folder, filename)

    # print(gray_image.max())
    # ipdb.set_trace()
    # print(img[np.where(img>0)])
    norm_image = img/65536
    # print(norm_image)
    # ipdb.set_trace()
    color_image = colormap(norm_image)[:, :, :3]  
    color_image = (color_image * 255).astype(np.uint8) 

    # out_path = '/home/cartin/work_shibo/PCD/img/img.png'
    # plt.imsave(out_path, color_image)
    # cv2.imshow('color',color_image)
    # cv2.imshow('raw',img)
    # cv2.waitKey(0)

    return color_image

    # plt.imsave(output_path, color_image)

    # print(f"Save to here: {output_path}")

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
    # mag_data[]
    mag_data = mag_data[np.argsort(mag_data[:, 0]),:]
    # print(len(mag_data), np.linalg.norm(mag_data[:,4:7],axis=1))
    max_intensity = 60#np.max(np.linalg.norm(mag_data[:,4:7],axis=1))
    min_intensity = 25#np.min(np.linalg.norm(mag_data[:,4:7],axis=1))
    # print(min_intensity, max_intensity)
    pose_path = args.bev_save_path + "/hull.txt"
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
            # for ii in range(len(mag_buffer)):
            #     mag_buffer_local[ii][0:3] = mag_buffer[ii][0:3]-mag_buffer[0][0:3]
            
            
            # if tmp_cnt ==0 or tmp_cnt >250: 
            img, img_raw, ds_points = getBEV(mag_buffer_local)
            #Draw the img with RGB Style
            # img = apply_colormap_to_images(img)
            # img_raw = apply_colormap_to_images(img_raw)
            img = trans_grey_color(img)
            img_raw = trans_grey_color_raw(img_raw)
        
            # ds_points = ds_points + mag_buffer[0][0:3]
            # hull = ConvexHull(ds_points)
            # print(img.dtype)
            print(args.bev_save_path+'/'+str(write_idx)+".png")
            cv2.imwrite(args.bev_save_path+'/'+str(write_idx)+".png",img)    
            cv2.imwrite(args.bev_save_path+'/'+str(write_idx)+"_raw.png",img_raw)     
            # print(args.bev_save_path+'/raw' + str(write_idx)+".png")
            # cv2.imwrite(args.bev_save_path+'/raw'+str(write_idx)+".png",img_raw)
            # print(args.bev_save_path+'/diff' + str(write_idx)+".png")
            # cv2.imwrite(args.bev_save_path+'/diff'+str(write_idx)+".png",img_diff)       
            with open(pose_path, 'a') as f:
                vertices_str = ','.join([f'{v[0]},{v[1]}' for v in ds_points[:,0:2]])
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
