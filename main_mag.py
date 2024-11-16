import argparse
from math import ceil
import random
import shutil
import json
from os.path import join, exists, isfile
from os import makedirs
import os
import cv2
from datetime import datetime
from mag_msg.msg import MagPointsXYZHT
from RANSAC import rigidRansac

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import h5py

from sklearn.decomposition import PCA

from tensorboardX import SummaryWriter
import numpy as np

from tqdm import tqdm
import faiss

# import kitti_dataset
import nclt_dataset
import dataset_loader 
import mag_dataset
import matplotlib.pyplot as plt

#ros
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from geometry_msgs.msg import Pose, Quaternion
import time
from scipy.spatial.transform import Slerp, Rotation as R
import skimage
import torch
import gpytorch
import threading
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def get_args():
    parser = argparse.ArgumentParser(description='BEVPlace++')
    parser.add_argument('--mode', type=str, default='test', help='Mode', choices=['train', 'test', 'ros'])
    
    parser.add_argument('--batchSize', type=int, default=2, 
            help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
    parser.add_argument('--cacheBatchSize', type=int, default=128, help='Batch size for caching and testing')
    parser.add_argument('--nEpochs', type=int, default=40, help='number of epochs to train for')
    parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
    parser.add_argument('--lrStep', type=float, default=10, help='Decay LR ever N steps.')
    parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
    parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')

    parser.add_argument('--threads', type=int, default=0, help='Number of threads for each data loader to use')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed to use.')


    parser.add_argument('--runsPath', type=str, default='./runs/', help='Path to save runs to.')
    parser.add_argument('--cachePath', type=str, default='./cache/', help='Path to save cache to.')


    parser.add_argument('--load_from', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
    parser.add_argument('--ckpt', type=str, default='best', 
            help='Load_from from latest or best checkpoint.', choices=['latest', 'best'])
    parser.add_argument('--gt_trans', type=str, default='', help='Path to load the global transformation matrix of GT.')
    

    opt = parser.parse_args()
    return opt

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.3

    def forward(self, anchor, positive, negative):
        
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        
        loss = F.relu(pos_dist-neg_dist + self.margin)
        return loss#.mean()

def train_epoch(epoch, model, train_set):
    
    epoch_loss = 0

    n_batches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    criterion = TripletLoss().to(device)
    
    
    model.eval()
    

    if epoch>=0:
        print('====> Building Cache for Hard Mining')
        train_set.mining=False
        train_set.cache = join(opt.cachePath, 'train_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5: 
            pool_size = model.global_feat_dim

            h5feat_query = h5.create_dataset("features_query", 
                                            [len(train_set), pool_size], 
                                            dtype=np.float32)
            h5feat_database = h5.create_dataset("features_database", 
                                                [train_set.len_database(), pool_size], 
                                                dtype=np.float32)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                                            batch_size=opt.batchSize, shuffle=False, 
                                            collate_fn=mag_dataset.collate_fn)
            with torch.no_grad():
                for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader, 1):
                    
                    query = query.to(device)
                    _, _, global_descs = model(query)
                    h5feat_query[indices, :] = global_descs.detach().cpu().numpy()
                for database_index in range(train_set.len_database()):
                    database_img, index = train_set.getDatabaseDes(database_index)
                    database_img = torch.tensor(database_img).to(device).unsqueeze(0)
                    _, _, global_descs = model(database_img)
                    h5feat_database[index, :] = global_descs.detach().cpu().numpy()

        train_set.mining=True
        train_set.refreshCache()
        
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, 
                                    batch_size=opt.batchSize, shuffle=True, 
                                    collate_fn=mag_dataset.collate_fn)
    
    model.train()

    for iteration, (query, positives, negatives, indices) in enumerate(training_data_loader):

        B, C, H, W = query.shape #shm: B = batch_size
        # print(query.shape)
        # print(positives.shape)
        # print(negatives.shape)
        # exit(0)
        # print(indices)

        input = torch.cat([query, positives, negatives])
        # print(input)

        input = input.to(device)
        
        _, _, global_descs = model(input)

        # print('1',global_descs.shape)
        global_descs_Q, global_descs_P, global_descs_N = torch.split(global_descs, [B, B, negatives.shape[0]])
        # print('2',global_descs_Q.shape)
        # print('3',global_descs_P.shape)
        # print('4',global_descs_N.shape)
        # print(global_descs_Q-global_descs_N[0])

        optimizer.zero_grad()

        # no need to train the kps feature
        loss = 0
        num_negs = negatives.shape[0]//B
        for i in range(len(global_descs_Q)):
            max_loss = torch.max(criterion(global_descs_Q[i], global_descs_P[i], global_descs_N[num_negs*i:num_negs*(i+1)]))
            loss += max_loss
        
        loss /= opt.batchSize
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss
        if iteration % 50 == 0 or n_batches <= 10:
            print("==> Epoch[{}]({}/{}): Loss: {:.8f}".format(epoch, iteration, 
                n_batches, batch_loss), flush=True)
            writer.add_scalar('Train/Loss', batch_loss, 
                    ((epoch-1) * n_batches) + iteration)
            

    optimizer.zero_grad()    
    avg_loss = epoch_loss / n_batches

    print("===> Epoch {} Complete: Avg. Loss: {:.8f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def infer(eval_set, return_local_feats = False):
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False)

    model.eval()
    model.to('cuda')    
    with torch.no_grad():    
        all_global_descs = []
        all_local_feats = []
        for _, (imgs, _) in enumerate(tqdm(test_data_loader)):            
            imgs = imgs.to(device)
            _, local_feat, global_desc = model(imgs)
            # print(local_feat.shape, global_desc.shape)
            all_global_descs.append(global_desc.detach().cpu().numpy())
            if return_local_feats:
                all_local_feats.append(local_feat.detach().cpu().numpy())
    
    if return_local_feats:
        return np.concatenate(all_local_feats, axis=0), np.concatenate(all_global_descs, axis=0)
    else:
        return np.concatenate(all_global_descs, axis=0)
    
def testPCA(eval_set, epoch=0, write_tboard=False):
    # TODO global descriptor PCA for faster inference speed
    pass
    # return recalls


def getClusters(cluster_set):
    n_descriptors = 10000
    n_per_image = 25
    n_im = ceil(n_descriptors/n_per_image)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), n_im, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                sampler=sampler)

    if not exists(opt.cachePath):
        makedirs(opt.cachePath)

    initcache = join(opt.cachePath, 'desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            all_feats = h5.create_dataset("descriptors", 
                        [n_descriptors, 128], 
                        dtype=np.float32)

            for iteration, (query, _, _, _) in enumerate(data_loader, 1):
                query = query.to(device)
                local_feat, _, _ = model(query)
                local_feat = local_feat.view(query.size(0), 128, -1).permute(0, 2, 1)
                
                batchix = (iteration-1)*opt.cacheBatchSize*n_per_image
                for ix in range(local_feat.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(local_feat.size(1), n_per_image, replace=False)
                    startix = batchix + ix*n_per_image
                    all_feats[startix:startix+n_per_image, :] = local_feat[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(n_im/opt.cacheBatchSize)), flush=True)
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(128, 64, niter=niter, verbose=False)
        kmeans.train(all_feats[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')


def saveCheckpoint(state, is_best, model_out_path, filename='checkpoint.pth.tar'):
    filename = model_out_path+'/'+filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_out_path+'/'+'model_best.pth.tar')



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


cnt = 0
res = 0.05
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

def getBEV(all_points): #N*3    

    points = np.array([point for point in all_points])
    mag_pos_intensity = np.empty([points.shape[0], 4])
    mag_pos_intensity[:,0:3] = points[:,1:4]
    mag_pos_intensity[:,3] = points[:,4]

    global res
    ds_points = voxel_filter(mag_pos_intensity, res)
    mags = ds_points[:,3]
    mean_ds_points = np.mean(ds_points[:,0:3], axis=0)
    points = ds_points[:,0:3] - mean_ds_points

    global cnt
    global likelihood
    global hyparam
    global model_gp

    if cnt == 0:    
        model_gp = ExactGPModel(torch.from_numpy(np.column_stack([points[:,1], points[:,0]])), torch.from_numpy(mags), likelihood)   
        model_gp.train()
        likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model_gp.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_gp)
        # this is for running the notebook in our testing framework
        training_iter = 50
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model_gp(torch.from_numpy(np.column_stack([points[:,1], points[:,0]])))
            # Calc loss and backprop gradients
            loss = -mll(output, torch.from_numpy(mags))
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model_gp.covar_module.base_kernel.lengthscale.item(),
                model_gp.likelihood.noise.item()
            ))
            optimizer.step()
        hyparam = model_gp.state_dict()
        model_gp.eval()
        likelihood.eval()
        # observed_pred = likelihood(model(torch.from_numpy(np.column_stack([points[:10,1], points[:100,0]]))))
    model_gp.load_state_dict(hyparam)
    model_gp.set_train_data(torch.from_numpy(np.column_stack([points[:,1], points[:,0]])), torch.from_numpy(mags), False)
    cnt += 1


    global x_max_ind
    global y_max_ind
    global x_num
    global y_num

    mat_global_image = np.zeros((y_num,x_num))
    # mat_global_image_raw = np.zeros((y_num,x_num))
    for i in range(points.shape[0]):
        y_ind = y_max_ind-np.floor(points[i,1]/res).astype(int)
        x_ind = x_max_ind+np.floor(points[i,0]/res).astype(int)
        if(x_ind>=x_num or y_ind>=y_num or x_ind<0 or y_ind<0): continue
        mat_global_image[y_ind,x_ind] = 1#np.linalg.norm(mags[i,0:3])
        # mat_global_image_raw[y_ind,x_ind] = mags[i]

    kernel_size = 10
    kernel = skimage.morphology.disk(kernel_size)
    mat_global_image = cv2.morphologyEx(mat_global_image, cv2.MORPH_CLOSE, kernel)

    # global max_intensity
    # global min_intensity
    max_intensity = 45
    min_intensity = 30

    img_id = np.where(mat_global_image==1)
    img_idx = np.column_stack([res*(-img_id[0]+y_max_ind), res*(img_id[1]-x_max_ind), img_id[0], img_id[1]])
    # print(np.where(mat_global_image==1)[0], np.where(mat_global_image==1)[1])
    # print(img_idx1-img_idx)

    observed_pred = likelihood(model_gp(torch.from_numpy(img_idx[:,0:2])))
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
    mat_global_image = mat_global_image*65535

    # mat_global_image_raw = (mat_global_image_raw-min_intensity)/(max_intensity-min_intensity)
    # mat_global_image_raw[np.where(mat_global_image_raw==-min_intensity/(max_intensity-min_intensity))] = 0 
    # mat_global_image_raw = mat_global_image_raw*65535.0

    # diff = mat_global_image - mat_global_image_raw
    # diff[mat_global_image_raw == 0] = 0
    # diff = np.abs(diff)


    return mat_global_image.astype(np.uint16), ds_points[:,0:3]

path_msg = Path()
path_msg.header.frame_id = "world"
def callback_gt(msg):
    global path_msg
    global gt_T_odom
    T_odom = np.eye(4)
    T_odom[0:3,0:3] = R.from_quat(np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])).as_matrix()
    T_odom[0:3,3] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
    T_gt = np.dot(gt_T_odom,T_odom)
    path_msg.header.stamp = msg.header.stamp
    pose = PoseStamped()
    pose.header.frame_id = "world"
    pose.header.stamp = msg.header.stamp
    # 设置点的坐标
    pose.pose.position.x = T_gt[0,3]
    pose.pose.position.y = T_gt[1,3]
    pose.pose.position.z = T_gt[2,3]
    path_msg.poses.append(pose)
    global gt_pub
    gt_pub.publish(path_msg)

mag_msgs = np.empty([0,7])
def callback_mag_points(msg):
    global mag_msgs
    # global mag_times
    global lock
    lock.acquire()
    for mag_id in range(6):
        mag_msgs = np.vstack([mag_msgs, np.array([msg.header.stamp.to_sec(), \
                  msg.mag_points[mag_id].position.x, msg.mag_points[mag_id].position.y, msg.mag_points[mag_id].position.z, \
                  msg.mag_points[mag_id].magnetic_field.x, msg.mag_points[mag_id].magnetic_field.y, msg.mag_points[mag_id].magnetic_field.z])])
    lock.release()
    # TODO: MTX
    # mag_msgs.append(msg)
    # mag_times = np.vstack([mag_times, msg.header.stamp.to_sec()])
    # return msg

last_odom_pose = np.zeros(7)
last_odom_time = 0
begin_odom_pos = np.zeros(3)
mag_buffer = []
# qqq = 0
def callback_odom(msg):
    global mag_msgs
    global begin_odom_pos
    global last_odom_pose
    global last_odom_time
    global mag_buffer
    global lock

    odom_time = msg.header.stamp.to_sec()
    cur_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, \
                         msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
    if np.array_equal(last_odom_pose, np.zeros(7)): 
        last_odom_pose = cur_pose
        last_odom_time = odom_time
        begin_odom_pos = cur_pose[0:3]
        return
    # print(mag_times.shape)
    if np.linalg.norm(cur_pose[0:3]-last_odom_pose[0:3]) < 0.2: return
    # print(mag_msgs[:,0].shape)
    lock.acquire()
    index = np.where((mag_msgs[:,0]<=odom_time) & (mag_msgs[:,0]>=last_odom_time))[0]
    lock.release()
    if len(index)==0: return
    # print(np.vstack((last_odom_pose[3:], cur_pose[3:])))
    slerp = Slerp(np.array([last_odom_time, odom_time]), R.from_quat(np.vstack((last_odom_pose[3:], cur_pose[3:]))))
    interp_quat = slerp(mag_msgs[index,0])
    interp_pos = np.empty([len(index),3])
    for j in range(3):
        interp_pos[:,j] = np.interp(mag_msgs[index,0], np.array([last_odom_time, odom_time]), np.array([last_odom_pose[j], cur_pose[j]]))
    for idx in index:
        b_T_m = np.eye(4)
        b_T_m[0:3,3] = mag_msgs[idx,1:4]
        b_T_m[1,1] = -1
        b_T_m[2,2] = -1
        w_T_b = np.eye(4)
        idx_odom = np.where(index==idx)
        # print(interp_quat[idx_odom], interp_pos, cur_pose[0:3])
        w_T_b[0:3,0:3] = interp_quat[idx_odom].as_matrix()
        w_T_b[0:3,3] = interp_pos[idx_odom,:]        
        w_T_m = np.dot(w_T_b,b_T_m)
        mag_buffer.append(np.array([mag_msgs[idx,0], w_T_m[0,3], w_T_m[1,3], w_T_m[2,3], np.linalg.norm(mag_msgs[idx,4:7])]))
        # print(mag_buffer[-1][1:4], interp_pos[idx_odom], cur_pose[0:3])
    lock.acquire()
    mag_msgs = mag_msgs[(np.max(index)+1):,:]
    lock.release()
    last_odom_pose = cur_pose
    last_odom_time = odom_time
    
    if np.linalg.norm(cur_pose[0:3]-begin_odom_pos) > 5:
        # Generate BEV
        # global qqq
        img, ds_points = getBEV(mag_buffer)
        
        # cv2.imwrite('/home/cartin/work_shm/bevplace_ws/src/BEVPlace/datasets/PSA_MAG'+'/'+str(qqq)+"_ros.png",img)   
        # print(qqq)
        # qqq = qqq+1
        # if qqq == 129:
            # cv2.imshow('local feature 1', img)
            # cv2.waitKey(0)
        img = (img.astype(np.float32))/65535
        img_rgb = img[np.newaxis, :, :].repeat(3,0)
        img_device = torch.tensor(img_rgb).to(device).unsqueeze(0)
        global model
        with torch.no_grad():    
            _, local_feat_query, global_desc_query = model(img_device)
        global_desc_query = global_desc_query.detach().cpu().numpy()
        local_feat_query = local_feat_query.detach().cpu().numpy()  
        global faiss_index   
        D, predictions = faiss_index.search(global_desc_query, 1)  #top1
        # print(D)

        for q_idx, pred in enumerate(predictions):
            db_img = cv2.imread(test_set.imgs_path[pred[0]], -1)
            query_img = img   
            db_uv = np.where(db_img>0)
            query_uv = np.where(query_img>0)
            # descs_dist = np.linalg.norm(global_desc_query - global_desc[pred[0]])
            global local_desc
            local_desc_db = local_desc[pred[0]].transpose(1,2,0) #u,v,feat
            faiss_index_local = faiss.IndexFlatL2(local_desc_db[db_uv].shape[1]) # dim of local featrue is 128
            faiss_index_local.add(np.array(local_desc_db[db_uv], order='C').astype('float32'))
            local_desc_query = local_feat_query[q_idx].transpose(1,2,0) #u,v,feat
            D_local, predictions_local = faiss_index_local.search(np.array(local_desc_query[query_uv], order='C').astype('float32'), 1)  #top1

            q_img_idx_local = np.empty([0,2])
            db_img_idx_local = np.empty([0,2])
            for q_idx_local, pred_local in enumerate(predictions_local):
                if np.sqrt(D_local[q_idx_local])>0.3: continue
                q_img_idx_local = np.vstack([q_img_idx_local, np.array([query_uv[0][q_idx_local],query_uv[1][q_idx_local]])])
                db_img_idx_local = np.vstack([db_img_idx_local, np.array([db_uv[0][pred_local[0]],db_uv[1][pred_local[0]]])])
            H, mask, max_csc_num = rigidRansac(q_img_idx_local,db_img_idx_local)
            T_db = np.eye(4)
            global position_mean
            global res
            T_db[0:3,3] =  position_mean[pred[0]] - np.array([x_num/2, y_num/2, 0])*res #Wdb_T_Idb
            dT_q = np.eye(4) # Idb_T_Iquery
            dT_q[0:2,0:2] = H[:,0:2]
            dT_q[0:2,3] = H[:,2]*res
            # print(H[:,2]*0.05)
            T = np.dot(T_db, dT_q)
            mean_ds_points = np.mean(ds_points[:,0:3], axis=0)
            invT_q = np.eye(4) # Idb_T_Iquery
            invT_q[0:3,3] = -(mean_ds_points - np.array([x_num/2, y_num/2, 0])*res)
            T = np.dot(T,invT_q)
            T_odom = np.eye(4)
            T_odom[0:3,0:3] = R.from_quat(cur_pose[3:]).as_matrix()
            T_odom[0:3,3] = cur_pose[0:3]
            T = np.dot(T,T_odom)
            # T_est.append(T)
            # print(H)
            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = "world"
            
            odom_msg.pose.pose.position.x = T[0, 3]
            odom_msg.pose.pose.position.y = T[1, 3]
            odom_msg.pose.pose.position.z = T[2, 3] 

            odom_quat = R.from_matrix(T[0:3,0:3]).as_quat()
            # print(odom_quat)

            odom_msg.pose.pose.orientation = Quaternion(odom_quat[0], odom_quat[1], odom_quat[2], odom_quat[3])
            
            odom_msg.twist.twist.linear.x = 0
            odom_msg.twist.twist.linear.y = 0
            odom_msg.twist.twist.angular.z = 0
            global odom_pub
            odom_pub.publish(odom_msg)
        # cv2.imshow('mag img', img)
        # cv2.waitKey(0)
        cut_index = np.floor(0.2*len(mag_buffer)).astype(int)
        begin_odom_pos = mag_buffer[cut_index][1:4]
        mag_buffer = mag_buffer[cut_index:]

if __name__ == "__main__":
    opt = get_args()

    device = torch.device("cuda")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    print('===> Building model')

    from REIN import REIN
    # from gem import REIN_GEM

    # model = REIN()
    model = REIN()
    model = model.cuda()
    
    # initialize netvlad with pre-trained or cluster
    if opt.load_from:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.load_from,  'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.load_from, 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)

            print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))
    else:
        initcache = join(opt.cachePath, 'desc_cen.hdf5')
        if not isfile(initcache):
            train_set = mag_dataset.TrainingDataset()
            print('===> Calculating descriptors and clusters')
            getClusters(train_set)
        with h5py.File(initcache, mode='r') as h5: 
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            model.pooling.init_params(clsts, traindescs) 
            model = model.cuda()

    # print(opt.mode.lower())
    if opt.mode.lower() == 'train':
        # preparing tensorboard
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')))

        logdir = writer.file_writer.get_logdir()
        try:
            makedirs(logdir)
        except:
            pass

        with open(join(logdir, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)


        print('===> Loading dataset(s)')

        train_set = mag_dataset.TrainingDataset() 
        # val_set={}
        # for seq in ['00', '02', '05', '06']:   
        # for seq in ['2012-02-04', '2012-03-17', '2012-06-15', '2012-09-28','2012-11-16','2013-02-23']:
            # val_set[seq] = kitti_dataset.InferDataset(seq=seq)

        # initilize model weights
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)    
    
        
            

        best_score = 0

        for epoch in range(opt.nEpochs):
            
            train_epoch(epoch, model, train_set)
            
            print('===> Testing')
                # writer.add_scalars('val', {'KITTI_'+seq: recall_top1}, epoch)
            eval = True
            if eval == True:
                eval_seq = ['YunnanGarden-mapping-local','YunnanGarden-false-loc-local']
                recalls = []
                precisions = []
                # F1s = []
                global_descs = []
                test_sets = []
                for seq in eval_seq:
                    test_set = mag_dataset.InferDataset(seq=seq, dataset='Husky/')
                    test_sets.append(test_set)
                    global_desc = infer(test_set)
                    global_descs.append(global_desc)
                recalls_mag, precision_mag, recall_top1, _ = mag_dataset.evaluateResultsPR(test_sets, global_descs)
                for iii in range(len(precision_mag)):
                    # recalls.append(recall_top1[iii])
                    print('===> Recall on Mag Sensor : %0.2f'%(recall_top1[iii]*100))
                    # print('===> Precision on Mag Sensor : %0.2f'%(np.mean(precisions)*100))
                
                mean_recall = np.mean(recalls)
            else:
                eval_seq =  ['Corridor_with_lift_B3_local']#,'Corridor-b3'
                recalls = []
                precisions = []
                for seq in eval_seq:
                    test_set = mag_dataset.InferDataset(seq=seq, dataset='Husky/')
                    global_descs = infer(test_set)
                    recalls_mag, precision_mag, recall_top1 = mag_dataset.evaluateResults(global_descs, test_set)
                    recalls.append(recall_top1)
                    precisions.append(np.mean(precision_mag))
                    # plt.rcParams.update({'font.size': 16})
                    # fig = plt.figure()
                    # plt.plot(recalls_mag, precision_mag)
                    # plt.xlim([0, 1])
                    # plt.ylim([0, 1])
                    # plt.xlabel("Recall [%]")
                    # plt.ylabel("Precision [%]")
                    # plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
                    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
                    # plt.show()

                # for seq in eval_seq:
                # test_set = mag_dataset.InferDataset(seq=eval_seq[0], dataset='Husky/')   
                # global_descs = infer(test_set)
            
                # recalls_mag = mag_dataset.evaluateResults(global_descs, test_set)# (q_descs, db_descs, q_dataset, db_dataset)
                # _, _, recall_top1 = mag_dataset.evaluateResults(eval_global_descs, eval_datasets)
            
            

                print('===> Mean Recall on Mag Sensor : %0.2f'%(np.mean(recalls)*100))
                print('===> Mean Precision on Mag Sensor : %0.2f'%(np.mean(precisions)*100))
            mean_recall = np.mean(recall_top1)
            # print(mean_recall)
            is_best = mean_recall > best_score 
            if is_best:   best_score = mean_recall
            
            saveCheckpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'recalls': mean_recall,
                    'best_score': best_score,
                    'optimizer' : optimizer.state_dict(),
            }, is_best, logdir)

        # print('===> Best Recall: %0.2f'%(mean_recall*100))
        writer.close()

    elif opt.mode.lower() == 'test':
        # import cv2
        # eval_seq =  ['CarparkB-loc-0829-easy']
        # recalls = []
        # precisions = []
        # F1s = []
        # for seq in eval_seq:
        #     test_set = mag_dataset.InferDataset(seq=seq, dataset='Husky/')
        #     global_descs = infer(test_set)
        #     recalls_mag, precision_mag, recall_top1 = mag_dataset.evaluateResults(global_descs, test_set)
        #     F1 = [2*((precision_mag[i]*recalls_mag[i])/(precision_mag[i]+recalls_mag[i])) for i in range(len(precision_mag))]            
        #     F1s.append(np.array(F1).max())
        #     recalls.append(recall_top1)
        #     precisions.append(np.mean(precision_mag))
        #     plt.rcParams.update({'font.size': 16})
        #     fig = plt.figure()
        #     plt.plot(recalls_mag, precision_mag)
        #     plt.xlim([0, 1])
        #     plt.ylim([0, 1])
        #     plt.xlabel("Recall [%]")
        #     plt.ylabel("Precision [%]")
        #     plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
        #     plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
        #     plt.show()
        # print('===> Recall on Mag Sensor : %0.2f'%(np.mean(recalls)*100))
        # print('===> Precision on Mag Sensor : %0.2f'%(np.mean(precisions)*100))

        # eval_seq = ['CarparkB-mapping-0829-local','CarparkB-loc-0830-hard-local']
        eval_seq = ['YunnanGarden-mapping-local', 'YunnanGarden-query-false-loc-local']
        recalls = []
        precisions = []
        F1s = []
        global_descs = []
        local_descs = []
        test_sets = []
        eval = True
        if eval == True:
            for seq in eval_seq:
                test_set = mag_dataset.InferDataset(seq=seq, dataset='Husky/')
                test_sets.append(test_set)
                local_desc, global_desc = infer(test_set, True)
                # tmp = local_desc[0].transpose(1,2,0)
                # tmp_norm = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)

                # data = local_desc[0]                
                # heatmap = data.sum(0)/data.shape[0]
                # heatmap = np.maximum(heatmap, 0)
                # heatmap /= np.max(heatmap)
                # heatmap = 1.0 - heatmap # 也可以不写，就是蓝色红色互换的作用
                # heatmap = cv2.resize(heatmap, (101,101)) # (224,224)指的是图像的size，需要resize到原图大小
                # heatmap = np.uint8(255 * heatmap)
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # cv2.imshow('local feature 0',heatmap)
                # data = local_desc[100]
                # heatmap = data.sum(0)/data.shape[0]
                # heatmap = np.maximum(heatmap, 0)
                # heatmap /= np.max(heatmap)
                # heatmap = 1.0 - heatmap # 也可以不写，就是蓝色红色互换的作用
                # heatmap = cv2.resize(heatmap, (101,101)) # (224,224)指的是图像的size，需要resize到原图大小
                # heatmap = np.uint8(255 * heatmap)
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # tmp_norm = np.linalg.norm(tmp, axis=-1)
                # print(tmp_norm)
                # cv2.imshow('local feature 1', heatmap)
                # cv2.waitKey(0)
                # print(tmp_norm.shape)
                # local_desc[0][]
                # global_desc = infer(test_set)
                # print(local_desc.shape, global_desc.shape)
                global_descs.append(global_desc)
                local_descs.append(local_desc)
            recalls_mag, precision_mag, recall_top1, T_est = mag_dataset.evaluateResultsPR(test_sets, global_descs, local_descs)
            
            est_traj = np.empty([0,2])
            for iii in range(len(T_est)):
                est_traj = np.vstack([est_traj, np.array([T_est[iii][0,3],T_est[iii][1,3]])])
                # plt.plot(T_est[iii][0,3], T_est[iii][1,3])
            plt.rcParams.update({'font.size': 16})
            fig = plt.figure()
            plt.scatter(est_traj[:,0],est_traj[:,1])
            plt.xlim([-100, 200])
            plt.ylim([-100, 200])
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
                # plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
                # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
            plt.show()
            for iii in range(len(precision_mag)):
                F1 = [2*((precision_mag[iii][i]*recalls_mag[iii][i])/(precision_mag[iii][i]+recalls_mag[iii][i])) for i in range(len(precision_mag[iii]))]            
                recalls.append(recall_top1[iii])
                # plt.rcParams.update({'font.size': 16})
                # fig = plt.figure()
                # plt.plot(recalls_mag[iii], precision_mag[iii])
                # plt.xlim([0, 1])
                # plt.ylim([0, 1])
                # plt.xlabel("Recall [%]")
                # plt.ylabel("Precision [%]")
                # plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
                # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
                # plt.show()
                print('===> Recall on Mag Sensor : %0.2f'%(recall_top1[iii]*100))
                # print('===> Precision on Mag Sensor : %0.2f'%(np.mean(precisions)*100))
        else:
            for seq in eval_seq:
                test_set = mag_dataset.InferDataset(seq=seq, dataset='Husky/')
                global_descs = infer(test_set)
                recalls_mag, precision_mag, recall_top1 = mag_dataset.evaluateResults(global_descs, test_set)
                F1 = [2*((precision_mag[i]*recalls_mag[i])/(precision_mag[i]+recalls_mag[i])) for i in range(len(precision_mag))]            
                F1s.append(np.array(F1).max())
                recalls.append(recall_top1)
                precisions.append(np.mean(precision_mag))
                plt.rcParams.update({'font.size': 16})
                fig = plt.figure()
                plt.plot(recalls_mag, precision_mag)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.xlabel("Recall [%]")
                plt.ylabel("Precision [%]")
                plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
                plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "20", "40", "60", "80", "100"])
                plt.show()
            print('===> Recall on Mag Sensor : %0.2f'%(np.mean(recalls)*100))
            print('===> Precision on Mag Sensor : %0.2f'%(np.mean(precisions)*100))

    elif opt.mode.lower() == 'ros':
        # import ipdb
        rospy.init_node('odometry_publisher')
        gt_pub = rospy.Publisher('/gt_path', Path, queue_size=10)
        odom_pub = rospy.Publisher('/estimated_odometry', Odometry, queue_size=10)
        gt_T_odom = np.genfromtxt(opt.gt_trans, delimiter=',')

        global_descs = []
        local_descs = []
        test_set = mag_dataset.InferDataset(seq='YunnanGarden-mapping-local', dataset='Husky/')
        # eval_set = mag_dataset.InferDataset(seq='YunnanGarden-query-false-loc-local', dataset='Husky/')
        # test_sets.append(test_set)
        local_desc, global_desc = infer(test_set, True)
        
        # print(global_desc.shape)
        faiss_index = faiss.IndexFlatL2(global_desc.shape[1])
        faiss_index.add(global_desc)

        position_mean = []
        for point in test_set.points:
            position_mean.append(np.mean(point, axis = 0))

        # ipdb.set_trace()
        T_est = []
        model.eval()
        model.to('cuda')
        
        lock = threading.Lock()
        rospy.Subscriber('/mag_array/MagPoints', MagPointsXYZHT, callback_mag_points, None, None, 1000, True)
        rospy.Subscriber('/Odometry',  Odometry, callback_odom, None, None, 100, True)
        rospy.Subscriber('/Odometry',  Odometry, callback_gt, None, None, 100, True)
        print("Network Prepared. Open Sensor Driver.")
        # for img_path in eval_set.imgs_path:
        #     img = cv2.imread(img_path, 0)
        #     img = (img.astype(np.float32))/256 
        #     img = img[np.newaxis, :, :].repeat(3,0)
        #     img = torch.tensor(img).to(device).unsqueeze(0)
        #     # img = img.to(device)
                        
        #     time_log_file = "execution_times_only_first.txt"

        #     with open(time_log_file, "a") as f:

        #         start_time = time.time()
        #         with torch.no_grad():    
        #             _, local_feat_query, global_desc_query = model(img)
        #         first_step_time = time.time()                    
        #         global_desc_query = global_desc_query.detach().cpu().numpy()
        #         local_feat_query = local_feat_query.detach().cpu().numpy()                    
        #         time1 = first_step_time - start_time
        #         f.write(f" {time1:.6f} \n")

        #     _, predictions = faiss_index.search(global_desc_query, 1)  #top1
        #     for q_idx, pred in enumerate(predictions):
        #         db_img = cv2.imread(test_set.imgs_path[pred[0]], -1)
        #         query_img = cv2.imread(img_path, -1)    
        #         db_uv = np.where(db_img>0)
        #         query_uv = np.where(query_img>0)
        #         descs_dist = np.linalg.norm(global_desc_query - global_desc[pred[0]])

        #         local_desc_db = local_desc[pred[0]].transpose(1,2,0) #u,v,feat
        #         faiss_index_local = faiss.IndexFlatL2(local_desc_db[db_uv].shape[1]) # dim of local featrue is 128
        #         faiss_index_local.add(np.array(local_desc_db[db_uv], order='C').astype('float32'))
        #         local_desc_query = local_feat_query[q_idx].transpose(1,2,0) #u,v,feat
        #         D_local, predictions_local = faiss_index_local.search(np.array(local_desc_query[query_uv], order='C').astype('float32'), 1)  #top1

        #         q_img_idx_local = np.empty([0,2])
        #         db_img_idx_local = np.empty([0,2])
        #         for q_idx_local, pred_local in enumerate(predictions_local):
        #             if np.sqrt(D_local[q_idx_local])>0.3: continue
        #             q_img_idx_local = np.vstack([q_img_idx_local, np.array([query_uv[0][q_idx_local],query_uv[1][q_idx_local]])])
        #             db_img_idx_local = np.vstack([db_img_idx_local, np.array([db_uv[0][pred_local[0]],db_uv[1][pred_local[0]]])])
        #         H, mask, max_csc_num = rigidRansac(q_img_idx_local,db_img_idx_local)
        #         T_db = np.eye(4)
        #         T_db[0:2,3] =  position_mean[pred[0]]
        #         T_q = np.eye(4)
        #         T_q[0:2,0:2] = H[:,0:2]
        #         T_q[0:2,3] = H[:,2]*0.05
        #         T = np.dot(T_db, T_q)
        #         T_est.append(T)

        # est_traj = np.empty([0,2])
        # for iii in range(len(T_est)):
        #     est_traj = np.vstack([est_traj, np.array([T_est[iii][0,3],T_est[iii][1,3]])])

        # sleep_duration = 0.5
        # for T in T_est:
        #     odom_msg = Odometry()
        #     odom_msg.header.stamp = rospy.Time.now()
        #     odom_msg.header.frame_id = "world"
            
        #     odom_msg.pose.pose.position.x = T[0, 3]
        #     odom_msg.pose.pose.position.y = T[1, 3]
        #     odom_msg.pose.pose.position.z = 0 

        #     odom_msg.pose.pose.orientation = Quaternion(0, 0, 0, 1)
            
        #     odom_msg.twist.twist.linear.x = 0
        #     odom_msg.twist.twist.linear.y = 0
        #     odom_msg.twist.twist.angular.z = 0
            
        #     odom_pub.publish(odom_msg)
        #     time.sleep(sleep_duration)
        rospy.spin()