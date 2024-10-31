import os
from os.path import join, exists
import numpy as np
import cv2
from imgaug import augmenters as iaa
import torch
import torch.utils.data as data

import h5py

import faiss
from RANSAC import rigidRansac
from sklearn.metrics import precision_recall_curve, average_precision_score
import re
# kitti_seq_split_points = {"00":3000, "02":3400, "05":1000, "06":600, '08':1000}

def extract_number(s):
    return int(re.search(r'\d+', s).group())

class InferDataset(data.Dataset):
    def __init__(self, seq, dataset = 'KITTI/', dataset_path = './datasets/bevplace++_dataset/datasets/'):
        super().__init__()

        # bev path
        imgs_p = os.listdir(dataset_path+dataset+seq+'/bev_imgs/')
        imgs_p = sorted(imgs_p, key=extract_number)
        # imgs_p.sort()
        self.imgs_path = [dataset_path+dataset+seq+'/bev_imgs/'+i for i in imgs_p]

        # gt_pose
        self.poses = np.loadtxt(dataset_path+dataset+'poses/'+seq+'.txt')


    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index], 0)
        if 0:  #test rotation
            mat = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), np.random.randint(0,360), 1)
            img = cv2.warpAffine(img, mat, img.shape[:2])

        img = (img.astype(np.float32))/256 
        img = img[np.newaxis, :, :].repeat(3,0)
        
        return  img, index

    def __len__(self):
        return len(self.imgs_path)


def evaluateResults(dataset_name, global_descs, local_feats, dataset, match_results_save_path=None):
    gt_thres = 5  # gt threshold
    faiss_index = faiss.IndexFlatL2(global_descs.shape[1]) 
    # print(len(global_descs)*0.6)
    faiss_index.add(global_descs[:int(len(global_descs)*0.65)])

    D, predictions = faiss_index.search(global_descs[int(len(global_descs)*0.65)+200:], 1)  #top1
    # print(D, predictions)
    
    
    eval_start_split_point = int(len(global_descs)*0.65)+200  
    all_positives = 0
    tp = 0
    fp = 0
    real_loop = []
    detected_loop = []
    for q_idx, pred in enumerate(predictions):
        # flag_all_positives = False
        # flag_tp = False
        # flag_fp = False

        for i in range(len(pred)):
            query_idx = eval_start_split_point+q_idx
            gt_dis = (dataset.poses[query_idx] - dataset.poses[:int(len(global_descs)*0.65)])**2
            # print(dataset.poses)
            positives = np.copy(gt_dis)
            if dataset_name == 'PSA/':
                positives = np.where(np.sum(gt_dis[:,0:3],axis=1) < gt_thres**2 )[0]
                # print(positives)
            if dataset_name == 'KITTI/':
                positives = np.where(np.sum(gt_dis[:,[3,7,11]],axis=1) < gt_thres**2 )[0]
                # print(min(np.sum(gt_dis[:,[3,7,11]],axis=1)))
            # print(len(positives))
            
            descs_dist = np.sqrt(D[q_idx])#np.linalg.norm(global_descs[query_idx] - global_descs[pred[i]])
            # print(np.sqrt(D[q_idx]), descs_dist)
            # descs_norm1 = np.linalg.norm(global_descs[query_idx])
            # descs_norm2 = np.linalg.norm(global_descs[pred[i]])
            # print(query_idx, pred[i])
            if len(positives)>0:
                # if flag_all_positives==False:
                all_positives+=1
                real_loop.append(1)
                    # flag_all_positives = True
                # if positives in pred:
                if pred[i] in positives:
                    tp += 1
                    # flag_tp = True
                # print('0',descs_dist, descs_norm1, descs_norm2)      
            # elif positives in pred:
            # elif pred[i] in positives and flag_fp == False:
            #     fp += 1
            #     flag_fp = True
            else:
                real_loop.append(0)
                # if descs_dist<0.8: 
                #     fp += 1
            detected_loop.append(-descs_dist)
                # print("1",descs_dist, descs_norm1, descs_norm2)
    # fn =  all_positives - tp
        
        
    if all_positives != 0:
        recall_top1 = tp / all_positives #tp/(tp+fp)
        # precision = tp/(tp+fp)
        # print(all_positives)
    else: 
        recall_top1 = 1000
        precision = 1000

    precision, recall, _ = precision_recall_curve(real_loop, detected_loop)
    return recall, precision, recall_top1

        
def collate_fn(batch):

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query=np.array(query)
    positive=np.array(positive)
    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    
    negatives = torch.cat(negatives, 0)
    indices = list(indices)

    return query, positive, negatives, indices


class TrainingDataset(data.Dataset):
    def __init__(self, dataset_path = './datasets/bevplace++_dataset/datasets/', dataset = ['KITTI/', 'PSA/'], seq=[['00'],['0325-9-11-new']]):
        super().__init__()
        # neg, pos threshold
         
        self.pos_thres = 5
        self.neg_thres = 7 # 
        self.positives = []
        self.negatives = []
        # compute pos and negs for each query
        self.num_neg = 10
        for dataset_id in range(len(dataset)):
            # if dataset[dataset_id] == 'PSA/':
            #     self.pos_thres = 10
            #     self.neg_thres = 14 # 
            # print(len(dataset))
            for sequence in seq[dataset_id]:
                # bev path
                imgs_p = os.listdir(dataset_path+dataset[dataset_id]+sequence+'/bev_imgs/')
                imgs_p = sorted(imgs_p, key=extract_number)
                # imgs_p.sort()                
                
                # gt_pose, only first 3000 frames of KITTI for training
                poses = np.loadtxt(dataset_path+dataset[dataset_id]+'poses/'+sequence+'.txt')
                if dataset[dataset_id] == 'KITTI/':
                    poses = poses[:int(len(imgs_p)*0.7),[3,7,11]] #shm: translation x,y,z
                elif dataset[dataset_id] == 'NCLT/':
                    poses = poses[:int(len(imgs_p)*0.7), [4,8,12]] #shm: translation x,y,z
                elif dataset[dataset_id] == 'PSA/':
                    poses = poses[:int(len(imgs_p)*0.7), 0:3] #shm: translation x,y,z
                if sequence==seq[0][0]:
                    self.imgs_path=[dataset_path+dataset[dataset_id]+sequence+'/bev_imgs/'+i for i in imgs_p]
                    len_poses = 0
                    self.poses = poses
                else:
                    self.imgs_path.extend([dataset_path+dataset[dataset_id]+sequence+'/bev_imgs/'+i for i in imgs_p])
                    len_poses = len(self.poses)
                    np.concatenate((self.poses, poses), axis=0)
                    # print("Hello")
                cnt = 0
                for qi in range(len(poses)):
                    q_pose = poses[qi]
                    dises = np.sqrt(np.sum(((q_pose-poses)**2),axis=1))            
                    indexes = np.argsort(dises)               

                    remap_index = indexes[np.where(dises[indexes]<self.pos_thres)[0]]
                    # print(remap_index)
                    remap_index = len_poses + remap_index
                    self.positives.append(np.copy(remap_index))
                    self.positives[-1] = self.positives[-1][1:] #exclude query itself
                    # if cnt==226:  print(self.positives[-1])
                    negs = indexes[np.where(dises[indexes]>self.neg_thres)[0]]
                    negs = len_poses + negs
                    self.negatives.append(negs)
                    cnt+=1
        # print(len(self.positives), len(self.negatives))
        # print(self.positives[0].shape)
        self.mining = False
        self.cache = None # filepath of HDF5 containing feature vectors for images



    # refresh cache for hard mining
    def refreshCache(self):
        h5 = h5py.File(self.cache, mode='r')
        self.h5feat = np.array(h5.get("features"))

    def __getitem__(self, index):
        
        if self.mining:
            q_feat = self.h5feat[index]
            # print(self.positives[index], self.h5feat)
            pos_feat = self.h5feat[self.positives[index]]
            dis_pos = np.sqrt(np.sum((q_feat.reshape(1,-1)-pos_feat)**2,axis=1))

            min_idx = np.where(dis_pos==np.max(dis_pos))[0][0] 
            pos_idx = np.random.choice(self.positives[index], 1)[0]#
            # pos_idx = self.positives[index][min_idx]

            neg_feat = self.h5feat[self.negatives[index].tolist()]
            dis_neg = np.sqrt(np.sum((q_feat.reshape(1,-1)-neg_feat)**2,axis=1))
            
            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg-1:-1]

            neg_idx = self.negatives[index][dis_inc_index_tmp[:self.num_neg]]

              
        else:
            # print('Hello')
            # print(index, self.positives[index][0])
            # print('Hello2')
            pos_idx = self.positives[index][0]            
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]
        

        query = cv2.imread(self.imgs_path[index])
        # print(self.imgs_path[index])
        # rot augmentation
        mat = cv2.getRotationMatrix2D((query.shape[1]//2, query.shape[0]//2 ), np.random.randint(0,360), 1)
        query = cv2.warpAffine(query, mat, query.shape[:2])
        
        query = query.transpose(2,0,1)


        positive = cv2.imread(join(self.imgs_path[pos_idx]))#           
        mat = cv2.getRotationMatrix2D((positive.shape[1]//2, positive.shape[0]//2 ), np.random.randint(0,360), 1)
        positive = cv2.warpAffine(positive, mat, positive.shape[:2])
        positive = positive.transpose(2,0,1)
        

    
        query = (query.astype(np.float32))/256
        positive = (positive.astype(np.float32)/256)

        negatives = []

        for neg_i in neg_idx:
        
            negative = cv2.imread(self.imgs_path[neg_i])
            mat = cv2.getRotationMatrix2D((negative.shape[1]//2, negative.shape[0]//2 ), np.random.randint(0,360), 1)
            negative = cv2.warpAffine(negative, mat, negative.shape[:2]) 
            negative = negative.transpose(2,0,1)
            negative = (negative)/256
            
            negatives.append(torch.from_numpy(negative.astype(np.float32)))

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, index

    def __len__(self):
        return len(self.poses)
