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
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from skimage.metrics import structural_similarity as ssim
import re

# kitti_seq_split_points = {"00":3000, "02":3400, "05":1000, "06":600, '08':1000}

from sklearn.metrics import precision_recall_curve, average_precision_score

def extract_number(s):
    return int(re.search(r'\d+', s).group())

class InferDataset(data.Dataset):
    def __init__(self, seq, dataset = 'KITTI/', dataset_path = './datasets/bevplace++_dataset/datasets/'):
        super().__init__()

        # bev path
        imgs_p = os.listdir(dataset_path+dataset+seq+'/bev_imgs/')
        # print(imgs_p)
        imgs_p = sorted(imgs_p, key=extract_number)
        self.imgs_path = [dataset_path+dataset+seq+'/bev_imgs/'+i for i in imgs_p]
        self.hulls = []
        # gt_hull
        with open(dataset_path+dataset+'hull/'+seq+'.txt', 'r') as file:
            for line in file:
                points = line.strip().split(',')
                points = np.array([(float(points[i]), float(points[i+1])) for i in range(0, len(points), 2)])
                hull = ConvexHull(points)
                vertices = [(points[v][0], points[v][1]) for v in hull.vertices]
                polygon = Polygon(vertices)
                self.hulls.append(polygon)


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


def evaluateResults(global_descs, dataset, match_results_save_path=None):
    gt_thres = 4  # gt threshold
    faiss_index = faiss.IndexFlatL2(global_descs.shape[1]) 
    faiss_index.add(global_descs[:int(len(global_descs)*0.65)])
    _, predictions = faiss_index.search(global_descs[int(len(global_descs)*0.65):], 1)  #top1
    # nearest = faiss_index.search(global_descs[:-1], 1)
    # print(nearest)
    eval_start_split_point = int(len(global_descs)*0.65)  
    # eval_start_split_point = 0
    all_positives = 0
    tp = 0
    # fp = 0
    real_loop = []
    detected_loop = []
    for q_idx, pred in enumerate(predictions):
        for i in range(len(pred)):
            query_idx = eval_start_split_point+q_idx
            overlap_area = np.empty([0,1])
            for hull in dataset.hulls[:int(len(global_descs)*0.65)]:
            # for hull in dataset.hulls:
                area = dataset.hulls[query_idx].intersection(hull).area
                overlap_area = np.vstack([overlap_area, area])
            positives = np.copy(overlap_area)
            # print(positives)
            # print(np.where(overlap_area > gt_thres))
            # print(overlap_area)
            positives = np.where(overlap_area > gt_thres)[0]

            descs_dist = np.linalg.norm(global_descs[query_idx] - global_descs[pred[i]])
            # print(descs_dist)
            if len(positives)>0:
                # if flag_all_positives==False:
                all_positives+=1
                real_loop.append(1)
                if pred[i] in positives:# and flag_tp==False:
                    tp += 1
                    # print('0', np.linalg.norm(global_descs[query_idx] - global_descs[pred[i]]))
            else:
                real_loop.append(0)
            detected_loop.append(-descs_dist)
    # print(all_positives)
    if all_positives != 0:
        recall_top1 = tp / all_positives #tp/(tp+fp)
    else: 
        recall_top1 = 1000
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
    def __init__(self, dataset_path = './datasets/bevplace++_dataset/datasets/', dataset = ['Husky/'], seq=[['CarparkB-mapping-0829']]):#'Corridor-b3', 
        super().__init__()
        # neg, pos threshold
         
        self.pos_thres = 4
        self.neg_thres = 2 # 
        self.positives = []
        self.negatives = []
        self.hulls = []
        # compute pos and negs for each query
        self.num_neg = 10
        for dataset_id in range(len(dataset)):
            for sequence in seq[dataset_id]:
                # print(dataset_path+dataset[dataset_id]+sequence)
                imgs_p = os.listdir(dataset_path+dataset[dataset_id]+sequence+'/bev_imgs/')
                imgs_p = sorted(imgs_p, key=extract_number)
                # print(imgs_p)
                
                # gt_pose, only first 3000 frames of KITTI for training
                # self.read_specific_line_from_file(dataset_path+dataset[dataset_id]+sequence+'/hull.txt',)
                # print(dataset_path+dataset[dataset_id]+sequence+'/hull.txt')
                # convex_hulls = []
                if sequence==seq[0][0]:
                    self.imgs_path=[dataset_path+dataset[dataset_id]+sequence+'/bev_imgs/'+i for i in imgs_p]
                else:
                    self.imgs_path.extend([dataset_path+dataset[dataset_id]+sequence+'/bev_imgs/'+i for i in imgs_p])
                len_poses = len(self.hulls)
                # print(len_poses)
                    # np.concatenate((self.hulls, poses), axis=0)
                
                hull_tmp = []
                with open(dataset_path+dataset[dataset_id]+'hull/'+sequence+'.txt', 'r') as file:
                    for line in file:
                        points = line.strip().split(',')
                        points = np.array([(float(points[i]), float(points[i+1])) for i in range(0, len(points), 2)])
                        hull = ConvexHull(points)
                        vertices = [(points[v][0], points[v][1]) for v in hull.vertices]
                        polygon = Polygon(vertices)
                        self.hulls.append(polygon)
                        hull_tmp.append(polygon)
                    
                for i in range(len(hull_tmp)):
                    overlap_areas = np.empty(0)
                    current_polygon = hull_tmp[i]                    
                    for j in range(len(hull_tmp)):
                        other_polygon = hull_tmp[j]
                        overlap_area = current_polygon.intersection(other_polygon).area
                        overlap_areas = np.append(overlap_areas,overlap_area)
                    indexes = np.argsort(overlap_areas)[::-1]
                    remap_index = len_poses + indexes[np.where(overlap_areas[indexes]>=self.pos_thres)[0]]
                    self.positives.append(np.copy(remap_index))
                    self.positives[-1] = self.positives[-1][1:] #exclude query itself
                    negs = indexes[np.where(overlap_areas[indexes]<self.neg_thres)[0]]
                    negs = negs+len_poses
                    self.negatives.append(negs)
        # print(len(self.positives))
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
            # pos_idx = self.positives[index][0]            
            pos_idx = np.random.choice(self.positives[index], 1)[0]
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]
        

        query = cv2.imread(self.imgs_path[index], -1)        
        query = (query.astype(np.float32))/ 65535.0
        query = np.expand_dims(query, axis=2)
        query = np.repeat(query, 3, axis=2)


        # print(query.shape, query.dtype)
        # exit(0)
        # print(self.imgs_path[index])
        # rot augmentation
        mat = cv2.getRotationMatrix2D((query.shape[1]//2, query.shape[0]//2 ), np.random.randint(0,360), 1)
        query = cv2.warpAffine(query, mat, query.shape[:2])
        query = query.transpose(2,0,1)
        


        
        # q_img = cv2.imread(self.imgs_path[index], -1)
        # # q_img = (q_img.astype(np.float32))/ 65535.0
        # p_img = cv2.imread(self.imgs_path[pos_idx], -1)#
        # # p_img = (p_img.astype(np.float32))/ 65535.0
        # n_img = cv2.imread(self.imgs_path[neg_idx[0]], -1)
        # # # n_img = (n_img.astype(np.float32))/ 65535.0
        # print(index,pos_idx,neg_idx[0])
        # cv2.imshow('q_img',q_img)
        # cv2.imshow('p_img',p_img)
        # cv2.imshow('n_img',n_img)
        # # # cv2.waitKey(0)
        # # # p_img = (p_img.astype(np.float32))/ 65535.0
        # # ssim_index_p, _ = ssim(q_img, p_img, full=True)
        # # ssim_index_n, _ = ssim(q_img, n_img, full=True)
        # # print(self.imgs_path[index])
        # # print(index,pos_idx,neg_idx[0])
        # # print("SSIM:", ssim_index_p, ssim_index_n)
        # cv2.waitKey(0)
        # hist1 = cv2.calcHist([q_img], [0], None, [256], [0, 256])
        # hist2 = cv2.calcHist([p_img], [0], None, [256], [0, 256])
        # hist3 = cv2.calcHist([n_img], [0], None, [256], [0, 256])
        # hist_diff_p = cv2.compareHist(hist1, hist2, method=cv2.HISTCMP_CORREL)
        # hist_diff_n = cv2.compareHist(hist1, hist3, method=cv2.HISTCMP_CORREL)
        # mse_p = np.mean((q_img - p_img) ** 2)
        # mse_n = np.mean((q_img - n_img) ** 2)
        
        # print("Histogram Comparison:", mse_p, mse_n)
        # print(pos_idx)
        # print(self.imgs_path[pos_idx])
        positive = cv2.imread(join(self.imgs_path[pos_idx]), -1)#
        positive = (positive.astype(np.float32)/65535.0)         
        positive = np.expand_dims(positive, axis=2)
        positive = np.repeat(positive, 3, axis=2)

        mat = cv2.getRotationMatrix2D((positive.shape[1]//2, positive.shape[0]//2 ), np.random.randint(0,360), 1)
        positive = cv2.warpAffine(positive, mat, positive.shape[:2])
        positive = positive.transpose(2,0,1)

        negatives = []
        for neg_i in neg_idx:
        
            negative = cv2.imread(self.imgs_path[neg_i], -1)
            negative = (negative.astype(np.float32)/65535.0)         
            negative = np.expand_dims(negative, axis=2)
            negative = np.repeat(negative, 3, axis=2)
            
            mat = cv2.getRotationMatrix2D((negative.shape[1]//2, negative.shape[0]//2 ), np.random.randint(0,360), 1)
            negative = cv2.warpAffine(negative, mat, negative.shape[:2]) 
            negative = negative.transpose(2,0,1)
            # negative = (negative)/256
            
            negatives.append(torch.from_numpy(negative.astype(np.float32)))

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, index

    def __len__(self):
        return len(self.hulls)
