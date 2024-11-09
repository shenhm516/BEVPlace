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
import skimage

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
        self.points = []
        # self.covs = []
        # gt_hull
        with open(dataset_path+dataset+'hull/'+seq+'.txt', 'r') as file:
            for line in file:
                points = line.strip().split(',')
                points = np.array([(float(points[i]), float(points[i+1])) for i in range(0, len(points), 2)])
                hull = ConvexHull(points)
                vertices = [(points[v][0], points[v][1]) for v in hull.vertices]
                polygon = Polygon(vertices)
                self.hulls.append(polygon)
                self.points.append(points)
        # with open(dataset_path+dataset+'cov/'+seq+'.txt', 'r') as file:
        #     for line in file:
        #         cov = line.strip().split(',')
        #         self.covs.append(cov)

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


def get_iou_2polys(gt_info_list, det_info_list, h, w, flag=False): 
    '''
    get the iou of 2 polys
    '''
    mask1 = np.zeros((h, w), np.uint8) 
    mask2 = np.zeros((h, w), np.uint8) 
    # cv2.imwrite("mask.jpg", mask) 

    gt_pts = (gt_info_list*50 / 20 + 50).astype(np.int32)
    mask1[gt_pts[:, 1],gt_pts[:, 0]] = 100
    # for gt_pt_info in gt_info_list:
    #     gt_pt = (gt_pt_info["points"] * 150 / 30 + 150).astype(np.int32)
    #     # print(gt_pt)
    #     # if gt_pt[:, 1] < h and gt_pt[:, 0]<w:
    #     mask1[gt_pt[:, 1],gt_pt[:, 0]] = 100
        # cv2.fillPoly(mask1, [gt_pt], 128)
    
    det_pts = (det_info_list* 50 / 20 + 50).astype(np.int32)
    mask2[det_pts[:, 1],det_pts[:, 0]] = 100
    # for det_pt_info in det_info_list:
    #     det_pt = (det_pt_info["points"] * 150 / 30 + 150).astype(np.int32)
    #     # if det_pt[:, 1] < h and det_pt[:, 0]<w:
    #     mask2[det_pt[:, 1],det_pt[:, 0]] = 100
    #     # cv2.fillPoly(mask2, [det_pt], 128)

    kernel_size = 10
    kernel = skimage.morphology.disk(kernel_size)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    

    # cv2.waitKey(0)
    ## method 1
    mask = mask1 + mask2
    # cv2.imwrite("mask1.jpg", mask1) 
    # cv2.imwrite("mask2.jpg", mask2) 
    # cv2.imwrite("mask.jpg", mask) 
    # cv2.waitKey(0)
    inter,_ = np.where(mask==200)
    # inter = (mask==200).sum()
    y1, _ = np.where(mask1==100)
    y2, _ = np.where(mask2==100)
    # print(len(inter), (mask==200).sum())
    # union = len(y1)
    iou1 = len(inter) / (len(y1) + 1e-6)
    iou2 = len(inter) / (len(y2) + 1e-6)
    iou = np.max([iou1,iou2])

    if flag:
        cv2.imshow('mask1',mask1)
        cv2.imshow('mask2',mask2)
        cv2.imshow('mask',mask)
        print('iou',iou)
        cv2.waitKey(0)
    # print(inter, len(y1), len(y2), union, iou)
    ## method 2
    # inter1_map = cv2.bitwise_and(mask1, mask2)
    # union1_map = cv2.bitwise_or(mask1, mask2)
    # inter1 = np.sum(inter1_map==1)
    # union1 = np.sum(union1_map==1)
    # iou1 = inter1 / (union1 + 1e-6)
    # print('===+++', iou, iou1)
    return iou

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
            
            for points in dataset.points[:int(len(global_descs)*0.65)]:
                    # query_info_list = [{"points": np.array(datasets[i].points[query_idx])}]
                    # database_info_list = [{"points": np.array(points)}]
                    p_dis = np.linalg.norm(np.mean(dataset.points[query_idx],axis=0) - np.mean(points,axis=0))
                    if p_dis <=5:
                        mean_query = np.mean(dataset.points[query_idx],axis=0)
                        # query_info_list = [{"points": dataset.points[query_idx]-mean_query}]
                        # database_info_list = [{"points": points-mean_query}]                        
                        iou = get_iou_2polys(points-mean_query, dataset.points[query_idx]-mean_query, 100, 100)
                    else: iou = 0
                    overlap_area = np.vstack([overlap_area, iou])
            # print(overlap_area)
            positives = np.copy(overlap_area)
            positives = np.where(overlap_area > 0.0)[0]
            # for hull in dataset.hulls[:int(len(global_descs)*0.65)]:
            # # for hull in dataset.hulls:
            #     area = dataset.hulls[query_idx].intersection(hull).area
            #     overlap_area = np.vstack([overlap_area, area])
            # positives = np.copy(ovquery_info_listerlap_area)
            # # print(positives)
            # # print(np.where(overlap_area > gt_thres))
            # # print(overlap_area)
            # positives = np.where(overlap_area > gt_thres)[0]

            descs_dist = np.linalg.norm(global_descs[query_idx] - global_descs[pred[i]])
            # print(descs_dist)
            if len(positives)>0:
                # if flag_all_positives==False:
                all_positives+=1
                real_loop.append(1)
                if pred[i] in positives:# and flag_tp==False:
                    tp += 1
                # else: 
                #     mean_query = np.mean(dataset.points[query_idx],axis=0)
                #     query_info_list = [{"points": dataset.points[query_idx]-mean_query}]
                #     database_info_list = [{"points": dataset.points[pred[i]]-mean_query}]
                #     iou = get_iou_2polys(database_info_list, query_info_list, 300, 300)
                #     print('0', iou)
            else:
                real_loop.append(0)
            detected_loop.append(-descs_dist)
    print(all_positives)
    if all_positives != 0:
        recall_top1 = tp / all_positives #tp/(tp+fp)
    else: 
        recall_top1 = 1000
    precision, recall, _ = precision_recall_curve(real_loop, detected_loop)
    return recall, precision, recall_top1

def evaluateResultsPR(global_descs, datasets):
    # gt_thres = 4  # gt threshold
    faiss_index = faiss.IndexFlatL2(global_descs[0].shape[1]) 
    # print(global_descs[0])
    faiss_index.add(global_descs[0])
    precisions = []
    recalls = []
    recalls_top1 = []
    # positives_buf = []
    for i in range(1, len(datasets)):
        _, predictions = faiss_index.search(global_descs[i], 1)  #top1
        all_positives = 0
        tp = 0
        real_loop = []
        detected_loop = []
        # positives_ = []
        for q_idx, pred in enumerate(predictions):
            # for iii in range(len(pred)):
            query_idx = q_idx
            overlap_area = np.empty([0,1])
            for points in datasets[0].points:
                p_dis = np.linalg.norm(np.mean(datasets[i].points[query_idx],axis=0) - np.mean(points,axis=0))
                if p_dis <= 5:
                    mean_query = np.mean(datasets[i].points[query_idx],axis=0)
                    iou = get_iou_2polys(points-mean_query, datasets[i].points[query_idx]-mean_query, 100, 100)
                else: iou = 0
                # area = 5*iou
                overlap_area = np.vstack([overlap_area, iou])
            positives = np.copy(overlap_area)
            positives = np.where(overlap_area > 0.0)[0]
            # positives_.append(positives)
            # positives_buf.append(positives_)
            # positives_buf[i-1][query_idx] = positives

            descs_dist = np.linalg.norm(global_descs[i][query_idx] - global_descs[0][pred[0]])
            if len(positives)>0:
                all_positives+=1
                real_loop.append(1)
                if pred[0] in positives:# and flag_tp==False:
                    tp += 1
            else:
                real_loop.append(0)
            detected_loop.append(-descs_dist)
        print(all_positives, tp)
        # if all_positives != 0:
        recall_top1 = tp / all_positives #tp/(tp+fp)
    # else: 
    #     recall_top1 = 1000
        # # 使用zip函数将两个队列打包在一起
        # combined = list(zip(detected_loop, real_loop))
        # # 根据queue1的值对combined进行排序
        # sorted_combined = sorted(combined, key=lambda x: x[0])
        # # 分离排序后的队列
        # sorted_queue1 = [item[0] for item in sorted_combined]
        # sorted_queue2 = [item[1] for item in sorted_combined]

        # print("排序后的队列1:", sorted_queue1)
        # print("排序后的队列2:", sorted_queue2)
        # print(len(sorted_queue1), np.sum(sorted_queue2))


        precision, recall, _ = precision_recall_curve(real_loop, detected_loop)
        # print(recall)
        precisions.append(precision)
        recalls.append(recall)
        recalls_top1.append(recall_top1)
    return recalls, precisions, recalls_top1
        
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
    def __init__(self, dataset_path = './datasets/bevplace++_dataset/datasets/', dataset = ['Husky/'], seq=[['YunnanGarden-mapping-local','YunnanGarden-query-mapping-local']]):#'YunnanGarden-query-btmleft-mapping-local', 'YunnanGarden-query-btmright-mapping-local','YunnanGarden-query-false-loc-local','YunnanGarden-query-topleft-mapping-local','YunnanGarden-query-topright-mapping-local']]):#'CarparkB-mapping-0829-global','CarparkB-mapping-0829-local'
        super().__init__()
        # neg, pos threshold
         
        # self.pos_thres = 4
        # self.neg_thres = 2 # 
        self.positives = []
        # self.positives_eval = []
        self.negatives = []
        # self.hulls = []
        self.points = []
        self.points_tmp = []
        self.imgs_path = []
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
                    self.imgs_database_path=[dataset_path+dataset[dataset_id]+sequence+'/bev_imgs/'+i for i in imgs_p]
                else:
                    if len(self.imgs_path)==0:
                        self.imgs_path=[dataset_path+dataset[dataset_id]+sequence+'/bev_imgs/'+i for i in imgs_p]
                    else:
                        self.imgs_path.extend([dataset_path+dataset[dataset_id]+sequence+'/bev_imgs/'+i for i in imgs_p])
                # len_poses = len(self.hulls)
                # print(len_poses)
                    # np.concatenate((self.hulls, poses), axis=0)
                
                # hull_tmp = []
                
                with open(dataset_path+dataset[dataset_id]+'hull/'+sequence+'.txt', 'r') as file:
                    for line in file:
                        points = line.strip().split(',')
                        points = np.array([(float(points[i]), float(points[i+1])) for i in range(0, len(points), 2)])
                        # hull = ConvexHull(points)
                        # vertices = [(points[v][0], points[v][1]) for v in hull.vertices]
                        # polygon = Polygon(vertices)
                        # self.hulls.append(polygon)
                        # hull_tmp.append(polygon)
                        if sequence==seq[0][0]:
                            self.points.append(points)
                        else:
                            self.points_tmp.append(points)
                    
                if sequence!=seq[-1][-1]: continue #database
                for i in range(len(self.points_tmp)):
                    overlap_areas = np.empty(0)
                    # current_polygon = hull_tmp[i]   
                    # current_points = points_tmp[i]                 
                    # for j in range(len(hull_tmp)):
                    for j in range(len(self.points)):
                        # other_polygon = hull_tmp[j]
                        # overlap_area = current_polygon.intersection(other_polygon).area
                        # overlap_areas = np.append(overlap_areas,overlap_area)
                        # print(points_tmp[j].shape)
                        # print(np.mean(points_tmp[j], axis=0))
                        
                        p_dis = np.linalg.norm(np.mean(self.points_tmp[i],axis=0) - np.mean(self.points[j],axis=0))
                        
                        if p_dis <=5:
                            mean_query = np.mean(self.points_tmp[i],axis=0)
                            # if i==199:
                            #     iou = get_iou_2polys(self.points[j]-mean_query, self.points_tmp[i]-mean_query, 100, 100,True)
                            # else:
                            iou = get_iou_2polys(self.points[j]-mean_query, self.points_tmp[i]-mean_query, 100, 100)
                        else: iou = 0
                        # area = 5*iou
                        overlap_areas = np.append(overlap_areas,iou)
                        # print(iou)
                    # if i==199: print(np.max(overlap_areas))   
                    indexes = np.argsort(overlap_areas)[::-1]
                    # if np.max(overlap_areas) < 0.8:  
                    #     # print(i,np.max(overlap_areas),indexes[np.where(overlap_areas[indexes]<0.8)[0]])
                    #     for j in range(len(self.points)):
                    #         p_dis = np.linalg.norm(np.mean(self.points_tmp[i],axis=0) - np.mean(self.points[j],axis=0))
                    #         if p_dis <=5:
                    #             get_iou_2polys(self.points[j]-mean_query, self.points_tmp[i]-mean_query, 100, 100, True)
                    # remap_index = len_poses + indexes[np.where(overlap_areas[indexes]>0.2)[0]]
                    # self.positives.append(np.copy(remap_index))
                    self.positives.append(indexes[np.where(overlap_areas[indexes]>0.7)[0]])
                    # self.positives_eval.append(indexes[np.where(overlap_areas[indexes]>0.1)[0]])
                    # if (len(indexes[np.where(overlap_areas[indexes]>0.2)[0]])==0): print("Hello!!!!!!!!!!!!")
                    # self.positives[-1] = self.positives[-1][1:] #exclude query itself
                    negs = indexes[np.where(overlap_areas[indexes]==0)[0]]
                    # negs = negs+len_poses
                    self.negatives.append(negs)
        # print(len(self.positives))
        self.mining = False
        self.cache = None # filepath of HDF5 containing feature vectors for images


    
    # refresh cache for hard mining
    def refreshCache(self):
        h5 = h5py.File(self.cache, mode='r')
        self.h5feat_database = np.array(h5.get("features_database"))
        self.h5feat_query = np.array(h5.get("features_query"))

    def getDatabaseDes(self, index):
        database = cv2.imread(self.imgs_database_path[index], -1)        
        database = (database.astype(np.float32))/ 65535.0
        database = np.expand_dims(database, axis=2)
        database = np.repeat(database, 3, axis=2)
        mat = cv2.getRotationMatrix2D((database.shape[1]//2, database.shape[0]//2 ), np.random.randint(0,360), 1)
        database = cv2.warpAffine(database, mat, database.shape[:2])
        database = database.transpose(2,0,1)
        return database, index
    
    def __getitem__(self, index):
        
        # if len(self.positives[index])==0 or len(self.negatives[index])<self.num_neg: continue
        if self.mining:
            # print(index)
            q_feat = self.h5feat_query[index]
            # print(self.positives[index], self.h5feat)
            # pos_feat = self.h5feat[self.positives[index]]
            # dis_pos = np.sqrt(np.sum((q_feat.reshape(1,-1)-pos_feat)**2,axis=1))

            # min_idx = np.where(dis_pos==np.max(dis_pos))[0][0] 
            pos_idx = np.random.choice(self.positives[index], 1)[0]#
            # dis_index = np.abs(index - self.positives[index])
            # max_idx = np.where(dis_index==np.max(dis_index))[0][0]
            # # print(dis_index, )
            # # max_idx = np.where(self.positives[index])
            # pos_idx = self.positives[index][max_idx]
            
            neg_feat = self.h5feat_database[self.negatives[index].tolist()]
            dis_neg = np.sqrt(np.sum((q_feat.reshape(1,-1)-neg_feat)**2,axis=1))
            
            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg-1:-1]

            # dis_index = np.abs(index - self.negatives[index])
            # # print(dis_index)
            # dis_inc_index_tmp = dis_index.argsort()[:-self.num_neg-1:-1]
            # print(dis_inc_index_tmp)
            # print('******')
            # neg_idx = np.random.choice(self.negatives[index], self.num_neg)#

            neg_idx = self.negatives[index][dis_inc_index_tmp[:self.num_neg]]

              
        else:
            # print('Hello')
            # print(index, self.positives[index][0])
            # print('Hello2')
            # pos_idx = self.positives[index][0]
            # print(index, len(self.positives[index]))
            pos_idx = np.random.choice(self.positives[index], 1)[0]
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]
        

        # print(index, len(self.imgs_path))
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
        


        
        # # # if index ==40:
        # q_img = cv2.imread(self.imgs_path[index], -1)
        # # # q_img = (q_img.astype(np.float32))/ 65535.0
        # p_img = cv2.imread(self.imgs_database_path[pos_idx], -1)#
        # # # p_img = (p_img.astype(np.float32))/ 65535.0
        # n_img = cv2.imread(self.imgs_database_path[neg_idx[0]], -1)
        # # # # n_img = (n_img.astype(np.float32))/ 65535.0
        # print(index,pos_idx,neg_idx[0])
        # cv2.imshow('q_img',q_img)
        # cv2.imshow('p_img',p_img)
        # cv2.imshow('n_img',n_img)
        # # # # # # cv2.waitKey(0)
        # # # # # # p_img = (p_img.astype(np.float32))/ 65535.0
        # ssim_index_p, _ = ssim(q_img, p_img, full=True)
        # ssim_index_n, _ = ssim(q_img, n_img, full=True)
        # # # # # print(self.imgs_path[index])
        # # # # # print(index,pos_idx,neg_idx[0])
        # print("SSIM:", ssim_index_p, ssim_index_n)
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
        # print(pos_idx, len(self.imgs_database_path))
        positive = cv2.imread(join(self.imgs_database_path[pos_idx]), -1)#
        positive = (positive.astype(np.float32)/65535.0)         
        positive = np.expand_dims(positive, axis=2)
        positive = np.repeat(positive, 3, axis=2)

        mat = cv2.getRotationMatrix2D((positive.shape[1]//2, positive.shape[0]//2 ), np.random.randint(0,360), 1)
        positive = cv2.warpAffine(positive, mat, positive.shape[:2])
        positive = positive.transpose(2,0,1)

        negatives = []
        for neg_i in neg_idx:
        
            negative = cv2.imread(self.imgs_database_path[neg_i], -1)
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
        return len(self.points_tmp)
    
    def len_database(self):
        return len(self.points)
