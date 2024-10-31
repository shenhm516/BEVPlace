import numpy as np
import os
import argparse
# # from tqdm import trange
from scipy.spatial.transform import Slerp, Rotation as R
import rosbag

parser = argparse.ArgumentParser(description='BEVPlace-Gen-BEV-Images')
parser.add_argument('--vel_path', type=str, default="/mnt/share_disk/KITTI/dataset/sequences/00/velodyne/", help='path to data')
parser.add_argument('--bev_save_path', type=str, default="./KITTI_new_imgs/00/imgs/", help='path to data')



if __name__ == "__main__":
    args = parser.parse_args()
    bag_file = args.vel_path +'/2024-08-25-B3-loc-GT.bag'
    bag = rosbag.Bag(bag_file, 'r')
    mag_txt = args.bev_save_path + "/mag_output.txt"
    if os.path.exists(mag_txt):
        os.remove(mag_txt)
    wg_T_wl =np.array([[-4.302238853554692100e-02,-9.990497517432845864e-01,6.976218631714011138e-03,7.080125745248196267e+00],
                        [9.990501231129520487e-01,-4.306869158799939146e-02,-6.628673540541365865e-03,1.631233456977225771e+01],
                        [6.922831263764838046e-03,6.684410714340462654e-03,9.999536954582991521e-01,2.127301819800166971e-01],
                        [0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])
    # wg_T_wl = np.eye(4)
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
                mag_str = ','.join([f'{elem:.6f}' for elem in np.array([msg.mag_points[i].magnetic_field.x, msg.mag_points[i].magnetic_field.y, msg.mag_points[i].magnetic_field.z])])
                f.write(str(mag_time) + ',' + pos_str + ',' + mag_str + ',' + str(i) +'\n')
