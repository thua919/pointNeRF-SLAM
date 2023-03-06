import glob

import copy
import os
import time, threading
rlock = threading.RLock()

import numpy as np
import torch
import cv2
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
from src.utils.utils_sys import Printer

from tqdm import tqdm
import platform 
    
from src.common import (get_camera_from_tensor, get_tensor_from_camera,get_samples)

from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer
from src.demo_superpoint import PointTracker

# superpoint tracker
#from src.superpoint_tracker import SuperpointTracker
from src.frame import Frame #match_frames
from src.utils.parameters import Parameters
from src.slam_dynamic_config import SLAMDynamicConfig
from src.search_points import propagate_map_point_matches
from src.map import Map
from src.keyframe import KeyFrame

# pyslam utils
from src.utils.utils_geom import triangulate_points, triangulate_normalized_points, add_ones, inv_T,poseRt,inv_poseRt
from src.utils.utils_draw import draw_feature_matches
from src.utils.mplot_thread import Mplot2d
#from src.utils.viewer3D import Viewer3D

if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')
  
font = cv2.FONT_HERSHEY_DUPLEX
font_clr = (255, 255, 255)
font_pt = (4, 12)
font_sc = 0.4

# parameter for matcher
index_params= dict(algorithm = 6,   # FLANN_INDEX_LSH = 6 Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search
                        table_number = 12,      # 12
                        key_size = 20,         # 20
                        multi_probe_level = 2) # 2 
search_params=dict(checks=32)
 

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

kFeatureMatchRatioTestInitializer = 0.8
kRansacThresholdNormalized = 0.0003  # metric threshold used for normalized image coordinates 
kRansacProb = 0.999 
kUseDynamicDesDistanceTh=True
   
class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']
        
        self.H=slam.H
        self.W=slam.W
        self.crop_edge = cfg['cam']['crop_edge']
        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None
        #=====================================================#
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        
        self.init_finished = slam.init_finished
        self.mapper_finished = slam.mapper_finished
        
        self.weak_depth = cfg['weak_depth']
        print("============ This is the weakly supervised run ===========")
        self.depth_patch = cfg['depth_patch']
        
        self.depth_supervision= slam.depth_supervision
        self.bundle_loss=slam.bundle_loss
        self.camera=slam.camera
        
        self.frames=slam.frames
        self.pseudo_depth_maps=slam.pseudo_depth_maps
        
        self.idx0=slam.idx0
        self.f_cur = slam.f_cur
        self.idxs_cur = slam.idxs_cur
        
        self.f_ref = slam.f_ref 
        self.idxs_ref = slam.idxs_ref
        
        self.idx_cur_inliers=None
        self.idx_ref_inliers=None
        
        self.depth_coord_cur=slam.depth_coord_cur
        self.depth_coord_ref=slam.depth_coord_ref
        self.depth_cur=slam.depth_cur
        self.depth_ref=slam.depth_ref
        self.depth_cur_weak=slam.depth_cur_weak
        self.depth_ref_weak=slam.depth_ref_weak
        
        self.dyn_config = SLAMDynamicConfig()
        #self.map=slam.map
        self.num_inliers=None # current number of matched points 
        self.mask_match=None
        
        self.viewer_show=cfg['viewer_show']
        self.now_Tcr=None
        #self.tracker = PointTracker(max_length=2)#max_length=2, nn_thresh=0.7
        
        self.median_gt_depth=None
        self.depth_scale=None
        self.debug=cfg['debug']
        self.track_every_frame=cfg['track_every_frame']
        
        self.depth_cur_gt=slam.depth_cur_gt
        
        if platform.system()  == 'Linux' and self.debug:     
            from src.utils.display2D import Display2D  #  !NOTE: pygame generate troubles under macOS!
            
        # ===================================================== #
        self.idx = slam.idx
        self.nice = slam.nice
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        '''
        # 在主线程NICE_SLAM.py被初始化得好好的：
        '''
        self.estimate_c2w_list = slam.estimate_c2w_list

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera'] #False
        print("GT_camera:",self.gt_camera)
        self.tracking_pixels = cfg['tracking']['pixels']
        self.seperate_LR = cfg['tracking']['seperate_LR'] #False
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W'] # 切掉一些边缘
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H'] # 切掉一些边缘
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption'] #True
 
        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def get_rays_from_uv(self,i, j, c2w, H, W, fx, fy, cx, cy):
        """
        Get corresponding rays from input uv.
        """
        if isinstance(c2w, np.ndarray):
            c2w = torch.from_numpy(c2w).to(self.device)
        dirs = torch.stack(
            [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(self.device)
        dirs = dirs.reshape(-1, 1, 3)
        # Rotate ray directions from camera frame to the world frame
        # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        rays_d = torch.sum(dirs * c2w[:3, :3], -1)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def select_uv(self,i, j, n, depth, color):
        """
        Select n uv from dense uv.
        """
        i = i.reshape(-1)
        j = j.reshape(-1)
        #self.depth_coord_cur
        depth = depth.reshape(-1)
        depth_np=depth.cpu().numpy()
        indices=np.where(depth_np>0.01) # originally 4
        indices=torch.tensor(indices, device=self.device)
        #indices = torch.randint(i.shape[0], (n,), device=self.device)
        indices = indices.clamp(0, i.shape[0])
        i = i[indices]  # (n)
        j = j[indices]  # (n)
        #depth = depth.reshape(-1)
        color = color.reshape(-1, 3)
        depth = depth[indices]  # (n)
        color = color[indices]  # (n,3)
        return i, j, depth, color


    def get_sample_uv(self, H0, H1, W0, W1, n, depth, color):
        """
        Sample n uv coordinates from an image region H0..H1, W0..W1
        """
        depth = depth[H0:H1, W0:W1]
        color = color[H0:H1, W0:W1]
        i, j = torch.meshgrid(torch.linspace(
            W0, W1-1, W1-W0).to(self.device), torch.linspace(H0, H1-1, H1-H0).to(self.device))
        i = i.t()  # transpose
        j = j.t()
        i, j, depth, color = self.select_uv(i, j, n, depth, color)
        return i, j, depth, color


    def get_samples(self, H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, depth, color):
        """
        Get n rays from the image region H0..H1, W0..W1.
        c2w is its camera pose and depth/color is the corresponding image tensor.

        """
        i, j, sample_depth, sample_color = self.get_sample_uv(
            H0, H1, W0, W1, n, depth, color)
        rays_o, rays_d = self.get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy)
        return rays_o, rays_d, sample_depth, sample_color

    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer,cam_iter):
        
        """
        # 在tracking的时候，是对每一帧都进行（I，D，Pose)的单一帧优化）
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        
        # 这里的get sample就是筛选ray了，这里 H-Hedge  W-Wedge 是忽略了图片的边缘信息
        # def get_samples(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, depth, color, device):
        # Get n rays from the image region H0..H1, W0..W1.
        # c2w is its camera pose and depth/color is the corresponding image tensor.
        
        if self.weak_depth: #self.get_samples
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = self.get_samples(
                Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color)
        else: #get_samples from common.py
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color)
        '''
        if self.nice:
            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]
        '''
        ret = self.renderer.render_batch_ray(
            self.c, self.decoders, batch_rays_d, batch_rays_o,  self.device, stage='color',  gt_depth=batch_gt_depth)
        
        depth, uncertainty, color = ret

        uncertainty = uncertainty.detach()
        if self.handle_dynamic: #对imap设为了False，但nice-slam是true
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        else:
            mask = batch_gt_depth > 0

        #在tracking的时候，是对每一帧都进行（I，D，Pose的单一帧优化）
        if not self.depth_supervision:
            if cam_iter == self.num_cam_iters-1:
                print("self.depth_supervision:",self.depth_supervision)
                print("====================This tracker is runing WITHOUT depth supervision=================")
            if self.use_color_in_tracking:
                loss = torch.abs(batch_gt_color - color)[mask].sum()
        else:
            if cam_iter == self.num_cam_iters-1:
                print("self.depth_supervision:",self.depth_supervision)
                print("====================This tracker is runing WITH weak depth supervision=================")    
            loss = (torch.abs(batch_gt_depth-depth) /
                    torch.sqrt(uncertainty+1e-10))[mask].sum()

            if self.use_color_in_tracking:
                color_loss = torch.abs(
                    batch_gt_color - color)[mask].sum()
                loss += self.w_color_loss*color_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

    '''def bundle_loss():
        loss=None
        return  loss'''

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                self.c[key] = val
            self.prev_mapping_idx = self.mapping_idx[0].clone()
    
    # here, the essential matrix algorithm uses the five-point algorithm solver by D. Nister (see the notes and paper above ) 
    def estimatePose(self, kpn_ref, kpn_cur):	     
        # here, the essential matrix algorithm uses the five-point algorithm solver by D. Nister (see the notes and paper above )     
        E, self.mask_match = cv2.findEssentialMat(kpn_cur, kpn_ref, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)                         
        _, R, t, mask = cv2.recoverPose(E, kpn_cur, kpn_ref, focal=1, pp=(0., 0.))                                                     
        return poseRt(R,t.T)  
    
    def angle_error_mat(self, R1, R2):
        cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
        cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
        return np.rad2deg(np.abs(np.arccos(cos)))

    def angle_error_vec(self, v1, v2):
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

    def compute_pose_error(self, f_ref, f_cur):
        '''
        这段evaluate pose error的方法是来自于superglue的
        T_0to1 是gt
        R,t 是从关键点 cv2.findEssentialMat+cv2.recoverPose 解算出来的帧间pose
        '''
        idx= f_cur.timestamp
        idx1=f_ref.timestamp
        T_c1tow=self.gt_c2w_list[idx]
        T_c0tow=self.gt_c2w_list[idx1]
        
        T_0to1=inv_T(T_c1tow @ inv_T(T_c0tow))

        #self.now_Tcr 是 estimated T_r2c 也就是 T_0to1
        R=self.now_Tcr[:3,:3]
        t=self.now_Tcr[:3,3]
        
        R_gt = T_0to1[:3, :3]
        t_gt = T_0to1[:3, 3]
        error_t = self.angle_error_vec(t, t_gt)
        #print("========> GT translation :",t_gt )
        #print("========> E translation :",t )
        error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
        error_R = self.angle_error_mat(R, R_gt)
        
        return error_t, error_R
    
    def nn_match_two_way(self, desc1, desc2, nn_thresh):
        """
        Performs two-way nearest neighbor matching of two sets of descriptors, such
        that the NN match from descriptor A->B must equal the NN match from B->A.

        Inputs:
        desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
        desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
        nn_thresh - Optional descriptor distance below which is a good match.

        Returns:
        matches - 3xL numpy array, of L matches, where L <= N and each column i is
                    a match of two descriptors, d_i in image 1 and d_j' in image 2:
                    [d_i index, d_j' index, match_score]^T
        """
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        if nn_thresh < 0.0:
            raise ValueError('\'nn_thresh\' should be non-negative')
        # Compute L2 distance. Easy since vectors are unit normalized.
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < nn_thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        return matches 
      
    def match_frames(self, desc1, desc2, ratio_test=2):
        matches=self.nn_match_two_way(self.f_cur.super_des,self.f_ref.super_des,nn_thresh=0.7) #0.7
        #feature_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        #matches = feature_matcher.knnMatch(np.asarray(f1.des,np.float32),np.asarray(f2.des,np.float32), ratio_test)
        
        # out: a vector of match index pairs [idx1[i],idx2[i]] such that the keypoint f1.kps[idx1[i]] is matched with f2.kps[idx2[i]]
        #idx1, idx2 = self.goodMatchesOneToOne(matches, f1.des, f2.des, ratio_test)
        #idx1 = np.asarray(idx1)
        #idx2 = np.asarray(idx2)
        idx1=matches[0].astype(int)
        idx2=matches[1].astype(int)
        #idx1=list(idx1.astype(int))
        #idx2 =list(idx2.astype(int))       
        return idx1, idx2
         
            
    def track_reference_frame(self, f_ref, f_cur, idx, name='match-frame-frame'):
        print('>>>> tracking reference %d ...' %(f_ref.id))        
        if f_ref is None:
            return
        
        # find keypoint matches between f_cur and kf_ref   
        #print('matching keypoints with ', Frame.feature_matcher.type.name)
         
        idxs_cur, idxs_ref = self.match_frames(f_cur.super_des, f_ref.super_des) # 其实用的就是 cv2.FlannBasedMatcher
        self.idxs_ref = idxs_ref 
        self.idxs_cur = idxs_cur
            
        self.num_matched_kps = idxs_cur.shape[0]    
        print("# keypoints matched: %d " % self.num_matched_kps)
        
        # out: Trc  homogeneous transformation matrix with respect to 'ref' frame,  pr_= Trc * pc_
        Trc = self.estimatePose(f_ref.kpsn[idxs_ref], f_cur.kpsn[idxs_cur])
        Tcr = inv_T(Trc)  # Tcr w.r.t. ref frame
        self.now_Tcr=Tcr
        #f_ref.update_pose(np.eye(4))
        f_cur_pose= Tcr @ f_ref.pose        
        f_cur.update_pose(f_cur_pose)
        
        mask_idxs = (self.mask_match.ravel() == 1)
        self.num_inliers = sum(mask_idxs)
        print('# keypoint inliers: ', self.num_inliers )
        
        idx_cur_inliers = idxs_cur[mask_idxs]
        idx_ref_inliers = idxs_ref[mask_idxs]
        #print('#=========debug: idx_cur_inliers: ==========', idx_cur_inliers )
        self.idx_cur_inliers=idx_cur_inliers
        self.idx_ref_inliers=idx_ref_inliers
        # =========================== local map for storing triangulated point ============================ #
        map = Map()
        #f_ref.reset_points()
        #f_cur.reset_points()
        
        kf_ref = KeyFrame(f_ref)
        #kf_ref.id=f_ref.timestamp
        kf_cur = KeyFrame(f_cur, f_cur.img)        
        map.add_keyframe(kf_ref)        
        map.add_keyframe(kf_cur)      
        
        pts3d, mask_pts3d = triangulate_normalized_points(kf_cur.Tcw, kf_ref.Tcw, kf_cur.kpsn[idx_cur_inliers], kf_ref.kpsn[idx_ref_inliers])
        new_pts_count, mask_points, _ = map.add_points(pts3d, mask_pts3d, kf_cur, kf_ref, idx_cur_inliers, idx_ref_inliers, kf_cur.img, do_check=True, cos_max_parallax=Parameters.kCosMaxParallaxInitializer)
        print("# triangulated points: ", new_pts_count) 
        #print("==================== coord kf == f? ==============",kf_cur.kpsn[idx_cur_inliers]==f_cur.kpsn[idx_cur_inliers])
        #上边全是true，放心commit
        
        pts = pts3d[mask_points]
        median_depth = kf_cur.compute_points_median_depth(pts) #30.02
        
        if idx == self.track_every_frame: # 也就是第一次累计到两个track帧
            self.depth_scale = self.median_gt_depth/median_depth
            print('forcing current median depth: ', median_depth,' to median_gt_depth: ',self.median_gt_depth)
             
            tcw_cur = kf_cur.tcw * self.depth_scale  # scale initial baseline 
            kf_cur.update_translation(tcw_cur)
            f_cur.update_translation(tcw_cur)
            
            tcw_ref = kf_ref.tcw * self.depth_scale  # scale initial baseline 
            kf_ref.update_translation(tcw_ref)
            f_ref.update_translation(tcw_ref)
            
        else:
            tcw_cur = kf_cur.tcw * self.depth_scale  # scale initial baseline 
            kf_cur.update_translation(tcw_cur)
            f_cur.update_translation(tcw_cur)
        
        self.now_Tcr[:3,3]=self.now_Tcr[:3,3]* self.depth_scale   
        pts[:,:3] = pts[:,:3] * self.depth_scale  # scale points
        uvs1, depths1 = kf_cur.project_points(pts) # scale 完的point和scale完的平移去投影
        uvs2, depths2 = kf_ref.project_points(pts)
        
        # 上边的空间点先被scale到和深度值同一个尺度
        if self.debug:
            m_gt=self.depth_cur_gt.mean()
            m_cur=np.mean(depths1)
            m_ref=np.mean(depths2)            
            print( f'The mean gt_depth now is: {m_gt} \n',
                  f'The mean kf_cur.project_points now is: {m_cur} \n',
                  f'The mean kf_ref.project_points now is: {m_ref} \n') 

        
        #depths1=depths1*self.depth_scale
        #depths2=depths2*self.depth_scale
        
        self.depth_coord_cur = uvs1
        self.depth_coord_ref = uvs2
        self.depth_cur=depths1
        self.depth_ref=depths2
        
        '''
        point3d=pts3d
        point3d_scaled= point3d[:,:3] * self.scale  # scale points
        cur_z=cur_z * self.scale
        ref_z=ref_z * self.scale
        tcw = kf_cur.tcw * self.scale  # scale initial baseline
        kf_cur.update_translation(tcw)
        f_cur.update_translation(tcw)
        
        self.cur_z=cur_z
        if self.idx==1:
            self.first_z=ref_z
            self.depth_coord_ref=f_ref.kps[idx_cur_inliers]
        
        self.depth_coord_cur=f_cur.kps[idx_cur_inliers]
        #print("============= debug self.depth_coord_cur==========: ", self.depth_coord_cur[0:10])
        #print("====== coordinate check, number matched? ========",len(self.cur_z)==len(self.depth_coord_cur) )  '''
        map.delete()
        #self.f_cur=kf_cur
        #self.f_ref=kf_ref
        
    def compute_pusdo_depth_np(self,depth_coord_cur,depth_cur):
        depth_map=np.random.rand(self.H,self.W)
        depth_map/= 10000
        patch_size=self.depth_patch
        
        for pixel_coord in depth_coord_cur:
            for pixel_depth in depth_cur:
                if pixel_coord[0]-patch_size>0 and pixel_coord[1]-patch_size>0 and  pixel_coord[0]+patch_size<self.H and pixel_coord[1]+patch_size<self.W:
                    row_ind0=int(pixel_coord[0]-patch_size)
                    row_ind1=int(pixel_coord[0]+patch_size)
                    col_ind0=int(pixel_coord[1]-patch_size)
                    col_ind1=int(pixel_coord[1]+patch_size)
                    depth_map[row_ind0:row_ind1,col_ind0:col_ind1]=pixel_depth
                else:
                    if pixel_coord[0]>0 and pixel_coord[1]>0 and  pixel_coord[0]<self.H and pixel_coord[1]<self.W:
                        row_ind=int(pixel_coord[0])
                        col_ind=int(pixel_coord[1])
                        depth_map[row_ind,col_ind]=pixel_depth 
        depth_map=depth_map.astype(np.float32)
        
        if self.viewer_show:
            depth_data = depth_map * self.png_depth_scale/self.scale
            depth_map_u=depth_data.astype(np.uint8)
            depth_map_plot=cv2.applyColorMap(depth_map_u, cv2.COLORMAP_JET)
            cv2.imshow("depth_map",depth_map_plot)
            cv2.waitKey(1)   
        return depth_map
       
    def compute_pusdo_depth_tensor(self,depth_coord_cur,depth_cur):   
        depth_data = self.compute_pusdo_depth_np(depth_coord_cur,depth_cur)
        depth_data = torch.from_numpy(depth_data)
        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            depth_data = F.interpolate(depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
        edge = self.crop_edge
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            depth_data = depth_data[edge:-edge, edge:-edge]
            
        return depth_data
                                         
    def show_superpoint_track(self):
        tracker=self.f_cur.tracker
        tracks=self.f_cur.tracks
        img=cv2.cvtColor(self.f_cur.img, cv2.COLOR_BGR2GRAY)

        pts=self.f_cur.super_pts
        #desc=self.f_cur.super_des
        heatmap=self.f_cur.heatmap
        '''
        # Primary output - Show point tracks overlayed on top of input image.
        out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
        tracks[:, 1] /= float(self.f_cur.fe.nn_thresh) # Normalize track scores to [0,1].
        tracker.draw_tracks(out1, tracks)
        if True:
            cv2.putText(out1, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)
            # Extra output -- Show current point detections.
        '''    
        out2 = (np.dstack((img,img,img)) * 255.).astype('uint8')
        for pt in pts.T:
            pt1 = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
        cv2.putText(out2, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)

        # Extra output -- Show the point confidence heatmap.
        if heatmap is not None:
            min_conf = 0.001
            heatmap[heatmap < min_conf] = min_conf
            heatmap = -np.log(heatmap)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
            out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
            out3 = (out3*255).astype('uint8')
        else:
            out3 = np.zeros_like(out2)
        cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)
        out = np.hstack((out2, out3))
        out = cv2.resize(out, (2*self.W//2, self.H//2))
        return out

    
    def run(self):
        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)
        if self.viewer_show:
            # ================== superpoint 原生 mian一开始的window设置 ================ #
            win='SuperPoint Tracker'
            cv2.namedWindow(win)
            # ======================= pyslam 的设置 =========================== #   
            if platform.system()  == 'Linux':
                from src.utils.display2D import Display2D    
                display2d = Display2D(self.camera.width, self.camera.height)  # pygame interface 
            else: 
                display2d = True  # enable this if you want to use opencv window       
        
        '''
        Tracker 是主动线程：
        
        通过For循环从已经构造好的frameloader中提取idx帧的信息
        并在循环最后赋值给全局变量 self.idx[0] = idx
        这个self.idx[0]在Mapper处就被用来判断是否开启一个Mapping的循环        
        '''
        
        for idx, gt_color, gt_depth, gt_c2w in pbar:
            #rlock.acquire()            
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            
            # markup For Replica room0 test:
            idx = idx[0] # shape []
            gt_depth = gt_depth[0] # shape [680,1200]
            gt_color = gt_color[0] # shape [680,1200,3]
            gt_c2w = gt_c2w[0] # shape [4,4]
            
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            self.depth_cur_gt = gt_depth.clone().cpu()
            
            # ============== 转numpy，为了可视化和frame构造 ============= #
            i=idx.numpy()
            depth_np = gt_depth.to("cpu").numpy()
            color_np = (gt_color.to("cpu").numpy()*255).astype(np.uint8)
            
            z=np.sort(depth_np.reshape(-1))
            
            depth_np = depth_np/np.max(depth_np)*255
            depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)
            depth_np = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
            
            color_np = np.clip(color_np, 0, 255)
            whole = np.concatenate([color_np, depth_np], axis=0)
            H, W, _ = whole.shape
            whole = cv2.resize(whole, (W//4, H//4))
            # ======================== 画图函数逻辑 ===================== # 
            #if self.viewer_show:
            #    cv2.imshow(f'Input RGB-D Sequence', whole[:, :, ::-1])
            #    cv2.waitKey(1)  
            # ======================================================== #
            
            if self.sync_method == 'strict':
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                    #pre_c2w = self.estimate_c2w_list[idx-1].to(device)
            elif self.sync_method == 'loose':
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.1)
            
            #res= feature_tracker.track(image_ref=None, image_cur=color_np, kps_ref=None, des_ref=None)
                        
            # ================================= 初始化 frame._pose，送入cuda ======================================= #
            # 在frame的成员变量pose是w2c
            # 注意坐标系变换问题：
            #   frame的pose是camera_pose这个类：self._pose = CameraPose(pose)
            #   在camera_pose 中，self.Tcw = self._pose.matrix()，homogeneous transformation matrix: (4, 4)
            #   这个Tcw给出的定义是：pc_ = Tcw * pw_ ，也就是w2c，所以，我们gt_pose和NeRF中的优化都是基于c2w的，也就是要有一个inverse
            
            #self.f_cur = Frame(color_np, self.camera, timestamp=i) #在这一刻，帧内的点已经被计算和写好了,track 
            
            """tracked_superpoint=self.show_superpoint_track()
            if self.viewer_show:
                cv2.imshow(win, tracked_superpoint)
                cv2.waitKey(1)
            # ====================================================================================================== #
            cv2.destroyAllWindows()
            print('==> Finshed Demo.')"""
   
            if idx == 0:
                #self.f_ref = self.f_cur
                c2w=gt_c2w
                gt_c2w_np=gt_c2w.cpu().numpy()
                gt_w2c_np=inv_T(gt_c2w_np)
                
                self.f_cur = Frame(color_np, self.camera, timestamp=i)
                
                self.f_cur.update_pose(gt_w2c_np)
                self.frames.append(self.f_cur)
                self.median_gt_depth = z[ ( len(z)-1)//2 ]
                
                self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
                    
            elif idx % self.track_every_frame != 0 or idx == 0: 
                print('This Tracker is waiting track_every_frame: ', self.track_every_frame,',current tracking idx is: ', idx)
                
            elif idx==self.track_every_frame:  # 达到第一个可以track的帧   #idx % self.track_every_frame == 0 and idx != 0: #now set to 5
                # 对 gt 后的下一帧预测 pose 
                self.f_ref = self.frames[-1] # 这时候还没有更新上去最新的帧，所以从deque中获取最后一帧
                # self.f_cur.update_pose(self.f_ref.pose) # 没必要设置上一帧的pose作为这一帧的先验了
                
                self.f_cur = Frame(color_np, self.camera, timestamp=i)
                
                f_ref=self.f_ref
                f_cur=self.f_cur
                #===================================================================================#
                self.track_reference_frame(f_ref, f_cur, idx) #在方程了update到self中去了
                
                error_t, error_R = self.compute_pose_error(f_ref, f_cur)
                ref_idx=f_ref.timestamp
                
                print(f"Translation from {ref_idx} to {idx} "
                        f'translation error: {error_t} ' +
                        f'rotation error: {error_R}')
                
                depth_ref_weak=self.compute_pusdo_depth_tensor(self.depth_coord_ref,self.depth_ref)
                depth_cur_weak=self.compute_pusdo_depth_tensor(self.depth_coord_cur,self.depth_cur)
                
                # ==================== 我们发现depth的tensor太大，没有办法在线程间传递，所以先存下二进制file，到
                '''save_path_0=self.output+'/pseudo_depth/0'
                save_path=self.output+'/pseudo_depth/'+str(idx.numpy())
                os.path.join(self.output, 'pseudo_depth','0')'''
                
                np.save(os.path.join(self.output, 'pseudo_depth','0'), depth_ref_weak)
                np.save(os.path.join(self.output, 'pseudo_depth',str(idx.numpy())), depth_cur_weak)

                
                depth_ref_weak = depth_ref_weak.detach().to(self.device)
                depth_cur_weak = depth_cur_weak.detach().to(self.device)
                self.depth_ref_weak = depth_ref_weak.clone().cpu()
                self.depth_cur_weak = depth_cur_weak.clone().cpu()
                
                if torch.count_nonzero(self.depth_cur_weak).item()!=0 and torch.count_nonzero(self.depth_ref_weak)!=0:
                    Printer.red('===># Message from Tracker: self.depth_cur_weak and self.depth_ref_weak computed')
                
                self.frames.append(self.f_cur)
                
                if len(self.pseudo_depth_maps)==0:
                    self.pseudo_depth_maps.append(self.depth_ref_weak)
                    self.pseudo_depth_maps.append(self.depth_cur_weak)
                else:
                    self.pseudo_depth_maps.append(self.depth_cur_weak)
                 
                # 为了在这一轮将c2w存入self.estimate_c2w_list，供Mapper使用
                c2w_0 = torch.tensor(inv_T(self.f_ref.Tcw)).detach().to(self.device)
                c2w_1 = torch.tensor(inv_T(self.f_cur.Tcw)).detach().to(self.device)
                self.estimate_c2w_list[0] = c2w_0.clone().cpu()
                self.estimate_c2w_list[idx] = c2w_1.clone().cpu()
                
                # ======================== 2D display (image display) ============================== #    
                # img_draw = self.frames[-1].draw_all_feature_trails(color_np)
                img_draw=self.f_cur.draw_all_feature_trails(color_np)
                if self.viewer_show:
                    if display2d is not None:
                        display2d.draw(img_draw)
                    else: 
                        cv2.imshow(cv2.cvtColor(img_draw,cv2.COLOR_RGB2BGR))
                
                '''
                在下边开始进入 MLP 去优化前
                
                对于 init 阶段：
                    1. 通过 track 解算 f_cur 和 f_ref 的 estimate—pose 和 psudo 深度值
                    2. -> Map 我们需要把上边的两帧存储起来，同步到 Mapper 中，做一次高强度的初始优化（self.BA==None 时仅优化NeRF）
                    3. Track <- 把 2.中的 model parameter copy回来, 同步更新一次pose
                对于持续迭代阶段：
                    1. 通过 track 解算 f_cur 的 estimate—pose 和 psudo 深度值
                    2. -> Map 让 Mapper 端存起来，每加入一张就就整体优化一次（这个时候可以开启 BA，使pose和NeRF一起优化）
                    3. Track <- 把 2.中的 model parameter copy回来
                '''
            else:
                # ======================== 当第一个track 完成时，我们开始进入Mapper线程进行init ============================== #
                #if idx == self.track_every_frame:
                Printer.red("# ===>Message from Tracker: Finished first [f_ref,f_cur], Start init in Mapper.")
                while self.init_finished==False:
                    sleep_s = 5
                    Printer.orange("# ===>Message from Tracker: self.init_finished==False, wait {sleep_s}s for Mapper")
                    time.sleep(sleep_s)
                # ===================== 这里开始从 mapper copy 过来初始化好了的参数进行pose优化 =========================== #
                
                if idx > self.track_every_frame:
                    if self.debug:
                        time.sleep(100)
                        Printer.orange("# ===>Message from Tracker: Now idx > self.track_every_frame, sleep for debug")
                        
                        
                    gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                    self.update_para_from_mapping() # 从map端将模型参数复制过来
                    estimated_new_cam_c2w=torch.tensor(inv_T(self.f_cur.Tcw)) #每5帧才做一次矫正
                    camera_tensor = get_tensor_from_camera(estimated_new_cam_c2w.detach())# torch 中的detach可以切断反向回传

                    # 在 cofusion 和 TUM_RGBD 数据集上 平移和旋转向量是分开来被优化的
                    if self.seperate_LR:
                        camera_tensor = camera_tensor.to(device).detach()
                        T = camera_tensor[-3:]
                        quad = camera_tensor[:4]
                        cam_para_list_quad = [quad]
                        quad = Variable(quad, requires_grad=True)
                        T = Variable(T, requires_grad=True)
                        camera_tensor = torch.cat([quad, T], 0)
                        cam_para_list_T = [T]
                        cam_para_list_quad = [quad]
                        optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},
                                                            {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}])               

                    else:               
                    # 这边的 pose loss 还是在用 gt supervise 啊？gt_camera_tensor.to(device)-camera_tensor
                        camera_tensor = Variable(camera_tensor.to(device), requires_grad=True)
                        cam_para_list = [camera_tensor]
                        optimizer_camera = torch.optim.Adam(cam_para_list, lr=self.cam_lr)
                    
                    # 这边的 initial loss 有 pose 的真值去 supervise
                    initial_loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor).mean().item()                
                    
                    candidate_cam_tensor = None
                    current_min_loss = 10000000000.

                    # iters=50         
                    for cam_iter in range(self.num_cam_iters):
                        if self.seperate_LR:
                            camera_tensor = torch.cat([quad, T], 0).to(self.device)

                        self.visualizer.vis(
                            idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)
                        # ============================ To be change: what is (cur_z)?? =============================== #                
                        if self.weak_depth:
                            gt_depth_weak = self.depth_cur_weak
                            tracking_pixels = len(self.depth_cur)
                            loss = self.optimize_cam_in_batch(
                                camera_tensor, gt_color, gt_depth_weak, tracking_pixels *3 , optimizer_camera,cam_iter)
                        # ============================================================================================== #                
                        else:    
                            loss = self.optimize_cam_in_batch(
                                camera_tensor, gt_color, gt_depth, self.tracking_pixels , optimizer_camera,cam_iter)
                            
                        if cam_iter == 0:
                            initial_loss = loss

                        loss_camera_tensor = torch.abs(
                            gt_camera_tensor.to(device)-camera_tensor).mean().item()
                        if self.verbose:
                            if cam_iter == self.num_cam_iters-1:
                                print(
                                    # 这里播报的loss，是加入当前帧优化的时候，模型loss的变化
                                    f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                    f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                        if loss < current_min_loss:
                            current_min_loss = loss
                            candidate_cam_tensor = camera_tensor.clone().detach()

                    bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                        [1, 4])).type(torch.float32).to(self.device)
                    c2w = get_camera_from_tensor(
                        candidate_cam_tensor.clone().detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    self.f_cur.update_pose(inv_T(c2w.cpu().numpy()))
                
            # 一个for循环一定产生一个c2w：
                # 对于idx==0，c2w就是 gt_pose
                # 对于idx==self.track_every_frame，c2w是通过本质矩阵解算的
                # 对于idx>self.track_every_frame, 正在debug，还没写进去
            
            sleep_wait_map=2
            
            # 这里两个if，是在关键的时候锁死idx的更新，等mapper端优化完再开始idx更新
            if idx % self.every_frame==0 and idx!=self.track_every_frame and idx!=0:    
                self.estimate_c2w_list[idx] = c2w.clone().cpu()
                self.idx[0] = idx
                Printer.red(f'===># Message from Tracker: estimate_c2w_list updated with every_frame, sleep {sleep_wait_map}s for Mapper')
                # 缺一个非第一帧 mapper 优化完的flag 逻辑！
                while not self.mapper_finished:
                    time.sleep(sleep_wait_map)                
            elif idx==self.every_frame:
                self.idx[0] = idx 
                Printer.red(f'===># Message from Tracker: estimate_c2w_list updated with 2 global prior! sleep {sleep_wait_map}s for Mapper init!')
                while not self.init_finished:
                    time.sleep(sleep_wait_map)
            else:
                self.idx[0] = idx
            
            #rlock.release()
            '''if self.low_gpu_mem:
                torch.cuda.empty_cache()'''
            
              





# 下面是match的插件逻辑

'''def goodMatchesOneToOne(self, matches, des1, des2, ratio_test=None):
    len_des2 = len(des2)
    idx1, idx2 = [], []  
    # good_matches = []           
    if ratio_test is None: 
        ratio_test = self.ratio_test
    if matches is not None:         
        float_inf = float('inf')
        dist_match = defaultdict(lambda: float_inf)   
        index_match = dict()  
        for m, n in matches:
            if m.distance > ratio_test * n.distance:
                continue     
            dist = dist_match[m.trainIdx]
            if dist == float_inf: 
                # trainIdx has not been matched yet
                dist_match[m.trainIdx] = m.distance
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                index_match[m.trainIdx] = len(idx2)-1
            else:
                if m.distance < dist: 
                    # we have already a match for trainIdx: if stored match is worse => replace it
                    #print("double match on trainIdx: ", m.trainIdx)
                    index = index_match[m.trainIdx]
                    assert(idx2[index] == m.trainIdx) 
                    idx1[index]=m.queryIdx
                    idx2[index]=m.trainIdx                        
    return idx1, idx2     '''