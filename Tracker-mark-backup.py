import copy
import os
import time
import numpy as np
import torch
import cv2
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import platform 
if platform.system()  == 'Linux':     
    from src.utils.display2D import Display2D  #  !NOTE: pygame generate troubles under macOS!
    
from src.common import (get_camera_from_tensor, get_tensor_from_camera,get_samples)

from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

# superpoint tracker
from src.superpoint_tracker import SuperpointTracker
from src.frame import Frame, match_frames
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
        self.weak_depth = cfg['weak_depth']
        self.depth_patch = cfg['depth_patch']
        
        self.depth_supervision= slam.depth_supervision
        self.bundle_loss=slam.bundle_loss
        self.camera=slam.camera
        
        self.frames=slam.frames
        self.f_cur = slam.f_cur
        self.idxs_cur = slam.idxs_cur
        
        
        self.f_ref = slam.f_ref 
        self.idxs_ref = slam.idxs_ref
        
        self.idx_cur_inliers=None
        self.idx_ref_inliers=None
        
        self.depth_coord_cur=None
        self.depth_coord_ref=None
        
        self.cur_z=None
        self.first_z=None
        
        self.dyn_config = SLAMDynamicConfig()
        #self.map=slam.map
        self.num_inliers=None # current number of matched points 
        self.mask_match=None
        
        self.viewer_show=cfg['viewer_show']
        #=====================================================#
        
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
        indices=np.where(depth_np>4)
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
        # 这里，我们设batch_size为len(self.cur_z)
        
        if self.weak_depth: #self.get_samples
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = self.get_samples(
                Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color)
        else: #get_samples from common.py
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color)
        print("============ sample debug:============",batch_rays_o[0], batch_rays_d[0], batch_gt_depth[0], batch_gt_color[0])    
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
                print("====================This tracker is runing WITH depth supervision=================")    
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

    def bundle_loss():
        loss=None
        return  loss


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
    
    #update current pose & triangulate point        
    def track_reference_frame(self, f_ref, f_cur, name='match-frame-frame'):
        print('>>>> tracking reference %d ...' %(f_ref.id))        
        if f_ref is None:
            return 
        # find keypoint matches between f_cur and kf_ref   
        print('matching keypoints with ', Frame.feature_matcher.type.name)              
        idxs_cur, idxs_ref = match_frames(f_cur, f_ref)
        self.idxs_ref = idxs_ref 
        self.idxs_cur = idxs_cur
            
        self.num_matched_kps = idxs_cur.shape[0]    
        print("# keypoints matched: %d " % self.num_matched_kps)
        
        # out: Trc  homogeneous transformation matrix with respect to 'ref' frame,  pr_= Trc * pc_
        Trc = self.estimatePose(f_ref.kpsn[idxs_ref], f_cur.kpsn[idxs_cur])
        Tcr = inv_T(Trc)  # Tcr w.r.t. ref frame 
        #f_ref.update_pose(np.eye(4))
        f_cur_pose= Tcr@f_ref.pose        
        f_cur.update_pose(f_cur_pose)
        
        mask_idxs = (self.mask_match.ravel() == 1)
        self.num_inliers = sum(mask_idxs)
        print('# keypoint inliers: ', self.num_inliers )
        
        idx_cur_inliers = idxs_cur[mask_idxs]
        idx_ref_inliers = idxs_ref[mask_idxs]
        self.idx_cur_inliers=idx_cur_inliers
        self.idx_ref_inliers=idx_ref_inliers
        
        # =========================== local map for storing triangulated point ============================ #
        map = Map()
        f_ref.reset_points()
        f_cur.reset_points()
        
        kf_ref = KeyFrame(f_ref)
        kf_cur = KeyFrame(f_cur, f_cur.img)        
        map.add_keyframe(kf_ref)        
        map.add_keyframe(kf_cur)      
        
        pts3d, mask_pts3d,cur_z,ref_z = triangulate_normalized_points(kf_cur.Tcw, kf_ref.Tcw, kf_cur.kpsn[idx_cur_inliers], kf_ref.kpsn[idx_ref_inliers])

        new_pts_count, mask_points, _ = map.add_points(pts3d, mask_pts3d, kf_cur, kf_ref, idx_cur_inliers, idx_ref_inliers, f_cur.img, do_check=True, cos_max_parallax=Parameters.kCosMaxParallaxInitializer)
        print("#===== triangulated points: =======", new_pts_count)
        
        #print("==================== coord kf == f? ==============",kf_cur.kpsn[idx_cur_inliers]==f_cur.kpsn[idx_cur_inliers])
        #上边全是true，放心commit
    
        # 计算 reprojected depth
        point3d=pts3d[mask_points]
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
        #print("====== coordinate check, number matched? ========",len(self.cur_z)==len(self.depth_coord_cur) )  
        map.delete()

        self.f_cur=f_cur
        self.f_ref=f_ref
        
    def compute_pusdo_depth_np(self):
        #self.points = np.array([None]*len(self.kpsu))
        depth_map=np.zeros((self.H,self.W))
        patch_size=self.depth_patch
        print("====== coordinate check, number matched? ========",len(self.cur_z)==len(self.depth_coord_cur) )
        
        for pixel_coord in self.depth_coord_cur:
            for pixel_depth in self.cur_z:
                if pixel_coord[0]-patch_size>=0 and pixel_coord[1]-patch_size>=0 and  pixel_coord[0]+patch_size<=self.H and pixel_coord[1]+patch_size<=self.W:
                    row_ind0=int(pixel_coord[0]-patch_size)
                    row_ind1=int(pixel_coord[0]+patch_size)
                    col_ind0=int(pixel_coord[1]-patch_size)
                    col_ind1=int(pixel_coord[1]+patch_size)
                    depth_map[row_ind0:row_ind1,col_ind0:col_ind1]=pixel_depth
                else:
                    if pixel_coord[0]>=0 and pixel_coord[1]>=0 and  pixel_coord[0]<=self.H and pixel_coord[1]<=self.W:
                        row_ind=int(pixel_coord[0])
                        col_ind=int(pixel_coord[1])
                        depth_map[row_ind,col_ind]=pixel_depth 
        depth_map=depth_map.astype(np.float32)
        
        if self.viewer_show:
            cv2.imshow(cv2.cvtColor(depth_map,cv2.COLOR_RGB2BGR))   
        return depth_map
       
    def compute_pusdo_depth_tensor(self):   
        depth_data = self.compute_pusdo_depth_np()
        depth_data = torch.from_numpy(depth_data)
        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            depth_data = F.interpolate(depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
        edge = self.crop_edge
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            depth_data = depth_data[edge:-edge, edge:-edge]
            
        return depth_data.to(self.device)
                                         

    def run(self):
        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)
        if self.viewer_show:   
            if platform.system()  == 'Linux':    
                display2d = Display2D(self.camera.width, self.camera.height)  # pygame interface 
            else: 
                display2d = True  # enable this if you want to use opencv window
        
        #viewer3D = Viewer3D()
        #matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')        
            
        # ============== superpoint tracker loaded ============= #
        feature_tracker = SuperpointTracker()
        self.f_cur.set_tracker(feature_tracker)
        # feature_mapper = Map()
        # ======================================================= #

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            # For Replica room0 test:
            idx = idx[0] # shape []
            #self.idx=idx
            gt_depth = gt_depth[0] # shape [680,1200]
            gt_color = gt_color[0] # shape [680,1200,3]
            gt_c2w = gt_c2w[0] # shape [4,4]
            
            # ============== 转numpy，为了可视化和frame构造 ============= #
            i=idx.numpy()
            depth_np = gt_depth.to("cpu").numpy()
            color_np = (gt_color.to("cpu").numpy()*255).astype(np.uint8)
            depth_np = depth_np/np.max(depth_np)*255
            depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)
            depth_np = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
            color_np = np.clip(color_np, 0, 255)
            
            whole = np.concatenate([color_np, depth_np], axis=0)
            H, W, _ = whole.shape
            whole = cv2.resize(whole, (W//4, H//4))
            #cv2.imshow(f'Input RGB-D Sequence', whole[:, :, ::-1])
            #cv2.waitKey(1)
            # ======================================================= #
            
            # ================================= 构造frame类别 ======================================= #
            # 在构造的时候，就已经完成了 detectAndCompute（keypoints and descriptors）这一步了，接下来都是 query 逻辑
            #self.f_cur = Frame(color_np, self.camera, timestamp=i)
            # reset indexes of matches 
            #if idx == 0:
            #    self.f_ref = self.f_cur
            #else:
            #    self.f_ref = self.frames[-1]  # take last frame in the buffer
            #self.frames.append(self.f_cur)    
            #point_tracker_img = self.f_cur.draw_all_feature_trails(color_np)
            #cv2.imshow(f'superpoint RGB Sequence',cv2.cvtColor(point_tracker_img,cv2.COLOR_RGB2BGR))
            #cv2.waitKey(1) 
            # ======================================================================================= #

            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            
            # ================================= 初始化 frame._pose，送入cuda ======================================= #
            # 在frame的成员变量pose是w2c
            # 注意坐标系变换问题：
            #   frame的pose是camera_pose这个类：self._pose = CameraPose(pose)
            #   在camera_pose 中，self.Tcw = self._pose.matrix()，homogeneous transformation matrix: (4, 4)
            #   这个Tcw给出的定义是：pc_ = Tcw * pw_ ，也就是w2c，所以，我们gt_pose和NeRF中的优化都是基于c2w的，也就是要有一个inverse
            self.f_cur = Frame(color_np, self.camera, timestamp=i)
            #point_tracker_img = self.f_cur.draw_all_feature_trails(color_np)
            #cv2.imshow(f'superpoint RGB Sequence',cv2.cvtColor(point_tracker_img,cv2.COLOR_RGB2BGR))
            #cv2.waitKey(1)
             
            self.idxs_ref = [] 
            self.idxs_cur = [] 
            if idx == 0:
                #self.f_ref = self.f_cur
                gt_c2w_np=gt_c2w.cpu().numpy()
                gt_w2c_np=inv_T(gt_c2w_np)
                self.f_cur.update_pose(gt_w2c_np)
                self.frames.append(self.f_cur)
            else:
                # 对gt后的下一帧预测pose 
                self.f_ref = self.frames[-1] # 这时候还没有更新上去最新的帧，所以从deque中获取最后一帧
                #self.f_cur.update_pose(self.f_ref.pose) # 没必要设置上一帧的pose作为这一帧的先验了
                
                f_ref=self.f_ref
                f_cur=self.f_cur
                #===================================================================================#
                self.track_reference_frame(f_ref, f_cur) #在方程了update到self中去了
                #===================================================================================#
                
                depth_cur_weak=self.compute_pusdo_depth_tensor()
                self.frames.append(self.f_cur)
                
                #2D display (image display)    
                img_draw = self.frames[-1].draw_all_feature_trails(color_np)
                if self.viewer_show:
                    if display2d is not None:
                        display2d.draw(img_draw)
                    else: 
                        cv2.imshow(cv2.cvtColor(img_draw,cv2.COLOR_RGB2BGR))
                        
            # ============================= 构造待优化的 estimate_c2w_list[idx],也就是estimated_new_cam_c2w =============================== #
            # ============================= 但是注意，这个idx是每五个帧才被优化一次的！ =============================== #
            #在这里开了strict，就是严格先mapping后tracking
            if self.sync_method == 'strict':
                # self.every_frame frames 就是每5帧
                # 初始化后，达到比如说第6帧(同时确定一下现在map的idx不是6)，则将estimate_c2w_list[6]推到device中，待优化
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                    #pre_c2w = self.estimate_c2w_list[idx-1].to(device)#NICE_SLAM.py中已初始化           
            '''
            elif self.sync_method == 'loose':
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.1)
            elif self.sync_method == 'free':
                # pure parallel, if mesh/vis happens may cause inbalance
                pass
            '''
            # 从map端将模型参数复制过来 self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            self.update_para_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)
                
            '''
            if idx == 0 or self.gt_camera: #False
                # 这里的第一帧用了数据集的真值。
                c2w = gt_c2w
                
                if 1:#not self.no_vis_on_first_frame
                    self.visualizer.vis(
                        idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders)
            '''
                
            if idx>0:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                estimated_new_cam_c2w=torch.tensor(inv_T(self.f_cur.Tcw))
                camera_tensor = get_tensor_from_camera(estimated_new_cam_c2w.detach())# torch 中的detach可以切断反向回传
                '''
                if self.const_speed_assumption and idx-1 >= 0: # const_speed_assumption： true
                    # 在选择线程交互模式的时候：                    
                    # pre_c2w = self.estimate_c2w_list[idx-1]
                    pre_c2w = pre_c2w.float()
                    #这里的delta计算idx-1和idx-2帧间的位姿“差”
                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        device).float().inverse()
                    
                    # 把idx-1和idx-2帧间的位姿“差” 左乘到代表idx-1的pre_c2w？？
                    # 因为是匀速模型假设所以认为estimated可以这么估计？
                    # 这样的确提升的空间很大啊
                    # estimated_new_cam_c2w就是代表idx的位姿
                    estimated_new_cam_c2w = delta@pre_c2w
                else:
                    estimated_new_cam_c2w = pre_c2w
                '''      
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
                # 这边的pose loss还是在用gt supervise啊？gt_camera_tensor.to(device)-camera_tensor
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

                    if self.weak_depth:
                        loss = self.optimize_cam_in_batch(
                            camera_tensor, gt_color, depth_cur_weak, len(self.cur_z), optimizer_camera,cam_iter)
                    
                '''    
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
            
            #在参数表里gt_camera 开True就直接到这儿了
            #else，优化完的estimate pose被推到数据队列的idx上，并把 gt pose 也正式记录一下
            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            pre_c2w = c2w.clone()
            '''
            # 在nice-slam初始化逻辑：
            self.idx = torch.zeros((1)).int()
            self.idx.share_memory_()
            

            self.idx[0] = idx
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
