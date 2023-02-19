import copy
import os
import time
import numpy as np
import torch
import cv2
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

# superpoint tracker
from src.superpoint_tracker import SuperpointTracker
from src.frame import Frame, match_frames


class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']
        
        #=====================================================#
        self.depth_supervision= slam.depth_supervision
        self.bundle_loss=slam.bundle_loss
        self.camera=slam.camera
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
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2w_list.share_memory_()
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
        
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
        
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

    def run(self):
        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)
            
        # ============== superpoint tracker loaded ============= #
        feature_tracker = SuperpointTracker()
        Frame.set_tracker(feature_tracker)
        # ======================================================= #

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            # For Replica room0 test:
            idx = idx[0] # shape []
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
            # cv2.imshow(f'Input RGB-D Sequence', whole[:, :, ::-1])
            # cv2.waitKey(1)
            # ======================================================= #
            
            # ==================== 构造frame类别 ===================== #
            f_cur = Frame(color_np, self.camera, timestamp=i)
            point_tracker_img = f_cur.draw_all_feature_trails(color_np)
            cv2.imshow(f'superpoint RGB Sequence',cv2.cvtColor(point_tracker_img,cv2.COLOR_RGB2BGR))
            cv2.waitKey(1) 
            # ======================================================= #
            
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
                
            #在这里开了strict，就是严格先mapping后tracking（To Be modified）
            if self.sync_method == 'strict':
                # self.every_frame frames 就是每5帧
                # 初始化后，达到比如说第6帧(同时确定一下现在map的idx不是6)，则将estimate_c2w_list[6]推到device中，待优化
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                    pre_c2w = self.estimate_c2w_list[idx-1].to(device)#NICE_SLAM.py中已初始化
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

            self.update_para_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera: #False
                # 这里的第一帧用了数据集的真值。
                c2w = gt_c2w
                '''
                if 1:#not self.no_vis_on_first_frame
                    self.visualizer.vis(
                        idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders)
                '''
                
            else:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                if self.const_speed_assumption and idx-2 >= 0: # const_speed_assumption： true
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
                
                # torch 中的detach可以切断反向回传
                camera_tensor = get_tensor_from_camera(
                    estimated_new_cam_c2w.detach())
                
                '''
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
                '''

                # else:               
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

                    loss = self.optimize_cam_in_batch(
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera,cam_iter)

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
            '''

            self.idx[0] = idx
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
