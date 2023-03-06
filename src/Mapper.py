import glob
import os
import time, threading
rlock = threading.RLock()

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera, random_select)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer
from src.utils.utils_sys import Printer
from tqdm import tqdm



class Mapper(object):
    """
    Mapper thread. Note that coarse mapper also uses this code.

    """

    def __init__(self, cfg, args, slam, coarse_mapper=False
                 ):

        self.cfg = cfg
        self.args = args
        self.coarse_mapper = coarse_mapper
        
        #=====================================================#
        self.init_finished = slam.init_finished

        self.depth_supervision= slam.depth_supervision
        self.bundle_loss=slam.bundle_loss
        self.camera=slam.camera
        
        self.frames=slam.frames
        self.pseudo_depth_maps=slam.pseudo_depth_maps

        self.depth_coord_cur=slam.depth_coord_cur
        self.depth_coord_ref=slam.depth_coord_ref
        self.depth_cur=slam.depth_cur
        self.depth_ref=slam.depth_ref
        
        self.depth_cur_weak=slam.depth_cur_weak
        self.depth_ref_weak=slam.depth_ref_weak
        
        self.weak_depth = cfg['weak_depth']
        self.debug_tracker_sperpoint=cfg['debug_tracker_sperpoint']
        self.track_every_frame=cfg['track_every_frame']
        
        self.mapper_finished = slam.mapper_finished
        self.idx0=slam.idx0
        
        self.depth_0 = None
        self.depth_every = None
        self.depth_idx = None
        #=====================================================#

        self.idx = slam.idx
        self.nice = slam.nice
        self.c = slam.shared_c
        self.bound = slam.bound
        self.logger = slam.logger
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.renderer = slam.renderer
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.mapping_first_frame = slam.mapping_first_frame

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.device = cfg['mapping']['device']
        self.fix_fine = cfg['mapping']['fix_fine']
        self.eval_rec = cfg['meshing']['eval_rec']
        self.BA = True  # Even if BA is enabled, it starts only when there are at least 4 keyframes
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']
        self.mesh_freq = 50 #cfg['mapping']['mesh_freq'] # 突然传参失败，why？
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.fix_color = cfg['mapping']['fix_color']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.num_joint_iters = cfg['mapping']['iters']
        self.clean_mesh = cfg['meshing']['clean_mesh']
        self.every_frame = cfg['mapping']['every_frame']
        self.color_refine = cfg['mapping']['color_refine']
        self.w_color_loss = cfg['mapping']['w_color_loss']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.fine_iter_ratio = cfg['mapping']['fine_iter_ratio']
        self.middle_iter_ratio = cfg['mapping']['middle_iter_ratio']
        self.mesh_coarse_level = cfg['meshing']['mesh_coarse_level']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        #self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.save_selected_keyframes_info = cfg['mapping']['save_selected_keyframes_info']
        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}
        '''
        if self.nice:
            if coarse_mapper:
                self.keyframe_selection_method = 'global'
        '''
        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        if 'Demo' not in self.output:  # disable this visualization in demo
            # 传参失败cfg['mapping']['vis_freq']
            self.visualizer = Visualizer(freq=5, inside_freq=cfg['mapping']['vis_inside_freq'],
                                         vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                         verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.

        Args:
            c2w (tensor): camera pose of current frame.
            key (str): name of this feature grid.
            val_shape (tensor): shape of the grid.
            depth_np (numpy.array): depth image of current frame.

        Returns:
            mask (tensor): mask for selected optimizable feature.
            points (tensor): corresponding point coordinates.
        """
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        X, Y, Z = torch.meshgrid(torch.linspace(self.bound[0][0], self.bound[0][1], val_shape[2]),
                                 torch.linspace(self.bound[1][0], self.bound[1][1], val_shape[1]),
                                 torch.linspace(self.bound[2][0], self.bound[2][1], val_shape[0]))

        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        if key == 'grid_coarse':
            mask = np.ones(val_shape[::-1]).astype(np.bool)
            return mask
        points_bak = points.clone()
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        cam_cord[:, 0] *= -1
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                 uv[i:i+remap_chunk, 0],
                                 uv[i:i+remap_chunk, 1],
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)

        edge = 0
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        # For ray with depth==0, fill it with maximum depth
        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        # depth test
        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths+0.5)
        mask = mask.reshape(-1)

        # add feature grid near cam center
        ray_o = c2w[:3, 3]
        ray_o = torch.from_numpy(ray_o).unsqueeze(0)

        dist = points_bak-ray_o
        dist = torch.sum(dist*dist, axis=1)
        mask2 = dist < 0.5*0.5
        mask2 = mask2.cpu().numpy()
        mask = mask | mask2

        points = points[mask]
        mask = mask.reshape(val_shape[2], val_shape[1], val_shape[0])
        return mask
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
        indices=np.where(depth_np>0.01) #originallly 4
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
    
    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, keyframe_dict, k, N_samples=16, pixels=100):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        if self.weak_depth:
            rays_o, rays_d, gt_depth, gt_color = self.get_samples(
            0, H, 0, W, pixels, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color)
        else:
            rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, pixels, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
        
        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        print("============= sample debug:near,far==========", near, far)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples, 3]
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            c2w = keyframe['est_c2w'].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [vertices, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
            cam_cord_homo = w2c@homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
            cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum()/uv.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        selected_keyframe_list = [dic['id']
                                  for dic in list_keyframe if dic['percent_inside'] > 0.00]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])
        return selected_keyframe_list

    def optimize_map(self, num_joint_iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, keyframe_dict, keyframe_list, cur_c2w):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enabled).

        Args:
            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            # 还是用了GT pose 来优化啊
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list of keyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        c = self.c
        cfg = self.cfg
        device = self.device
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
            [1, 4])).type(torch.float32).to(device)

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global': # imap
                num = self.mapping_window_size-2
                #这里用random select 啊？
                optimize_frame = random_select(len(self.keyframe_dict)-1, num)
                
            elif self.keyframe_selection_method == 'overlap': # nice-slam
                num = self.mapping_window_size-2
                optimize_frame = self.keyframe_selection_overlap(
                    cur_gt_color, cur_gt_depth, cur_c2w, keyframe_dict[:-1], num)

        # add the last keyframe and the current frame(use -1 to denote)
        oldest_frame = None
        if len(keyframe_list) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list)-1]
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1]

        if self.save_selected_keyframes_info:
            keyframes_info = []
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    frame_idx = keyframe_list[frame]
                    tmp_gt_c2w = keyframe_dict[frame]['gt_c2w']
                    tmp_est_c2w = keyframe_dict[frame]['est_c2w']
                else:
                    frame_idx = idx
                    tmp_gt_c2w = gt_cur_c2w
                    tmp_est_c2w = cur_c2w
                keyframes_info.append(
                    {'idx': frame_idx, 'gt_c2w': tmp_gt_c2w, 'est_c2w': tmp_est_c2w})
            self.selected_keyframes[idx] = keyframes_info

        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        decoders_para_list = []
        
        '''
        coarse_grid_para = []
        middle_grid_para = []
        fine_grid_para = []
        color_grid_para = []
        gt_depth_np = cur_gt_depth.cpu().numpy()
        
        if self.nice:
            if self.frustum_feature_selection:
                masked_c_grad = {}
                mask_c2w = cur_c2w
                
            for key, val in c.items():
                if not self.frustum_feature_selection:
                    val = Variable(val.to(device), requires_grad=True)
                    c[key] = val
                    if key == 'grid_coarse':
                        coarse_grid_para.append(val)
                    elif key == 'grid_middle':
                        middle_grid_para.append(val)
                    elif key == 'grid_fine':
                        fine_grid_para.append(val)
                    elif key == 'grid_color':
                        color_grid_para.append(val)

                else:
                    mask = self.get_mask_from_c2w(
                        mask_c2w, key, val.shape[2:], gt_depth_np)
                    mask = torch.from_numpy(mask).permute(2, 1, 0).unsqueeze(
                        0).unsqueeze(0).repeat(1, val.shape[1], 1, 1, 1)
                    val = val.to(device)
                    # val_grad is the optimizable part, other parameters will be fixed
                    val_grad = val[mask].clone()
                    val_grad = Variable(val_grad.to(
                        device), requires_grad=True)
                    masked_c_grad[key] = val_grad
                    masked_c_grad[key+'mask'] = mask
                    if key == 'grid_coarse':
                        coarse_grid_para.append(val_grad)
                    elif key == 'grid_middle':
                        middle_grid_para.append(val_grad)
                    elif key == 'grid_fine':
                        fine_grid_para.append(val_grad)
                    elif key == 'grid_color':
                        color_grid_para.append(val_grad)

        if self.nice:
            if not self.fix_fine:
                decoders_para_list += list(
                    self.decoders.fine_decoder.parameters())
            if not self.fix_color:
                decoders_para_list += list(
                    self.decoders.color_decoder.parameters())
        else:
        '''
        
            # imap*, single MLP (就是一开始在Nice-SLAM中load进来的东西)
        decoders_para_list += list(self.decoders.parameters())
        
        if self.BA:
            camera_tensor_list = []
            gt_camera_tensor_list = []
            for frame in optimize_frame:
                # the oldest frame should be fixed to avoid drifting
                if frame != oldest_frame:
                    if frame != -1:
                        c2w = keyframe_dict[frame]['est_c2w']
                        gt_c2w = keyframe_dict[frame]['gt_c2w']
                    else:
                        c2w = cur_c2w
                        gt_c2w = gt_cur_c2w
                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor)
                    gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                    gt_camera_tensor_list.append(gt_camera_tensor)
            '''
        if self.nice:
            if self.BA:
                # The corresponding lr will be set according to which stage the optimization is in
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': coarse_grid_para, 'lr': 0},
                                              {'params': middle_grid_para, 'lr': 0},
                                              {'params': fine_grid_para, 'lr': 0},
                                              {'params': color_grid_para, 'lr': 0},
                                              {'params': camera_tensor_list, 'lr': 0}])
            else:
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': coarse_grid_para, 'lr': 0},
                                              {'params': middle_grid_para, 'lr': 0},
                                              {'params': fine_grid_para, 'lr': 0},
                                              {'params': color_grid_para, 'lr': 0}])
            '''
        else:

            # imap*, single MLP
            if self.BA:
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': camera_tensor_list, 'lr': 0}])
            else:
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0}])
                
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=200, gamma=0.8)

        for joint_iter in tqdm(range(num_joint_iters)):
            '''
            if self.nice:
                if self.frustum_feature_selection:
                    for key, val in c.items():
                        if (self.coarse_mapper and 'coarse' in key) or \
                                ((not self.coarse_mapper) and ('coarse' not in key)):
                            val_grad = masked_c_grad[key]
                            mask = masked_c_grad[key+'mask']
                            val = val.to(device)
                            val[mask] = val_grad
                            c[key] = val

                if self.coarse_mapper:
                    self.stage = 'coarse'
                elif joint_iter <= int(num_joint_iters*self.middle_iter_ratio):
                    self.stage = 'middle'
                elif joint_iter <= int(num_joint_iters*self.fine_iter_ratio):
                    self.stage = 'fine'
                else:
                    self.stage = 'color'

                optimizer.param_groups[0]['lr'] = cfg['mapping']['stage'][self.stage]['decoders_lr']*lr_factor
                optimizer.param_groups[1]['lr'] = cfg['mapping']['stage'][self.stage]['coarse_lr']*lr_factor
                optimizer.param_groups[2]['lr'] = cfg['mapping']['stage'][self.stage]['middle_lr']*lr_factor
                optimizer.param_groups[3]['lr'] = cfg['mapping']['stage'][self.stage]['fine_lr']*lr_factor
                optimizer.param_groups[4]['lr'] = cfg['mapping']['stage'][self.stage]['color_lr']*lr_factor
                if self.BA:
                    if self.stage == 'color':
                        optimizer.param_groups[5]['lr'] = self.BA_cam_lr
            else:
            '''
            self.stage = 'color'
            optimizer.param_groups[0]['lr'] = cfg['mapping']['imap_decoders_lr']
            
            if self.BA:
                optimizer.param_groups[1]['lr'] = self.BA_cam_lr
                

            #if (not (idx == 0 and self.no_vis_on_first_frame)) and ('Demo' not in self.output):
            self.visualizer.vis(idx.clone().to(self.device), joint_iter, cur_gt_depth.clone().to(self.device), 
                                cur_gt_color.clone().to(self.device), cur_c2w.clone().to(self.device), 
                                self.c, self.decoders)

            optimizer.zero_grad()
            batch_rays_d_list = []
            batch_rays_o_list = []
            batch_gt_depth_list = []
            batch_gt_color_list = []

            camera_tensor_id = 0
            for frame in optimize_frame:
                if frame != -1:
                    if self.weak_depth:
                        gt_depth = keyframe_dict[frame]['pseudo_depth'].to(device)
                    else:
                        gt_depth = keyframe_dict[frame]['depth'].to(device)
                            
                    gt_color = keyframe_dict[frame]['color'].to(device)
                    
                    if self.BA and frame != oldest_frame:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        camera_tensor_id += 1
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                    
                        c2w = keyframe_dict[frame]['est_c2w']

                else:
                    gt_depth = cur_gt_depth.to(device)
                    gt_color = cur_gt_color.to(device)
                    
                    if self.BA:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                      
                        c2w = cur_c2w
                if self.idx==0:
                    batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                        0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color,device=self.device)
                else:
                    if self.idx >0 & self.weak_depth:
                        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = self.get_samples(
                        0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color)
                    if self.idx >0 & self.weak_depth is not True:
                        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                        0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color,device=self.device)
                    
                    
                batch_rays_o_list.append(batch_rays_o.float())
                batch_rays_d_list.append(batch_rays_d.float())
                batch_gt_depth_list.append(batch_gt_depth.float())
                batch_gt_color_list.append(batch_gt_color.float())

            batch_rays_d = torch.cat(batch_rays_d_list)
            batch_rays_o = torch.cat(batch_rays_o_list)
            batch_gt_depth = torch.cat(batch_gt_depth_list)
            batch_gt_color = torch.cat(batch_gt_color_list)
            
            '''
            if self.nice:
                # should pre-filter those out of bounding box depth value
                with torch.no_grad():
                    det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    t = (self.bound.unsqueeze(0).to(
                        device)-det_rays_o)/det_rays_d
                    t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                    inside_mask = t >= batch_gt_depth
                batch_rays_d = batch_rays_d[inside_mask]
                batch_rays_o = batch_rays_o[inside_mask]
                batch_gt_depth = batch_gt_depth[inside_mask]
                batch_gt_color = batch_gt_color[inside_mask]
            '''
            
            ret = self.renderer.render_batch_ray(c, self.decoders, batch_rays_d,
                                                 batch_rays_o, device, self.stage,batch_gt_depth)
                                                 #gt_depth=None if self.coarse_mapper else batch_gt_depth)
            depth, uncertainty, color = ret

            depth_mask = (batch_gt_depth > 0)
            
            # Depth supervision 选择回路
            if not self.depth_supervision:
                if joint_iter == num_joint_iters-1:
                    print("self.depth_supervision:",self.depth_supervision)
                    print("==================== per is runing WITHOUT depth supervision =================")
                if ((not self.nice) or (self.stage == 'color')):
                    loss = torch.abs(batch_gt_color - color).sum()
            else:
                if joint_iter == num_joint_iters-1:
                    print("self.depth_supervision:",self.depth_supervision)
                    print("====================This Mapper is runing WITH depth supervision=================")    
                loss = torch.abs(
                    batch_gt_depth[depth_mask]-depth[depth_mask]).sum()
                if ((not self.nice) or (self.stage == 'color')):
                    color_loss = torch.abs(batch_gt_color - color).sum()
                    weighted_color_loss = self.w_color_loss*color_loss
                    loss += weighted_color_loss

                # for imap*, it uses volume density
                # discourage any geometry from the camera center to 0.85*depth.
                regulation = (not self.occupancy)
                if regulation:
                    point_sigma = self.renderer.regulation(
                        c, self.decoders, batch_rays_d, batch_rays_o, batch_gt_depth, device, self.stage)
                    regulation_loss = torch.abs(point_sigma).sum()
                    loss += 0.0005*regulation_loss

            loss.backward(retain_graph=False)
            optimizer.step()
            if not self.nice:
                # for imap*
                scheduler.step()
            optimizer.zero_grad()

            # put selected and updated features back to the grid
            '''if self.nice and self.frustum_feature_selection:
                for key, val in c.items():
                    if (self.coarse_mapper and 'coarse' in key) or \
                            ((not self.coarse_mapper) and ('coarse' not in key)):
                        val_grad = masked_c_grad[key]
                        mask = masked_c_grad[key+'mask']
                        val = val.detach()
                        val[mask] = val_grad.clone().detach()
                        c[key] = val
            '''
        if self.BA:
            # put the updated camera poses back
            camera_tensor_id = 0
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    if frame != oldest_frame:
                        c2w = get_camera_from_tensor(
                            camera_tensor_list[camera_tensor_id].detach())
                        c2w = torch.cat([c2w, bottom], dim=0)
                        camera_tensor_id += 1
                        keyframe_dict[frame]['est_c2w'] = c2w.clone()
                else:
                    c2w = get_camera_from_tensor(
                        camera_tensor_list[-1].detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    cur_c2w = c2w.clone()
        if self.BA:
            return cur_c2w
        else:
            return None
    
    '''
    Mapper 是被动线程
    '''
    
    def run(self):
        # =========================== 第一帧从frame loader中提取 ============================ #
        cfg = self.cfg
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0] #这里的idx其实很快就被下面覆盖了，意义不大
        self.estimate_c2w_list[0] = gt_c2w.cpu() # estimate_c2w_list在主线程NICE_SLAM.py被初始化

        # 一开始先初始化第一帧的keyframe，'depth': gt_depth.cpu()，在线程收到Tracker的数据后要修改成咱们的pseudo depth
        self.keyframe_list.append(idx)
        self.keyframe_dict.append({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(
                    ), 'depth': gt_depth.cpu(), 'est_c2w': gt_c2w.clone()})
        
        # =================== 开始被动循环，在while True中判断是否需要这个循环 ================ #
        
        while (1):
            #rlock.acquire()
            # =============== 这个 while True:开关仅在满足条件时开启下边的优化 =============== #
            while True:
                idx = self.idx[0] #.clone()
                sleep_s = 5
                # 当：idx达到倒数第二帧 ；达到关键帧(比如设定了self.every_frame=5)；idx和prev_idx不相等
                # 跳出循环，直接将当前的idx赋值给prev_idx
                if idx == self.n_img-1:
                    break
                if self.sync_method == 'loose':
                    idx_every=torch.tensor((self.every_frame)).int()
                    '''
                    下列条件满足任何一个
                    1. 现在的.estimate_c2w_list[idx_every] 没有0
                    2. self.depth_cur_weak 没有0
                    3. self.depth_ref_weak 没有0
                    4. 是非0 every frame
                    才 break 掉 while True 循环，开始训练
                    '''
                    
                    if torch.count_nonzero(self.estimate_c2w_list[idx_every]).item()!=0 and idx % self.every_frame == 0 and idx != 0: # and torch.count_nonzero(self.depth_cur_weak).item()!=0 and torch.count_nonzero(self.depth_ref_weak)!=0 and  
                        if idx == self.every_frame:
                            try:
                                pseudo_depth_paths = sorted(glob.glob(f'{self.output}/pseudo_depth/*'))
                                
                                dpath_0=pseudo_depth_paths[0]
                                dpath_idx=pseudo_depth_paths[1]
                                
                                depth_0=np.load(dpath_0)
                                depth_every=np.load(dpath_idx)
                                
                                self.depth_0 = torch.from_numpy(depth_0)
                                self.depth_every = torch.from_numpy(depth_every)
                                Printer.green(f"===># Message from Mapper: init two .npy loaded! ")
                                break
                            except:
                                Printer.yellow(f"===># Message from Mapper: Warning! .npy loading fail for init")
                        if idx > self.every_frame:
                            try:
                                pseudo_depth_path = os.path.join(self.output, 'pseudo_depth',str(idx.numpy()),'.npy') 
                                depth_idx = np.load(pseudo_depth_path)
                                self.depth_idx = torch.from_numpy(depth_idx)
                                break               
                            except:
                                Printer.yellow(f"===># Message from Mapper: Warning! .npy loading fail for {idx} frame!")        
                        #Printer.green(f'This mapper receive self.estimate_c2w_list[idx_every] as 0, so sleep {sleep_s}s to wait Tracker')
                        #time.sleep(sleep_s) 
                    #if idx % self.every_frame == 0: #and idx != prev_idx:
                    #    break
                '''
                if (idx % self.every_frame == 1 ): # or self.every_frame == 1
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.5) 
                
                elif self.sync_method == 'loose':
                    if idx == 0 or idx >= prev_idx+self.every_frame//2:
                        break
                elif self.sync_method == 'free':
                    break
                '''
            time.sleep(0.1)
            #prev_idx = idx
            Printer.green(f"===># Message from Mapper: suffice criteria, Wake Up! begin a map optimization in {idx} Frame")

            '''if self.verbose:
                #print(Fore.GREEN)
                #prefix = 'Coarse ' if self.coarse_mapper else ''
                print("Mapping Frame ", idx.item())
                #print(Style.RESET_ALL)'''

            _, gt_color, gt_depth, gt_c2w = self.frame_reader[idx]

            # ================================= 设置优化参数 ================================== #
            # init_finished = True
            if not self.init_finished:
                    outer_joint_iters = 1
                    lr_factor = cfg['mapping']['lr_first_factor'] #5
                    num_joint_iters = 1500#cfg['mapping']['iters_first'] #1500
            
            else:
                if idx !=0 or idx !=self.track_every_frame:
                    lr_factor = cfg['mapping']['lr_factor'] # 1
                    num_joint_iters = cfg['mapping']['iters'] # 300

                    # here provides a color refinement postprocess
                    if idx == self.n_img-1 and self.color_refine and not self.coarse_mapper:
                        outer_joint_iters = 5
                        self.mapping_window_size *= 2
                        self.middle_iter_ratio = 0.0
                        self.fine_iter_ratio = 0.0
                        num_joint_iters *= 5
                        self.fix_color = True
                        self.frustum_feature_selection = False
                    else:
                        if self.nice:
                            outer_joint_iters = 1
                        else:
                            outer_joint_iters = 3
            # ================================================================================ #
             
            num_joint_iters = num_joint_iters//outer_joint_iters
            
            # outer_joint_iters:
            #   初始化: 1
            #   倒数第二帧：5
            #   其他帧：3
            self.mapper_finished = False            
            for outer_joint_iter in tqdm(range(outer_joint_iters)):
                # 初始化
                if not self.init_finished: #idx == self.track_every_frame and
                    #idx0=torch.zeros((1)).int()[0]
                    while len(self.estimate_c2w_list[idx])==0 or len(self.estimate_c2w_list[self.idx0])==0:
                        Printer.green(f'Mapper waits self.estimate_c2w_list to fill')
                        time.sleep(sleep_s) 
                    # 在run的一开始已经初始化了第一帧的list和dict，这里把收到的值替换一下    
                    self.keyframe_dict[self.idx0]['pseudo_depth'] = self.depth_0.clone()
                
                    cur_c2w = self.estimate_c2w_list[idx].to(self.device) 
                    self.keyframe_list.append(idx)
                    self.keyframe_dict.append({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(
                    ), 'depth': gt_depth.cpu(), 'est_c2w': cur_c2w.clone(),'pseudo_depth': self.depth_every.clone().to(self.device)})
                
                if idx > self.track_every_frame and self.init_finished:
                    while len(self.estimate_c2w_list[idx])==0:
                            Printer.green(f'Mapper waits self.estimate_c2w_list to fill')
                            time.sleep(sleep_s)
                    cur_c2w = self.estimate_c2w_list[idx].to(self.device) 
                    self.keyframe_list.append(idx)
                    self.keyframe_dict.append({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(
                    ), 'depth': gt_depth.cpu(), 'est_c2w': cur_c2w.clone(),'pseudo_depth':self.depth_idx.clone()})
                    
                '''else:
                    # add new frame to keyframe set
                    # 到准备get out iteration的时候，查询当前帧是否满足关键帧定义并加入
                    if outer_joint_iter == outer_joint_iters-1 and init_ok:
                        if (idx % self.keyframe_every == 0 or (idx == self.n_img-2)) \
                            and (idx not in self.keyframe_list):
                                
                            self.keyframe_list.append(idx)
                            self.keyframe_dict.append({'gt_c2w': gt_c2w.cpu(), 'idx': idx, 'color': gt_color.cpu(
                            ), 'depth': gt_depth.cpu(), 'est_c2w': cur_c2w.clone()})
                    if idx==0:
                        break'''
                    
                #imap self.BA是设为False，而nice-slam是True
                self.BA = (len(self.keyframe_list) > 4) and cfg['mapping']['BA'] and (
                    not self.coarse_mapper) 
                
                # optimize_map返回 cur_c2w/None (tensor/None)，如果没BA就啥都不返回 
                # imap 就是关闭了BA，所以不返回更新的pose，只更新mapping
                # then optimize scene representation and camera poses(if local BA enabled).
                if self.weak_depth and not self.init_finished: #& (idx == self.track_every_frame).sum() and not self.depth_cur_weak.sum():
                    #gt_depth=self.depth_cur_weak
                    print('============= sanity check self.keyframe_dict for idx 0,5 ===========:::',
                          self.keyframe_dict[0],
                          self.keyframe_dict[-1])
                    _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth,
                                      gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w=cur_c2w)
                    
                    self.init_finished = True
                    break 
                                
                if self.weak_depth & idx>self.track_every_frame:
                    #gt_depth=self.depth_cur_weak
                    _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth,
                                      gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w=cur_c2w)

                else:    
                    _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth,
                                      gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w=cur_c2w)
                
                if self.BA:
                    cur_c2w = _
                    self.estimate_c2w_list[idx] = cur_c2w
            

            if self.low_gpu_mem:
                torch.cuda.empty_cache()
            
            # 设置了一个通讯开关
            self.mapper_finished = True  
            
            # mapping of first frame is done, can begin tracking（在NICE_SLAM.py 的主线程中tracking就是在等他）
            self.mapping_first_frame[0] = 1
            
            # mapping 构造函数在一开始就把coarse写False了
            # ====================================== 输出mesh的逻辑 ======================================== #
            Printer.yellow(f'===> Message from Mapper: The mesh is creating, curent idx is {idx}')
            if not self.coarse_mapper:
                if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) \
                        or idx == self.n_img-1:
                    self.logger.log(idx, self.keyframe_dict, self.keyframe_list,
                                    selected_keyframes=self.selected_keyframes
                                    if self.save_selected_keyframes_info else None)

                self.mapping_idx[0] = idx
                self.mapping_cnt[0] += 1

                if (idx % self.mesh_freq == 0) and (not (idx == 0)): #and self.no_mesh_on_first_frame)):
                    mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx,  self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)

                if idx == self.n_img-1:
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx,  self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)
                    os.system(
                        f"cp {mesh_out_file} {self.output}/mesh/{idx:05d}_mesh.ply")
                    if self.eval_rec:
                        mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                        self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict,
                                             self.estimate_c2w_list, idx, self.device, show_forecast=False,
                                             clean_mesh=self.clean_mesh, get_mask_use_all_frames=True)
                    break
            # ============================================================================================= #
            if idx == self.n_img-1:
                break
            #rlock.release()