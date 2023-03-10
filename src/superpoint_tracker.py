"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np 
from enum import Enum
import cv2

from src.superpoint_manager import SuperpointManager
from src.feature_superpoint import SuperPointFeature2D 
from src.superpoint_matcher import feature_matcher_factory, FeatureMatcherTypes
from src.utils.utils_features import sat_num_features, kdt_nms, ssc_nms, octree_nms, grid_nms

#from utils_sys import Printer, import_from
#from utils_geom import hamming_distance, hamming_distances, l2_distance, l2_distances
#from parameters import Parameters 
#from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo


kMinNumFeatureDefault = 2000
kLkPyrOpticFlowNumLevelsMin = 3   # maximal pyramid level number for LK optic flow 
kRatioTest = 0.7
feature_detector = SuperPointFeature2D()

class FeatureTrackerTypes():
    LK        = 0   # Lucas Kanade pyramid optic flow (use pixel patch as "descriptor" and matching by optimization)
    DES_BF    = 1   # descriptor-based, brute force matching with knn 
    DES_FLANN = 2   # descriptor-based, FLANN-based matching 

class FeatureDetectorTypes():   
    NONE        = 0 
    SUPERPOINT  = 1  # [end-to-end] joint detector-descriptor - "SuperPoint: Self-Supervised Interest Point Detection and Description"

class FeatureDescriptorTypes():
    NONE        = 0 
    SUPERPOINT  = 1  # [end-to-end] only with SUPERPOINT detector - "SuperPoint: Self-Supervised Interest Point Detection and Description"

class KeyPointFilterTypes(Enum):
    NONE         = 0
    SAT          = 1      # sat the number of features (keep the best N features: 'best' on the basis of the keypoint.response)
    KDT_NMS      = 2      # Non-Maxima Suppression based on kd-tree
    SSC_NMS      = 3      # Non-Maxima Suppression based on https://github.com/BAILOOL/ANMS-Codes
    OCTREE_NMS   = 4      # Distribute keypoints by using a octree (as a matter of fact, a quadtree): from ORBSLAM2
    GRID_NMS     = 5      # NMS by using a grid 
    
FeatureTrackerTypes=FeatureTrackerTypes()
FeatureDetectorTypes=FeatureDetectorTypes()
FeatureDescriptorTypes=FeatureDescriptorTypes()

class FeatureTrackingResult(object): 
    def __init__(self):
        self.kps_ref = None          # all reference keypoints (numpy array Nx2)
        self.kps_cur = None          # all current keypoints   (numpy array Nx2)
        self.des_cur = None          # all current descriptors (numpy array NxD)
        self.idxs_ref = None         # indexes of matches in kps_ref so that kps_ref_matched = kps_ref[idxs_ref]  (numpy array of indexes)
        self.idxs_cur = None         # indexes of matches in kps_cur so that kps_cur_matched = kps_cur[idxs_cur]  (numpy array of indexes)
        self.kps_ref_matched = None  # reference matched keypoints, kps_ref_matched = kps_ref[idxs_ref]
        self.kps_cur_matched = None  # current matched keypoints, kps_cur_matched = kps_cur[idxs_cur]

# Base class for a feature tracker.
# It mainly contains a feature manager and a feature matcher. 
class FeatureTracker(object): 
    def __init__(self, num_features=kMinNumFeatureDefault, 
                       num_levels = 1,                                   # number of pyramid levels for detector and descriptor  
                       scale_factor = 1.2,                               # detection scale factor (if it can be set, otherwise it is automatically computed) 
                       detector_type = FeatureDetectorTypes.SUPERPOINT, 
                       descriptor_type = FeatureDescriptorTypes.SUPERPOINT,
                       match_ratio_test = kRatioTest, 
                       tracker_type = FeatureTrackerTypes.DES_FLANN):
        self.detector_type = detector_type
        self.descriptor_type = descriptor_type
        self.tracker_type = tracker_type

        self.feature_manager = None      # it contains both detector and descriptor  
        self.matcher = None              # it contain descriptors matching methods based on BF, FLANN, etc.
                
    @property
    def num_features(self):
        return self.feature_manager.num_features
    @property
    def num_levels(self):
        return self.feature_manager.num_levels      
    @property
    def scale_factor(self):
        return self.feature_manager.scale_factor    
    @property
    def norm_type(self):
        return self.feature_manager.norm_type       
    @property
    def descriptor_distance(self):
        return self.feature_manager.descriptor_distance               
    @property
    def descriptor_distances(self):
        return self.feature_manager.descriptor_distances               
    # out: keypoints and descriptors 
    def detectAndCompute(self, frame, mask): 
        return None, None 
    # out: FeatureTrackingResult()
    def track(self, image_ref, image_cur, kps_ref, des_ref):
        return FeatureTrackingResult()             

# Extract features by using desired detector and descriptor, match keypoints by using desired matcher on computed descriptors
class SuperpointTracker(FeatureTracker): 
    def __init__(self, num_features=kMinNumFeatureDefault, 
                       num_levels = 1,                                    # number of pyramid levels for detector 
                       scale_factor = 1.2,                                # detection scale factor (if it can be set, otherwise it is automatically computed)                
                       detector_type = FeatureDetectorTypes.SUPERPOINT, 
                       descriptor_type = FeatureDescriptorTypes.SUPERPOINT,
                       match_ratio_test = kRatioTest, 
                       tracker_type = FeatureTrackerTypes.DES_FLANN):
        super().__init__(num_features=num_features, 
                         num_levels=num_levels, 
                         scale_factor=scale_factor, 
                         detector_type=detector_type, 
                         descriptor_type=descriptor_type, 
                         match_ratio_test = match_ratio_test,
                         tracker_type=tracker_type)
        self.feature_manager = SuperpointManager(num_features=num_features, 
                                                       num_levels=num_levels, 
                                                       scale_factor=scale_factor, 
                                                       detector_type=detector_type, 
                                                       descriptor_type=descriptor_type)                     
        self.keypoint_filter_type = KeyPointFilterTypes.SAT            # default keypoint-filter type
        self.keypoint_nms_filter_type = KeyPointFilterTypes.KDT_NMS    # default keypoint-filter type if NMS is needed 
    
        if tracker_type == FeatureTrackerTypes.DES_FLANN: # ?????????????????????????????????
            self.matching_algo = FeatureMatcherTypes.FLANN
        elif tracker_type == FeatureTrackerTypes.DES_BF:
            self.matching_algo = FeatureMatcherTypes.BF
        else:
            raise ValueError("Unmanaged matching algo for feature tracker %s" % self.tracker_type)                   
                    
        # init matcher ?????????FlannFeatureMatcher
        self.matcher = feature_matcher_factory(norm_type=self.norm_type, # cv2.NORM_L2()
                                               ratio_test=match_ratio_test, # 0.7
                                               type=self.matching_algo)   # FeatureMatcherTypes.FLANN    
    
    def filter_keypoints(self, type, frame, kps, des=None):
        filter_name = type.name
        if type == KeyPointFilterTypes.SAT: # default keypoint-filter type                                                        
            if len(kps) > self.num_features:
                kps, des = sat_num_features(kps, des, self.num_features)
                 
        elif type == KeyPointFilterTypes.NONE:
            pass  
        elif type == KeyPointFilterTypes.KDT_NMS:      
            kps, des = kdt_nms(kps, des, self.num_features)
        elif type == KeyPointFilterTypes.SSC_NMS:    
            kps, des = ssc_nms(kps, des, frame.shape[1], frame.shape[0], self.num_features)   
        elif type == KeyPointFilterTypes.OCTREE_NMS:
            if des is not None: 
                raise ValueError('at the present time, you cannot use OCTREE_NMS with descriptors')
            kps = octree_nms(frame, kps, self.num_features)
        elif type == KeyPointFilterTypes.GRID_NMS:    
            kps, des, _ = grid_nms(kps, des, frame.shape[0], frame.shape[1], self.num_features, dist_thresh=4)            
            kps, des, _ = grid_nms(kps, des, frame.shape[0], frame.shape[1], self.num_features, dist_thresh=4)            
        elif type == KeyPointFilterTypes.SAT:                                                        
            if len(kps) > self.num_features:
                kps, des = sat_num_features(kps, des, self.num_features)      
            kps, des, _ = grid_nms(kps, des, frame.shape[0], frame.shape[1], self.num_features, dist_thresh=4)               
        elif type == KeyPointFilterTypes.SAT:                                                        
            if len(kps) > self.num_features:
                kps, des = sat_num_features(kps, des, self.num_features)      
        else:             
            raise ValueError("Unknown match-filter type")
                 
        return kps, des, filter_name 


    # out: keypoints and descriptors ?????????feature_superpoint ?????????detectAndComput
    def detectAndCompute(self, frame, mask=None, filter = True):
        if frame.ndim>2:     # check if we have to convert to gray image 
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        # ================= ?????????  feature_superpoint.py?????? detectAndCompute ============================= #
        kps, des = feature_detector.detectAndCompute(frame,mask)
        # ================================================================================================= # 
        print('detector = descriptor?:', self.detector_type==self.descriptor_type,', #features detected:',len(kps))           
        filter_name = 'NONE'
        
        if filter:                                                                 
            kps, des, filter_name  = self.filter_keypoints(self.keypoint_filter_type, frame, kps, des)
            kps, des, filter_name  = self.filter_keypoints(self.keypoint_filter_type, frame, kps, des)           
            kps, des, filter_name  = self.filter_keypoints(self.keypoint_filter_type, frame, kps, des)             

        print('filter_name is: ',filter_name,
            ', #features detected after keypoint filter:', 
                  len(kps),' (#original expected keypoints', self.num_features, '), [keypoint filter type:',filter_name,']')                                         
        return kps, des 
        
    
    #def detectAndCompute(self, frame): 
    #    return self.feature_manager.detectAndCompute(frame) 

    '''
    # out: FeatureTrackingResult()
    def track(self, image_ref, image_cur, kps_ref, des_ref):
        # ============================================================= #
        # feature_superpoint ?????????detectAndComput,??????????????????heatmap?????????????????????
        kps_cur, des_cur= self.detectAndCompute(image_cur)
        # ============================================================= #
        
        # convert from list of keypoints to an array of points 
        kps_cur = np.array([x.pt for x in kps_cur], dtype=np.float32) 
        
        # ============================================================= #
        # ??????superpoint_matcher.py ?????? cv2.FlannBasedMatcher
        idxs_ref, idxs_cur = self.matcher.match(des_ref, des_cur)
        # ============================================================= #

        res = FeatureTrackingResult()

        
        res.kps_ref = kps_ref  # all the reference keypoints  
        res.kps_cur = kps_cur  # all the current keypoints       
        res.des_cur = des_cur  # all the current descriptors         
        
        res.kps_ref_matched = np.asarray(kps_ref[idxs_ref]) # the matched ref kps  
        res.idxs_ref = np.asarray(idxs_ref)                  
        
        res.kps_cur_matched = np.asarray(kps_cur[idxs_cur]) # the matched cur kps  
        res.idxs_cur = np.asarray(idxs_cur)
        
        return res
    '''