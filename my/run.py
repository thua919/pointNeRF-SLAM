
import argparse
import random

import numpy as np
import torch

from src import config
from src.pointNeRF_SLAM import pointNeRF_SLAM

# 限制0号设备的显存的使用量为0.5，就是半张卡那么多，比如12G卡，设置0.5就是6G。
#torch.cuda.set_per_process_memory_fraction(0.99, 5)
#torch.cuda.empty_cache()

def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the pointNeRF-SLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    #我们用这中add_argument('--nice', dest='nice', action='store_true')来做bundle loss 和 sampled ray的 ablation
    '''
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    #add_argument将 --nice 和 --imap 命令都解析为参数 nice，但action上 --imap 将 nice设置为了False
    #所以，在类中初始化参数的时候，arg.nice=False
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    #parser.set_defaults(nice=True)
    '''
    
    args = parser.parse_args()

    cfg = config.load_config(
        args.config,'pointNeRF-SLAM/configs/point_slam/point_slam.yaml')

    slam = pointNeRF_SLAM(cfg, args)

    slam.run()


if __name__ == '__main__':
    main()
