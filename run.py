import argparse
import random

import numpy as np
import torch

from src import config
from src.NICE_SLAM import NICE_SLAM
import torch

# 限制0号设备的显存的使用量为0.5，就是半张卡那么多，比如12G卡，设置0.5就是6G。
#torch.cuda.set_per_process_memory_fraction(0.99, 5)
torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    '''
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    '''
    
    parser.set_defaults(nice=False)
    
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/pointNeRF_slam.yaml')

    slam = NICE_SLAM(cfg, args)

    slam.run()


if __name__ == '__main__':
    main()
