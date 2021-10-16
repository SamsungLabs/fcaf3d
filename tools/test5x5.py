import os
import torch
import argparse
import numpy as np
from collections import defaultdict
from mmcv import Config
from mmcv.runner import load_checkpoint, init_dist, get_dist_info
from mmcv.parallel import MMDistributedDataParallel
from mmdet.apis import set_random_seed, multi_gpu_test
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a 5 models 5 times')
    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', help='checkpoints directory')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_dist(args.launcher, **cfg.dist_params)
    checkpoints = tuple(filter(lambda x: x.endswith('.pth'), os.listdir(args.checkpoint)))
    print('found checkpoints: ', checkpoints)
    metrics = defaultdict(list)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    for i, checkpoint in enumerate(checkpoints):
        load_checkpoint(model, os.path.join(args.checkpoint, checkpoint), map_location='cpu')
        for j in range(5):
            set_random_seed(j * 111)
            dataset = build_dataset(cfg.data.test)
            data_loader = build_dataloader(
                dataset,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=True,
                shuffle=False)
            outputs = multi_gpu_test(model, data_loader)
            if get_dist_info()[0] == 0:
                for k, v in dataset.evaluate(outputs).items():
                    metrics[k].append(v)

    if get_dist_info()[0] == 0:
        for k in ('mAP_0.25', 'mAP_0.50'):
            print(k, 'min, mean, max:', np.min(metrics[k]), np.mean(metrics[k]), np.max(metrics[k]))


if __name__ == '__main__':
    main()
