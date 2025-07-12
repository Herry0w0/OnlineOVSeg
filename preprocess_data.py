import os
import argparse
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from src.datasets.scannet_preprocess import ScanNet_scene

def parse_args():
    parser = argparse.ArgumentParser(description="ScanNet Scene Preprocessing")
    parser.add_argument('--base_dir', type=str, default="/media/ssd/jiangxirui/projects/2/data/ScanNetV2", help='Base directory of ScanNet data')
    parser.add_argument('--scene_list', type=str, default="1.txt", help='scenes to process')
    parser.add_argument('--step_size', type=int, default=30, help='Step size for frame selection')
    parser.add_argument('--vis_dis', type=float, default=0.15, help='Visibility distance threshold')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for processing')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='CLIP model to use for encoding')

    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')
    parser.add_argument('--multiprocess', action='store_true', help='Enable multiprocessing')

    return parser.parse_args()

def process_scenes_on_gpu(gpu_id, scene_list, args, worker_id):
    """
    处理分配给特定GPU的场景列表
    
    Args:
        gpu_id: 实际的GPU ID（如0,2,3,5中的一个）
        scene_list: 要处理的场景列表
        args: 命令行参数
        worker_id: 工作进程的索引（0,1,2,3），用于tqdm显示
    """
    # 设置当前进程使用指定的GPU
    args.device = f'cuda:{str(gpu_id)}'
    
    # 使用worker_id作为tqdm的position，gpu_id作为描述
    for scene_id in tqdm(sorted(scene_list), 
                        desc=f"GPU {gpu_id}", 
                        position=worker_id):
        try:
            scene_processor = ScanNet_scene(scene_id, args)
            scene_processor.prepocess()
        except Exception as e:
            print(f"GPU {gpu_id}: Error processing {scene_id}: {str(e)}")
            continue

def main():
    args = parse_args()
    
    # 解析GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)
    
    # 读取所有场景
    train_split = os.path.join(args.base_dir, "meta", args.scene_list)
    with open(train_split, 'r') as f:
        all_scenes = [line.strip() for line in f.readlines()]
    
    if args.multiprocess:
        # 多进程模式
        mp.set_start_method('spawn', force=True)

        # 将场景列表分成num_gpus份
        scenes_per_gpu = np.array_split(all_scenes, num_gpus)
        
        # 创建进程池
        processes = []
        
        # 为每个指定的GPU启动一个进程
        for worker_id, gpu_id in enumerate(gpu_ids):
            # 获取该GPU要处理的场景
            gpu_scenes = scenes_per_gpu[worker_id].tolist()
            
            print(f"Assigning {len(gpu_scenes)} scenes to GPU {gpu_id}")
            
            # 创建进程
            p = mp.Process(
                target=process_scenes_on_gpu,
                args=(gpu_id, gpu_scenes, args, worker_id)
            )
            p.start()
            processes.append(p)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
            
        print("All GPUs finished processing!")
    else:
        # 单进程模式 - 使用第一个指定的GPU
        args.device = f'cuda:{str(gpu_ids[0])}'
        process_scenes_on_gpu(gpu_ids[0], all_scenes, args, 0)

if __name__ == '__main__':
    exit(main())