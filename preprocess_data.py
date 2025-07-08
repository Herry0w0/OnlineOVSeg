import argparse
from src.datasets.scannet_preprocess import ScanNet_scene
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="ScanNet Scene Preprocessing")
    parser.add_argument('--base_dir', type=str, default="/media/ssd/jiangxirui/projects/2/data/ScanNetV2", help='Base directory of ScanNet data')
    parser.add_argument('--step_size', type=int, default=25, help='Step size for frame selection')
    parser.add_argument('--vis_dis', type=float, default=0.15, help='Visibility distance threshold')
    parser.add_argument('--device', type=str, default='cuda:5', help='Device to use for processing')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='CLIP model to use for encoding')
    return parser.parse_args()

def main():
    args = parse_args()

    train_split = "/media/ssd/jiangxirui/projects/2/data/ScanNetV2/meta/scannetv2_train_mini.txt"

    with open(train_split, 'r') as f:
        scenes = [line.strip() for line in f.readlines()]

    for scene_id in tqdm(sorted(scenes)):
        print(scene_id)
        scene_processor = ScanNet_scene(scene_id, args)
        scene_processor.prepocess()

if __name__ == '__main__':
    exit(main())