import os
import numpy as np
import cv2
import torch
import open3d as o3d
import argparse
from natsort import natsorted
import glob
from tqdm import tqdm
import json
import pickle

from ..models.clip_encoder import CLIPTextEncoder, LLaVATextGenerator


class ScanNet_scene:
    def __init__(self, scene_id, args):
        self.base_dir = args.base_dir
        self.scene_id = scene_id
        self.device = args.device
        self.points = self._load_scene_point_cloud(scene_id)
        self.N = self.points.shape[0]
        self.step_size = args.step_size
        self.vis_dis = args.vis_dis

        self.poses, self.depths, self.color_intrinsics, self.depth_intrinsics, self.masks, self.color_images = \
            self.init_data(self.base_dir, scene_id)
        self.M = self.masks.shape[0]
        self.CH, self.CW = self.masks.shape[-2:]
        self.DH, self.DW = self.depths.shape[-2:]

        # Initialize CLIP encoder and text generator
        self.clip_encoder = CLIPTextEncoder(device=self.device, model_name=args.clip_model, cache_dir="/media/ssd/jiangxirui/projects/2/pretrained/clip")
        self.llava_generator = LLaVATextGenerator(cache_dir="/media/ssd/jiangxirui/projects/2/pretrained/llava", device_map=self.device)

    def _load_scene_point_cloud(self, scene_id: str) -> torch.Tensor:
        """Load complete scene point cloud"""

        # Try to load cached scene point cloud
        scene_pc_path = os.path.join(self.base_dir, scene_id, "processed", "scene_point_cloud.pth")
        if os.path.exists(scene_pc_path):
            return torch.load(scene_pc_path).to(self.device)
        
        # Otherwise, aggregate from all frames (simplified)
        ply_path = os.path.join(self.base_dir, scene_id, f"{scene_id}_vh_clean_2.ply")

        pcd = o3d.io.read_point_cloud(ply_path)
        # Extract coordinates
        coords = np.asarray(pcd.points)  # (N, 3)
        # Extract colors
        colors = np.asarray(pcd.colors)  # (N, 3), normalized [0, 1]
        # Combine into Nx6 tensor
        scene_pc = torch.tensor(np.concatenate([coords, colors], axis=1), device=self.device, dtype=torch.float32)

        # Cache for future use
        os.makedirs(os.path.dirname(scene_pc_path), exist_ok=True)
        torch.save(scene_pc, scene_pc_path)
        return scene_pc

    def init_data(self, base_dir, scene_id):
        data_dir = os.path.join(base_dir, scene_id)
        color_list = natsorted(glob.glob(os.path.join(data_dir, '*.jpg')))

        poses = []
        depths = []
        color_intrinsics = []
        depth_intrinsics = []
        masks = []
        color_images = []

        for color_path in tqdm(color_list, desc='Read 2D data'):
            color_name = os.path.basename(color_path)
            num = int(color_name[:-4])
            if num % self.step_size != 0:
                continue
            poses.append(np.loadtxt(color_path.replace('.jpg', '.txt')).astype(np.float32))

            depth = cv2.imread(color_path.replace('.jpg', '.png'), -1).astype(np.float32) / 1000.
            depths.append(depth)

            color_intrinsic_path = os.path.join(os.path.dirname(color_path), 'intrinsic_color.txt')
            depth_intrinsic_path = os.path.join(os.path.dirname(color_path), 'intrinsic_depth.txt')
            color_intrinsic = np.loadtxt(color_intrinsic_path).astype(np.float32)[:3, :3]
            depth_intrinsic = np.loadtxt(depth_intrinsic_path).astype(np.float32)[:3, :3]
            color_intrinsics.append(color_intrinsic)
            depth_intrinsics.append(depth_intrinsic)

            mask = cv2.imread(os.path.join(data_dir, "2D_instance_masks", color_name.replace('.jpg', '.png')), -1)
            masks.append(mask)

            # Load color image for text generation
            color_img = cv2.imread(color_path)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            color_images.append(color_img)

        poses = np.stack(poses, 0)  # (M, 4, 4)
        depths = np.stack(depths, 0) # (M, H, W)
        color_intrinsics = np.stack(color_intrinsics, 0)  # (M, 3, 3)
        depth_intrinsics = np.stack(depth_intrinsics, 0)  # (M, 3, 3)
        masks = np.stack(masks, 0)  # (M, H, W)
        color_images = np.stack(color_images, 0)  # (M, H, W, 3)

        return poses, depths, color_intrinsics, depth_intrinsics, masks, color_images

    def prepocess(self):
        """assign instance labels and visibility for all points in the scene 

        :return points_label: (N, M), resulting instance labels of all points
        :return points_seen: (N, M), seen flag of all points in all views
        """

        # project N points to M images
        pts_cam, color_pixes, depth_pixes = self.torch_world2cam_pixel(
            self.points[:, :3],  # (N, 3)
            self.color_intrinsics, 
            self.depth_intrinsics,
            self.poses)  # (N, M, 3), (N, M, 2)

        # (N, M)通过投影确定每个3D点在每个视图上是否可见
        points_label, points_seen = self.get_points_label_and_seen(pts_cam, color_pixes, depth_pixes, vis_dis=self.vis_dis)
        np.save(os.path.join(self.base_dir, self.scene_id, "processed", 'points_label.npy'), points_label)
        np.save(os.path.join(self.base_dir, self.scene_id, "processed", 'points_seen.npy'), points_seen)

        self.generate_and_encode_descriptions()

    @torch.inference_mode()
    def torch_world2cam_pixel(self, points_world_all: torch.tensor, color_intrinsic: np.array, depth_intrinsic: np.array, pose: np.array):
        """project N points to M images

        :param points_world: (N, 3) the points coordinates in the world axis
        :param color_intrinsic, depth_intrinsic: (M, 3, 3) the intrinsics of color and depth camera
        :param pose: (M, 4, 4)
        :return: points_cam(N,M,3), color_points_pixel(N,M,2), depth_points_pixel(N,M,2)
        """

        batch_size = 10000

        color_intrinsic = torch.tensor(color_intrinsic, device=self.device, dtype=torch.float32)
        depth_intrinsic = torch.tensor(depth_intrinsic, device=self.device, dtype=torch.float32)
        pose = torch.tensor(pose, device=self.device, dtype=torch.float32)
        pose_inv = torch.linalg.inv(pose)   # (M, 4, 4)
        del pose

        N = self.N
        M = self.M

        final_points_cam = np.zeros((N, M, 3), dtype=np.float32)
        final_color_points_pixel = np.zeros((N, M, 2), dtype=int)
        final_depth_points_pixel = np.zeros((N, M, 2), dtype=int)

        for batch_start in range(0, points_world_all.shape[0], batch_size):
            points_world = points_world_all[batch_start: batch_start+batch_size]
            points_world_homo = torch.cat((points_world, torch.ones((points_world.shape[0], 1), dtype=torch.float32, device=self.device)), 1)

            points_cam_homo = torch.matmul(pose_inv[None], points_world_homo[:, None, :, None])
            points_cam_homo = points_cam_homo[..., 0]        # (N, M, 4)

            points_cam = torch.div(points_cam_homo[..., :-1], points_cam_homo[..., [-1]])  # (N, M, 3)

            # (M, 3, 3) @ (N, M, 3, 1) = (N, M, 3, 1)
            color_points_pixel_homo = torch.matmul(color_intrinsic, points_cam[..., None])
            depth_points_pixel_homo = torch.matmul(depth_intrinsic, points_cam[..., None])

            # (N, M, 3)
            color_points_pixel_homo = color_points_pixel_homo[..., 0]
            depth_points_pixel_homo = depth_points_pixel_homo[..., 0]

            color_points_pixel = \
                torch.div(color_points_pixel_homo[..., :-1], torch.clip(color_points_pixel_homo[..., [-1]], min=1e-8)).round().to(torch.int32)  # (u, v) coordinate, (N, M, 2)
            depth_points_pixel = \
                torch.div(depth_points_pixel_homo[..., :-1], torch.clip(depth_points_pixel_homo[..., [-1]], min=1e-8)).round().to(torch.int32)

            final_points_cam[batch_start: batch_start + batch_size] = points_cam.cpu().numpy()
            final_color_points_pixel[batch_start: batch_start + batch_size] = color_points_pixel.cpu().numpy()
            final_depth_points_pixel[batch_start: batch_start + batch_size] = depth_points_pixel.cpu().numpy()

        torch.cuda.empty_cache()
        return final_points_cam, final_color_points_pixel, final_depth_points_pixel

    def get_points_label_and_seen(self, pts_cam, color_pixes, depth_pixes,  vis_dis):
        """get label and seen flag of all points in all views

        :param pts_cam: (N, M, 3), transformed to camera coordinate
        :param color_pixes, depth_pixes: (N, M, 2), projected pixel locations to color and depth images
        :param vis_dis: the distance threshold for judging whether a point is visible
        
        :return all_label: (N, M), labels of all points in all views
        :return all_seen_flag: (N, M), seen flag of all points in all views
        """

        batch_size = 50000
        all_label, all_seen_flag = np.zeros([self.N, self.M], dtype=int), np.zeros([self.N, self.M], dtype=bool)

        for start_id in range(0, self.N, batch_size):
            p_cam0 = pts_cam[start_id: start_id + batch_size]
            color_pix0 = color_pixes[start_id: start_id + batch_size]
            depth_pix0 = depth_pixes[start_id: start_id + batch_size]

            cw0, ch0 = np.split(color_pix0, 2, axis=-1)
            cw0, ch0 = cw0[..., 0], ch0[..., 0]  # (N, M)
            bounded_flag0 = (0 <= cw0)*(cw0 <= self.CW - 1)*(0 <= ch0)*(ch0 <= self.CH - 1)  # (N, M)

            # (N, M), querying labels from masks (M, H, W) by h (N, M) and w (N, M)
            label0 = self.masks[np.arange(self.M), 
                                ch0.clip(0, self.CH - 1), 
                                cw0.clip(0, self.CW - 1)]

            dw0, dh0 = np.split(depth_pix0, 2, axis=-1)
            dw0, dh0 = dw0[..., 0], dh0[..., 0]  # (N, M)

            # judge whether the point is visible
            real_depth0 = p_cam0[..., -1]  # (N, M)
            capture_depth0 = self.depths[np.arange(self.M), 
                                         dh0.clip(0, self.DH - 1), 
                                         dw0.clip(0, self.DW - 1)]  # (N, M), querying depths
            visible_flag0 = np.isclose(real_depth0, capture_depth0, rtol=vis_dis)

            seen_flag = bounded_flag0 * visible_flag0

            label0 = label0 * seen_flag  # set label of invalid point to 0

            all_seen_flag[start_id: start_id + batch_size] = seen_flag
            all_label[start_id: start_id + batch_size] = label0

        return all_label, all_seen_flag

    def generate_and_encode_descriptions(self):
        """Generate text descriptions for all instances and encode them with CLIP"""
        
        # Create directories for saving
        text_desc_dir = os.path.join(self.base_dir, self.scene_id, "processed", 'text_descriptions')
        clip_feat_dir = os.path.join(self.base_dir, self.scene_id, "processed", 'clip_features')
        os.makedirs(text_desc_dir, exist_ok=True)
        os.makedirs(clip_feat_dir, exist_ok=True)
        
        # Process each frame
        for frame_idx in tqdm(range(self.M), desc='Generating text descriptions'):
            # Get image and mask for this frame
            image = self.color_images[frame_idx]  # (H, W, 3)
            mask = self.masks[frame_idx]  # (H, W)
            
            # Convert to torch tensor for LLaVA generator
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.to(self.device)
            mask_tensor = torch.from_numpy(mask).to(self.device)
            
            # Generate descriptions for all instances in this frame
            descriptions = self.llava_generator.generate_descriptions(image_tensor, mask_tensor)
            
            # Encode descriptions with CLIP
            clip_features = {}
            if descriptions:
                # Get list of descriptions in order of instance IDs
                instance_ids = sorted(descriptions.keys())
                desc_list = [descriptions[inst_id] for inst_id in instance_ids]
                
                # Encode all descriptions at once
                text_features = self.clip_encoder.encode_text(desc_list)  # [num_instances, feature_dim]
                
                # Store features by instance ID
                for idx, inst_id in enumerate(instance_ids):
                    clip_features[inst_id] = text_features[idx].cpu().numpy()
            
            # Save descriptions and features for this frame
            frame_desc_path = os.path.join(text_desc_dir, f'frame_{frame_idx * self.step_size}_desc.json')
            frame_feat_path = os.path.join(clip_feat_dir, f'frame_{frame_idx * self.step_size}_feat.pkl')
            
            # Save text descriptions as JSON
            with open(frame_desc_path, 'w') as f:
                json.dump(descriptions, f, indent=2)
            
            # Save CLIP features as pickle
            with open(frame_feat_path, 'wb') as f:
                pickle.dump(clip_features, f)
        
        # Save metadata
        metadata = {
            'num_frames': self.M,
            'clip_model': self.clip_encoder.model_name if hasattr(self.clip_encoder, 'model_name') else 'ViT-B/32',
            'feature_dim': self.clip_encoder.feature_dim,
            'frame_indices': list(range(self.M) * self.step_size)
        }
        
        metadata_path = os.path.join(self.base_dir, self.scene_id, 'clip_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Generated and encoded descriptions for {self.M} frames in scene {self.scene_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScanNet Scene Preprocessing")
    parser.add_argument('--base_dir', type=str, default="/media/ssd/jiangxirui/projects/2/data/ScanNetV2", help='Base directory of ScanNet data')
    parser.add_argument('--step_size', type=int, default=25, help='Step size for frame selection')
    parser.add_argument('--vis_dis', type=float, default=0.15, help='Visibility distance threshold')
    parser.add_argument('--device', type=str, default='cuda:5', help='Device to use for processing')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='CLIP model to use for encoding')

    args = parser.parse_args()

    train_split = "/media/ssd/jiangxirui/projects/2/data/ScanNetV2/meta/scannetv2_train.txt"
    
    with open(train_split, 'r') as f:
        scenes = [line.strip() for line in f.readlines()]

    for scene_id in tqdm(sorted(scenes)):
        print(scene_id)
        scene_processor = ScanNet_scene(scene_id, args)
        scene_processor.prepocess()