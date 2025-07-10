import os
from typing import List, Dict, Optional, Union
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
import clip
from transformers import Gemma3ForConditionalGeneration, AutoProcessor


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder for semantic feature extraction"""
    
    def __init__(self, device, model_name: str = "ViT-B/32", cache_dir: Optional[str] = None):
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        # Load CLIP model
        self.clip_model, _ = clip.load(model_name, device=self.device, download_root=cache_dir, jit=False)
        self.clip_model.eval()
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.feature_dim = self.clip_model.text_projection.shape[1]
        print(f"Loaded CLIP model {model_name} with feature dim {self.feature_dim}")
    
    @torch.no_grad()
    def encode_text(self, text_descriptions: List[str]) -> torch.Tensor:
        """
        Encode text descriptions to features
        
        Args:
            text_descriptions: List of text descriptions
        Returns:
            [len(text_descriptions), feature_dim] text features
        """
        if not text_descriptions:
            return torch.empty(0, self.feature_dim).to(self.device)
        
        # Tokenize text
        text_tokens = clip.tokenize(text_descriptions, truncate=True).to(self.device)
        
        # Encode text
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def forward(self, text_descriptions: List[str]) -> torch.Tensor:
        """Forward pass for text encoding"""
        return self.encode_text(text_descriptions)


class Gemma3TextGenerator:
    """
    使用 Gemma 3 多模态模型生成实例描述
    """
    
    def __init__(self, device, local_model_path, torch_dtype = torch.bfloat16):
        """
        初始化 Gemma 3 模型
        """
        self.device = device
        
        print("Loading Gemma 3 model")
        
        # 加载模型和处理器
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            local_model_path,
            device_map=self.device,
            torch_dtype=torch_dtype,
            local_files_only=True,
            attn_implementation="eager"
        )

        self.processor = AutoProcessor.from_pretrained(local_model_path, local_files_only=True, use_fast=True)
        
        # 设置为评估模式
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")

    @torch.no_grad()
    def generate_descriptions(self, 
                            image: torch.Tensor, 
                            instance_mask: torch.Tensor,
                            max_new_tokens: int = 100,
                            do_sample: bool = False) -> Dict[int, str]:
        """
        为每个实例掩码生成文本描述
        
        Args:
            image: [3, H, W] RGB图像张量 (归一化到[0, 1])
            instance_mask: [H, W] 实例分割掩码
            max_new_tokens: 生成的最大token数
            do_sample: 是否采样
        Returns:
            字典 instance_id -> 文本描述
        """

        # 获取唯一实例
        unique_instances = torch.unique(instance_mask)
        # unique_instances = unique_instances[unique_instances > 0]
        
        if len(unique_instances) == 0:
            return {}
        
        descriptions = {}
        
        # 转换图像格式
        image_np = image.cpu().numpy()
        if image_np.shape[0] == 3:  # CHW -> HWC
            image_np = np.transpose(image_np, (1, 2, 0))
        image_np = (image_np * 255).astype(np.uint8)
        
        # 创建原始场景的PIL图像
        original_pil = PILImage.fromarray(image_np)
        
        # 处理每个实例
        for instance_id in unique_instances:
            instance_id_int = instance_id.item()
            instance_binary_mask = (instance_mask == instance_id).cpu().numpy()

            # 检查掩码大小
            mask_area = np.sum(instance_binary_mask)
            total_pixels = instance_binary_mask.shape[0] * instance_binary_mask.shape[1]
            mask_ratio = mask_area / total_pixels
            
            # 设置最小阈值（可根据需要调整）
            min_ratio_threshold = 0.045  # 最小占比

            # 跳过太小的掩码
            if mask_ratio < min_ratio_threshold:
                print(f"Skipping instance {instance_id_int}: too small (area={mask_area}, ratio={mask_ratio:.4f})")
                continue

            # 获取掩码的边界框
            coords = np.where(instance_binary_mask)
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()

            # 添加一些padding（可选，保留一些上下文）
            padding = 10
            y_min = max(0, y_min - padding)
            y_max = min(image_np.shape[0], y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(image_np.shape[1], x_max + padding)

            # 裁剪图像
            cropped_image = image_np[y_min:y_max, x_min:x_max]

            # 创建裁剪后的掩码
            cropped_mask = instance_binary_mask[y_min:y_max, x_min:x_max]

            # 创建白色背景
            cropped_with_bg = np.ones_like(cropped_image) * 255  # 白色背景

            # 只复制掩码覆盖的像素
            cropped_with_bg[cropped_mask] = cropped_image[cropped_mask]

            cropped_pil = PILImage.fromarray(cropped_with_bg.astype(np.uint8))
            
            output_dir = "/media/ssd/jiangxirui/projects/2/data/ScanNetV2/scene0000_00/processed/cropped_instances"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"cropped_instance_{instance_id_int}.png")
            cropped_pil.save(filename)

            # 构建消息格式（根据官方文档）
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant that describes objects in images with rich detail."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": original_pil},
                        {"type": "text", "text": "Please observe this scene carefully. I will ask you about specific objects in it."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I can see the scene. Please show me which specific object you'd like me to describe."}]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": cropped_pil},
                        {"type": "text", "text": f"""Based on the scene I showed you earlier, describe this specific object.
Output Format: [NAMES] | [ATTRIBUTES], [FUNCTION]

NAMES: 5-8 DIFFERENT nouns for this object (NO REPEATS)
- Separated by commas
- From specific to general
- Example: desk, table, workstation, furniture

ATTRIBUTES: [adjective phrase] about physical description (color, material, size, shape)
FUNCTION: [verb phrase] describing its purpose

Example outputs:
"desk, table, workstation, work surface, furniture | wooden brown rectangular, holds office items"
"chair, seat, office chair, swivel chair, furniture | black leather adjustable, provides seating"

Rules:
- First part: ONLY unique nouns, NO repeated words
- Use pipe | to separate parts
- Second part: TWO phrases separated by comma

Output ONLY in the format shown. No other text."""}

#                         {"type": "text", "text": f"""Based on the scene I showed you earlier, describe this specific object using EXACTLY 3 phrases:

# Phrase 1: [adjectives] + [noun1] - describing the object type
# Phrase 2: [adjectives] + [noun2] - describing the same object with a different noun
# Phrase 3: [verb phrase] - describing its function/purpose

# Rules:
# - Use different nouns in phrase 1 and 2
# - Adjectives can describe: size, color, material, shape
# - Function phrase should start with a verb

# Example output: "small wooden desk, beige office table, holds work items"

# Output ONLY the 3 phrases separated by commas. No other text.
# """}
                    ]
                }
            ]
            
            # 应用聊天模板并处理输入
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            
            # 获取输入长度（用于后续解码）
            input_len = inputs["input_ids"].shape[-1]
            
            # 生成描述
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample
                )
                
                # 只提取生成的部分（去除输入部分）
                generated_tokens = generation[0][input_len:]
            
            # 解码生成的文本
            description = self.processor.decode(generated_tokens, skip_special_tokens=True)
            
            # 清理和存储描述
            description = description.strip()
            if description:
                descriptions[instance_id_int] = description
            else:
                descriptions[instance_id_int] = f"Object instance {instance_id_int}"
        
        return descriptions