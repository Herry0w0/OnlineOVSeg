"""
CLIP text encoder wrapper for generating semantic features and LLaVA for text generation
"""
import torch
import torch.nn as nn
import clip
from typing import List, Dict, Optional, Union
import logging
import numpy as np
from PIL import Image
import requests
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration,
    AutoTokenizer,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)

class CLIPTextEncoder(nn.Module):
    """CLIP text encoder for semantic feature extraction"""
    
    def __init__(self, device, model_name: str = "ViT-B/32", cache_dir: Optional[str] = None):
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        # Load CLIP model
        self.clip_model, _ = clip.load(model_name, device=self.device, download_root=cache_dir)
        self.clip_model.eval()
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.feature_dim = self.clip_model.text_projection.shape[1]
        logger.info(f"Loaded CLIP model {model_name} with feature dim {self.feature_dim}")
    
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


class LLaVATextGenerator:
    """
    LLaVA text generator for generating instance descriptions
    Using Hugging Face transformers implementation
    """
    
    def __init__(self, 
                 model_id: str = "llava-hf/llava-1.5-7b-hf",
                 load_in_4bit: bool = True,
                 cache_dir: Optional[str] = None,
                 device_map: Union[str, Dict] = "auto"):
        """
        Initialize LLaVA model
        
        Args:
            model_id: Hugging Face model ID for LLaVA
            load_in_4bit: Whether to load model in 4-bit quantization for memory efficiency
            cache_dir: Directory to cache downloaded models
            device_map: Device mapping for model layers
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading LLaVA model: {model_id}")
        
        # Configure quantization if requested
        if load_in_4bit and self.device.type == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            model_kwargs = {
                "quantization_config": bnb_config,
                "device_map": device_map,
                "torch_dtype": torch.float16
            }
        else:
            model_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32
            }
        
        # Load model and processor
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            **model_kwargs
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        
        # Set model to eval mode
        self.model.eval()
        
        # Conversation template for generating descriptions
        self.conversation_template = """USER: <image>
Please describe each distinct object instance in this image that has the instance ID {instance_id} (shown in the segmentation mask). 
Focus on the object's visual attributes including:
- Object category (chair, table, monitor, etc.)
- Color and material
- Shape and size characteristics
- Any distinctive features

Provide a concise description in one sentence.
ASSISTANT: """
        
        logger.info("LLaVA model loaded successfully")
    
    @torch.no_grad()
    def generate_descriptions(self, 
                            image: torch.Tensor, 
                            instance_mask: torch.Tensor,
                            max_new_tokens: int = 50,
                            temperature: float = 0.2) -> Dict[int, str]:
        """
        Generate text descriptions for each instance in the mask
        
        Args:
            image: [3, H, W] RGB image tensor (normalized to [0, 1])
            instance_mask: [H, W] instance segmentation mask
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (lower = more deterministic)
        Returns:
            Dictionary mapping instance_id -> text description
        """
        # Get unique instances (excluding background)
        unique_instances = torch.unique(instance_mask)
        unique_instances = unique_instances[unique_instances > 0]
        
        if len(unique_instances) == 0:
            return {}
        
        descriptions = {}
        
        # Convert tensor image to PIL Image
        image_np = image.cpu().numpy()
        if image_np.shape[0] == 3:  # CHW -> HWC
            image_np = np.transpose(image_np, (1, 2, 0))
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # Process each instance
        for instance_id in unique_instances:
            instance_id_int = instance_id.item()
            
            # Create visualization with highlighted instance
            # Create a copy of the image with the instance highlighted
            instance_binary_mask = (instance_mask == instance_id).cpu().numpy()
            
            # Create highlighted image
            highlighted_image = image_np.copy()
            # Dim non-instance regions
            highlighted_image[~instance_binary_mask] = (highlighted_image[~instance_binary_mask] * 0.3).astype(np.uint8)
            
            # Add colored overlay for the instance
            overlay = np.zeros_like(highlighted_image)
            overlay[instance_binary_mask] = [255, 255, 0]  # Yellow highlight
            highlighted_image = cv2.addWeighted(highlighted_image, 0.7, overlay, 0.3, 0)
            
            highlighted_pil = Image.fromarray(highlighted_image)
            
            # Prepare prompt
            prompt = self.conversation_template.format(instance_id=instance_id_int)
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=highlighted_pil,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate description
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
            
            # Decode output
            generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
            description = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # Clean up description
            description = description.strip()
            if description:
                descriptions[instance_id_int] = description
            else:
                # Fallback description
                descriptions[instance_id_int] = f"an object instance with ID {instance_id_int}"
        
        return descriptions
    
    def generate_descriptions_batch(self,
                                  images: List[torch.Tensor],
                                  instance_masks: List[torch.Tensor],
                                  batch_size: int = 4,
                                  **kwargs) -> List[Dict[int, str]]:
        """
        Generate descriptions for multiple images in batches
        
        Args:
            images: List of [3, H, W] RGB image tensors
            instance_masks: List of [H, W] instance segmentation masks
            batch_size: Batch size for processing
            **kwargs: Additional arguments for generate_descriptions
        Returns:
            List of dictionaries mapping instance_id -> text description
        """
        all_descriptions = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_masks = instance_masks[i:i+batch_size]
            
            batch_descriptions = []
            for img, mask in zip(batch_images, batch_masks):
                desc = self.generate_descriptions(img, mask, **kwargs)
                batch_descriptions.append(desc)
            
            all_descriptions.extend(batch_descriptions)
        
        return all_descriptions
