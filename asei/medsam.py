"""Frozen FP32 MedSAM ViT-B and a reusable image embedding cache."""
from functools import partial
from pathlib import Path
import hashlib
import uuid
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from third_party.segment_anything.modeling import (
    Sam, ImageEncoderViT, PromptEncoder, MaskDecoder, TwoWayTransformer,
)
from .utils import json_digest, sha_file, write_json


class MedSAM:
    def __init__(self, checkpoint, device):
        self.device = device
        self.sam = Sam(
            image_encoder=ImageEncoderViT(depth=12, embed_dim=768, img_size=1024,
                mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_heads=12,
                patch_size=16, qkv_bias=True, use_rel_pos=True,
                global_attn_indexes=[2, 5, 8, 11], window_size=14, out_chans=256),
            prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(64, 64),
                input_image_size=(1024, 1024), mask_in_chans=16),
            mask_decoder=MaskDecoder(num_multimask_outputs=3,
                transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
                transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256),
            pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375])
        self.sam.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.sam.to(device=device, dtype=torch.float32).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode_batch(self, images):
        images = F.interpolate(images.float().repeat(1, 3, 1, 1), (1024, 1024),
                               mode='bicubic', align_corners=False, antialias=True)
        low = images.amin((1, 2, 3), keepdim=True)
        scale = images.amax((1, 2, 3), keepdim=True) - low
        return self.sam.image_encoder((images - low) / scale.clamp_min(1e-8))

    @torch.inference_mode()
    def probabilities(self, embedding, boxes, shape, batch_size=24):
        if not len(boxes):
            return np.empty((0, *shape), dtype=np.float32)
        if embedding is None:
            raise ValueError('An image embedding is required for nonempty prompts')
        boxes = torch.as_tensor(np.asarray(boxes), dtype=torch.float32, device=self.device)
        scale = boxes.new_tensor([1024 / shape[1], 1024 / shape[0]] * 2)
        results = []
        for start in range(0, len(boxes), batch_size):
            sparse, dense = self.sam.prompt_encoder(points=None,
                boxes=(boxes[start:start+batch_size] * scale)[:, None], masks=None)
            logits, _ = self.sam.mask_decoder(image_embeddings=embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False)
            probabilities = F.interpolate(logits.sigmoid(), shape, mode='bilinear', align_corners=False)
            results.append(probabilities[:, 0].cpu().numpy())
        return np.concatenate(results)


class EmbeddingCache:
    def __init__(self, root, checkpoint, device, encoder_batch_size=2):
        descriptor = dict(version=1, checkpoint=sha_file(checkpoint), preprocessing='repeat3_bicubic1024_minmax_fp32',
            encoder_batch_size=encoder_batch_size, torch=str(torch.__version__),
            device=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)
        self.root = Path(root) / json_digest(descriptor)
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / 'descriptor.json', descriptor)

    @staticmethod
    def image_key(image):
        image = np.ascontiguousarray(image, dtype=np.float32)
        return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()

    def path(self, key):
        return self.root / key[:2] / (key + '.npy')

    def get(self, key):
        path = self.path(key)
        if not path.is_file():
            return None
        value = np.load(path, allow_pickle=False)
        if value.shape != (1, 256, 64, 64) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'Invalid cached image feature: {path}')
        return value

    def put(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (1, 256, 64, 64) or not np.isfinite(value).all():
            raise ValueError('Expected a finite FP32 [1,256,64,64] image embedding')
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, value, allow_pickle=False)
        temporary.replace(path)
