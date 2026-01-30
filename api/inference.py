"""
Inference module for RescueNet-SegFormer demo.
Refactored from viz_smooth_stitch.py for web serving.
"""

import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

NUM_CLASSES = 11

CLASS_NAMES = [
    "Background",
    "Water",
    "Building No Damage",
    "Building Minor Damage",
    "Building Major Damage",
    "Building Total Destruction",
    "Vehicle",
    "Road-Clear",
    "Road-Blocked",
    "Tree",
    "Pool",
]

# RGB palette for frontend display
PALETTE_RGB = [
    (0, 0, 0),         # 0  Background
    (0, 0, 255),       # 1  Water (Blue)
    (20, 255, 20),     # 2  No Damage (Green)
    (255, 215, 0),     # 3  Minor Damage (Gold)
    (255, 0, 0),       # 4  Major Damage (Red)
    (139, 0, 0),       # 5  Total Destruction (Dark Red)
    (128, 0, 128),     # 6  Vehicle (Purple)
    (128, 128, 128),   # 7  Road-Clear (Grey)
    (64, 64, 64),      # 8  Road-Blocked (Dark Grey)
    (0, 100, 0),       # 9  Tree (Dark Green)
    (0, 128, 255),     # 10 Pool (Light Blue)
]


def load_model(checkpoint_path: str, device: str = None):
    """Load the trained SegFormer model and processor."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path)
    model = model.to(device).eval()
    processor = SegformerImageProcessor(do_resize=False, do_normalize=True, reduce_labels=False)
    return model, processor, device


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert HxW class index mask to HxWx3 RGB image."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, rgb in enumerate(PALETTE_RGB):
        color[mask == idx] = rgb
    return color


def predict(model, processor, image: Image.Image, device: str,
            crop_size: int = 1024, stride: int = 768) -> tuple[Image.Image, Image.Image]:
    """
    Run sliding-window smooth-stitch inference on a PIL image.
    Returns (colored_mask, overlay) as PIL Images.
    """
    image_np = np.array(image.convert("RGB"))
    h, w, _ = image_np.shape

    full_probs = torch.zeros((NUM_CLASSES, h, w), device=device)
    count_map = torch.zeros((1, h, w), device=device)

    # Small image: pad and predict in one shot
    if h <= crop_size and w <= crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        img_padded = cv2.copyMakeBorder(image_np, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        inputs = processor(images=img_padded, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs.pixel_values.to(device))
            logits = F.interpolate(outputs.logits, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)
        probs = probs[0, :, :h, :w]
        pred_mask = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)
    else:
        # Sliding window with overlap
        n_rows = math.ceil((h - crop_size) / stride) + 1
        n_cols = math.ceil((w - crop_size) / stride) + 1

        for r in range(n_rows):
            for c in range(n_cols):
                y1 = int(r * stride)
                x1 = int(c * stride)
                y2 = min(y1 + crop_size, h)
                x2 = min(x1 + crop_size, w)

                if y2 - y1 < crop_size:
                    y1 = h - crop_size
                if x2 - x1 < crop_size:
                    x1 = w - crop_size
                y2, x2 = y1 + crop_size, x1 + crop_size

                tile = image_np[y1:y2, x1:x2]
                inputs = processor(images=tile, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(inputs.pixel_values.to(device))
                    logits = F.interpolate(outputs.logits, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
                    probs = F.softmax(logits, dim=1)

                full_probs[:, y1:y2, x1:x2] += probs[0]
                count_map[:, y1:y2, x1:x2] += 1.0

        full_probs /= count_map
        pred_mask = torch.argmax(full_probs, dim=0).cpu().numpy().astype(np.uint8)

    # Colorize
    mask_rgb = colorize_mask(pred_mask)
    mask_pil = Image.fromarray(mask_rgb)

    # Overlay (60% original, 40% mask)
    overlay_np = cv2.addWeighted(image_np, 0.6, mask_rgb, 0.4, 0)
    overlay_pil = Image.fromarray(overlay_np)

    return mask_pil, overlay_pil
