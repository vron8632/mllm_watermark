"""
YOLO26-seg Instance Segmentation → Content-Adaptive Embedding Strength Map

YOLO26-seg detects semantic instances (persons, cars, objects) in the image.
A strength map is generated where:
  - Instance regions: high strength (0.8) — more watermark bits embedded here
  - Background regions: low strength (0.3) — fewer watermark bits

This enables graceful degradation: tampering with one object destroys
only that instance's watermark, while other instances remain verifiable.
"""
import torch
from PIL import Image
from torchvision import transforms as T
from torch.nn.functional import interpolate
from typing import Optional


class StrengthMapGenerator:
    """
    Generate content-adaptive watermark embedding strength map
    using YOLO26-seg instance segmentation.
    """

    def __init__(self, img_size: int = 256):
        self.img_size = img_size
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        self.model: Optional[torch.nn.Module] = None

    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO('yolo26n-seg.pt')

    def generate(self, image_path: str) -> torch.Tensor:
        """
        Generate strength map from image.

        Args:
            image_path: path to input image

        Returns:
            strength_map: (1, H, W) tensor, values in [0.1, 1.0]
        """
        self._load_model()
        pil_image = Image.open(image_path).convert('RGB')
        H, W = pil_image.size

        # Run YOLO26-seg inference
        results = self.model(pil_image, verbose=False)
        r = results[0]

        # Initialize with background strength
        strength = torch.ones(1, self.img_size, self.img_size) * 0.3

        if r.masks is not None and len(r.masks) > 0:
            # Get instance masks (N, H_yolo, W_yolo)
            masks = r.masks.data.cpu()

            # Resize to target size
            masks_resized = interpolate(
                masks.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False,
            )[0]

            # Accumulate instance strengths
            for i in range(len(masks_resized)):
                strength = torch.maximum(
                    strength,
                    masks_resized[i:i+1] * 0.8
                )

            # Smooth with Gaussian blur
            strength = T.GaussianBlur(
                kernel_size=15, sigma=5
            )(strength)

        return strength.clamp(0.1, 1.0)
