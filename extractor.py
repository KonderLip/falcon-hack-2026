"""
Extractor contract for Vehicle ReID benchmark.

Основной интерфейс для замеров производительности:
    extractor = VehicleExtractor(weights_dir="weights/", device="cuda:0")
    embeds = extractor.extract_batch(image_paths, bboxes) # (N, D) float32, L2-норм.

Полный цикл на одно ТС:
    чтение файла -> декодирование -> crop по BBox -> preprocessing ->
    forward -> postprocessing -> L2-нормализация
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from transformers import ConvNextV2ForImageClassification

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


class VehicleExtractor:
    def __init__(
        self,
        weights_dir: str | Path = "weights/",
        device: str = "cuda:0",
        batch_size: int = 32,
    ) -> None:
        self.weights_dir = Path(weights_dir)
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.backbone = (
            ConvNextV2ForImageClassification.from_pretrained(self.weights_dir)
            .to(self.device)
            .eval()
        )
        self.proj = nn.Linear(1024, 512, bias=False).to(self.device)
        self.proj.load_state_dict(
            torch.load(
                self.weights_dir / "proj.pt",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.proj.eval()

        self.transform = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
                T.ConvertImageDtype(torch.float),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_one(image_path: Path, bbox: BBox, image: torch.Tensor) -> None:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            raise ValueError(f"Non-positive w/h {bbox} for {image_path}")
        _, H, W = image.shape
        if x < 0 or y < 0 or x + w > W or y + h > H:
            raise ValueError(
                f"BBox {bbox} out of bounds {image.shape} for {image_path}"
            )

    # ------------------------------------------------------------------ #
    # Single-image path
    # ------------------------------------------------------------------ #

    def _load_crop(self, image_path: Path, bbox: BBox) -> torch.Tensor:
        image = torchvision.io.read_image(str(image_path))
        if image.shape[0] == 4:
            image = image[:3]
        elif image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        self._validate_one(image_path, bbox, image)

        x, y, w, h = bbox
        crop = image[:, y : y + h, x : x + w]
        return self.transform(crop)

    @torch.inference_mode()
    def extract(self, image_path: str | Path, bbox: BBox) -> np.ndarray:
        tensor = self._load_crop(Path(image_path), bbox).unsqueeze(0)
        return self._forward(tensor).squeeze(0).numpy()

    # ------------------------------------------------------------------ #
    # Batch path
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def extract_batch(
        self,
        image_paths: Sequence[str | Path],
        bboxes: Sequence[BBox],
    ) -> np.ndarray:
        if len(image_paths) != len(bboxes):
            raise ValueError(
                f"len(image_paths)={len(image_paths)} != len(bboxes)={len(bboxes)}"
            )
        if len(image_paths) == 0:
            return np.zeros((0, 512), dtype=np.float32)

        chunks: List[np.ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            end = start + self.batch_size
            batch_imgs = [
                self._load_crop(Path(p), b)
                for p, b in zip(image_paths[start:end], bboxes[start:end])
            ]
            tensor = torch.stack(batch_imgs, dim=0)
            chunks.append(self._forward(tensor).numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.device)
        embeds = self.proj(self.backbone(pixel_values=tensor).logits)
        embeds = F.normalize(embeds.float(), p=2, dim=-1)
        return embeds.cpu()
