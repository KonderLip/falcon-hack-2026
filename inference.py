"""
Inference entrypoint for Vehicle ReID.

Usage:
    python -m inference \
        --images-dir data/images \
        --query-csv data/test_query.csv \
        --gallery-csv data/test_gallery.csv \
        --output-dir submission/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from transformers import ConvNextV2ForImageClassification


class Model(nn.Module):
    def __init__(self, weights_dir: Path, device: str) -> None:
        super(Model, self).__init__()
        self.backbone = ConvNextV2ForImageClassification.from_pretrained(
            weights_dir, device_map=device
        )
        self.proj = nn.Linear(1024, 512, bias=False, device=device)
        self.proj.load_state_dict(
            torch.load(weights_dir / "proj.pt", map_location=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.proj(self.backbone(pixel_values=x).logits)
        embeds = F.normalize(embeds, p=2, dim=-1)
        return embeds


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, data, transform=None):
        self.root_path = root_path
        self.data = data[["image_id", "x", "y", "w", "h"]].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]

        img_path = (self.root_path / row[0]).with_suffix(".jpg")
        assert img_path.is_file()
        image = torchvision.io.read_image(img_path)

        x, y, w, h = row[1:]
        image = image[:, y : y + h, x : x + w]
        assert image.shape == (3, h, w)

        if self.transform is not None:
            image = self.transform(image)

        return image


def run_inference(
    images_dir: Path,
    query_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    model: Model,
    output_dir: Path,
    threshold: float,
    batch_size: int,
    num_workers: int,
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)

    device = model.backbone.device

    transforms = T.Compose(
        [
            T.Resize((224, 224), antialias=True),
            T.ConvertImageDtype(torch.float),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataset = torch.utils.data.ConcatDataset(
        [
            InferenceDataset(images_dir, query_df, transforms),
            InferenceDataset(images_dir, gallery_df, transforms),
        ]
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    y_pred = []

    with torch.inference_mode():
        for x in dataloader:
            x = x.to(device, non_blocking=True)
            embeds = model(x)
            y_pred.append(embeds)

    y_pred = torch.cat(y_pred).cpu().float().numpy()
    np.save(output_dir / "embeddings.npy", y_pred)

    query_embeds = y_pred[: len(query_df)]
    gallery_embeds = y_pred[-len(gallery_df) :]

    sim = query_embeds @ gallery_embeds.T
    order = np.argsort(-sim, axis=1)[:, :top_k]

    query_ids = query_df["image_id"].to_numpy()
    gallery_ids = gallery_df["image_id"].to_numpy()

    rows = np.empty((len(query_ids), top_k + 1), dtype=object)
    rows[:, 0] = query_ids
    rows[:, 1:] = gallery_ids[order]
    submission_df = pd.DataFrame(rows)
    np.savetxt(
        output_dir / "submission.csv",
        rows,
        fmt="%s",
        delimiter=",",
    )

    top1_scores = sim[np.arange(len(query_ids)), order[:, 0]]
    keep = top1_scores >= threshold
    candidates_df = pd.DataFrame(
        {
            "query_id": query_ids[keep],
            "gallery_id": gallery_ids[order[keep, 0]],
            "confidence": top1_scores[keep],
        },
        columns=["query_id", "gallery_id", "confidence"],
    )
    candidates_df.to_csv(output_dir / "candidates.csv", index=False)

    return submission_df, candidates_df, y_pred


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle ReID inference")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--query-csv", type=Path, required=True)
    parser.add_argument("--gallery-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default="weights/")
    parser.add_argument("--threshold", type=float, default=0.7578)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    query_df = pd.read_csv(args.query_csv)
    gallery_df = pd.read_csv(args.gallery_csv)

    model = Model(args.weights_dir, args.device)

    run_inference(
        images_dir=args.images_dir,
        query_df=query_df,
        gallery_df=gallery_df,
        model=model,
        output_dir=args.output_dir,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
