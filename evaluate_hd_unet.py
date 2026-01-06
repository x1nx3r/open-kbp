#!/usr/bin/env python3
"""
HD U-Net evaluation script for OpenKBP dose prediction.

This script evaluates a trained HD U-Net model on the validation or test set
using the official OpenKBP competition metrics.

Features:
- Loads trained HD U-Net model
- Runs inference on validation/test set
- Computes dose score (MAE) and DVH score
- Exports results in competition format

Usage:
    # Evaluate on validation set
    python evaluate_hd_unet.py --model-dir results/hd_unet_lite_20260106_063133

    # Evaluate on test set (competition evaluation)
    python evaluate_hd_unet.py --model-dir results/hd_unet_lite_20260106_063133 --data-split test

    # Specify output directory
    python evaluate_hd_unet.py --model-dir results/hd_unet_lite_20260106_063133 --output-dir evaluation_results
"""

import argparse
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from src import OpenKBPDataset, get_inference_transforms, get_patient_dirs
from src.evaluation import evaluate_predictions, print_evaluation_results
from src.export import export_evaluation_results
from src.hd_unet import get_hd_unet
from src.hd_unet_model import HDUNetDosePredictionModel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate HD U-Net for OpenKBP dose prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to model results directory (contains models/ subfolder)",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for predictions and results (default: model-dir/evaluation)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="lite",
        choices=["lite", "standard", "large"],
        help="HD U-Net variant (must match trained model)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)",
    )

    return parser.parse_args()


def load_model(
    model_dir: Path,
    variant: Literal["lite", "standard", "large"],
    device: torch.device,
) -> torch.nn.Module:
    """
    Load trained HD U-Net model from checkpoint.

    Args:
        model_dir: Path to model results directory
        variant: HD U-Net variant
        device: PyTorch device

    Returns:
        Loaded model in eval mode
    """
    # Create model with same architecture
    from src.constants import NUM_INPUT_CHANNELS

    model = get_hd_unet(
        variant=variant,
        in_channels=NUM_INPUT_CHANNELS,
        out_channels=1,
        use_checkpoint=False,  # Not needed for inference
        use_attention=variant != "lite",  # Lite has no attention
        deep_supervision=False,  # Disabled for inference
    )

    # Load weights
    model_path = model_dir / "models" / "best_model.pt"
    if not model_path.exists():
        # Try to find latest checkpoint
        checkpoints = list((model_dir / "models").glob("epoch_*.pt"))
        if checkpoints:
            epochs = [int(c.stem.split("_")[1]) for c in checkpoints]
            latest_epoch = max(epochs)
            model_path = model_dir / "models" / f"epoch_{latest_epoch}.pt"
            print(f"Best model not found, using checkpoint from epoch {latest_epoch}")
        else:
            raise FileNotFoundError(f"No model checkpoints found in {model_dir / 'models'}")

    # Load state dict
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {model_path}")
    return model


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
) -> None:
    """
    Run inference and save predictions in sparse CSV format.

    Args:
        model: Trained model
        data_loader: DataLoader for evaluation data
        output_dir: Directory to save predictions
        device: PyTorch device
    """
    import numpy as np
    import pandas as pd
    from torch.amp import autocast
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(data_loader, desc="Generating predictions"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        patient_ids = batch["patient_id"]

        # Forward pass with AMP
        with autocast("cuda", enabled=device.type == "cuda"):
            pred = model(images)

            # Handle deep supervision output (only use main output)
            if isinstance(pred, tuple):
                pred = pred[0]

            # Apply ReLU to ensure non-negative dose
            pred = torch.relu(pred)

        # Apply mask
        pred = pred * masks

        # Convert to numpy and save
        pred_np = pred.float().cpu().numpy()

        for i, patient_id in enumerate(patient_ids):
            dose_pred = pred_np[i, 0]  # Shape: (128, 128, 128)

            # Save in sparse CSV format (competition format)
            flat_dose = dose_pred.flatten()
            nonzero_mask = flat_dose > 0
            indices = np.where(nonzero_mask)[0]
            values = flat_dose[nonzero_mask]

            df = pd.DataFrame({"data": values}, index=indices)
            df.to_csv(output_dir / f"{patient_id}.csv")


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Set output directory
    if args.output_dir is None:
        output_dir = args.model_dir / "evaluation" / args.data_split
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data directory
    data_dir = Path("provided-data")
    if args.data_split == "validation":
        eval_dir = data_dir / "validation-pats"
    else:
        eval_dir = data_dir / "test-pats"

    print(f"\n{'=' * 60}")
    print("HD U-Net Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"Data split: {args.data_split}")
    print(f"Evaluation data: {eval_dir}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Get patient directories
    patient_dirs = get_patient_dirs(eval_dir)
    print(f"\nNumber of patients: {len(patient_dirs)}")

    # Create dataset and loader
    transform = get_inference_transforms()
    dataset = OpenKBPDataset(
        patient_dirs=patient_dirs,
        transform=transform,
        include_dose=True,  # Needed for evaluation
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load model
    model = load_model(args.model_dir, args.variant, device)

    # Run inference
    pred_dir = output_dir / "predictions"
    print(f"\nGenerating predictions...")
    run_inference(model, data_loader, pred_dir, device)
    print(f"Predictions saved to: {pred_dir}")

    # Evaluate predictions
    print(f"\nEvaluating predictions...")
    results = evaluate_predictions(
        pred_dir=pred_dir,
        ref_dir=eval_dir,
        verbose=True,
    )

    # Print results
    print_evaluation_results(results)

    # Export results
    results_path = output_dir / "evaluation_results.json"
    export_evaluation_results(
        results=results,
        output_path=results_path,
        model_name=args.model_dir.stem,
        config={
            "variant": args.variant,
            "data_split": args.data_split,
            "num_patients": len(patient_dirs),
        },
    )
    print(f"\nResults exported to: {results_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
