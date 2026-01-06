#!/usr/bin/env python3
"""
3D Dose Visualization Script for OpenKBP.

Creates angled 3D views comparing predicted vs true dose distributions
using isosurface rendering.

Usage:
    python visualize_3d.py --model-dir results/hd_unet_lite_20260106_064658 \
        --patient pt_001 --data-split test

    # Multiple isosurface levels
    python visualize_3d.py --model-dir results/hd_unet_lite_20260106_064658 \
        --patient pt_001 --iso-levels 20 40 60
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from src.constants import PATIENT_SHAPE
from src.data_utils import load_sparse_file, sparse_to_dense


def create_isosurface(
    volume: np.ndarray,
    level: float,
    step_size: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create isosurface mesh from 3D volume using marching cubes.
    
    Args:
        volume: 3D numpy array
        level: Isosurface level
        step_size: Step size for marching cubes (higher = faster but coarser)
        
    Returns:
        Tuple of (vertices, faces)
    """
    try:
        verts, faces, _, _ = measure.marching_cubes(
            volume, 
            level=level, 
            step_size=step_size,
            allow_degenerate=False,
        )
        return verts, faces
    except ValueError:
        # No surface at this level
        return None, None


def plot_3d_isosurface(
    volume: np.ndarray,
    ax: plt.Axes,
    levels: List[float],
    colors: List[str],
    alphas: List[float],
    title: str = "",
    step_size: int = 2,
) -> None:
    """
    Plot 3D isosurface on given axes.
    
    Args:
        volume: 3D dose array
        ax: Matplotlib 3D axes
        levels: List of isosurface levels (Gy)
        colors: List of colors for each level
        alphas: List of alpha values for each level
        title: Plot title
        step_size: Step size for marching cubes
    """
    for level, color, alpha in zip(levels, colors, alphas):
        verts, faces = create_isosurface(volume, level, step_size)
        
        if verts is None:
            continue
            
        # Create mesh
        mesh = Poly3DCollection(
            verts[faces],
            alpha=alpha,
            facecolor=color,
            edgecolor='none',
        )
        ax.add_collection3d(mesh)
    
    # Set axis limits
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=12)


def plot_3d_comparison(
    pred_dose: np.ndarray,
    ref_dose: np.ndarray,
    iso_levels: List[float] = [20, 40, 60],
    view_angle: Tuple[float, float] = (30, 45),
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 7),
    step_size: int = 2,
) -> plt.Figure:
    """
    Create side-by-side 3D comparison of predicted vs reference dose.
    
    Args:
        pred_dose: Predicted 3D dose array
        ref_dose: Reference 3D dose array
        iso_levels: Isosurface levels in Gy
        view_angle: Tuple of (elevation, azimuth) for 3D view
        save_path: Path to save figure
        figsize: Figure size
        step_size: Step size for marching cubes (higher = faster)
        
    Returns:
        Matplotlib figure
    """
    # Define colors for isosurface levels (cool to hot)
    cmap = plt.cm.jet
    colors = [cmap(level / max(iso_levels)) for level in iso_levels]
    alphas = [0.3 + 0.2 * i for i in range(len(iso_levels))]
    
    fig = plt.figure(figsize=figsize)
    
    # Predicted dose
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_isosurface(
        pred_dose, ax1, iso_levels, colors, alphas,
        title='Predicted Dose', step_size=step_size,
    )
    ax1.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Reference dose
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_isosurface(
        ref_dose, ax2, iso_levels, colors, alphas,
        title='Reference Dose', step_size=step_size,
    )
    ax2.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Add legend for isosurface levels
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha, label=f'{level} Gy')
        for level, color, alpha in zip(iso_levels, colors, alphas)
    ]
    fig.legend(
        handles=legend_elements, 
        loc='upper center', 
        ncol=len(iso_levels),
        bbox_to_anchor=(0.5, 0.02),
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D visualization to {save_path}")
    
    return fig


def plot_3d_difference(
    pred_dose: np.ndarray,
    ref_dose: np.ndarray,
    threshold: float = 5.0,
    view_angle: Tuple[float, float] = (30, 45),
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
    step_size: int = 2,
) -> plt.Figure:
    """
    Create 3D visualization of dose difference.
    
    Shows regions where prediction differs from reference by more than threshold.
    
    Args:
        pred_dose: Predicted 3D dose array
        ref_dose: Reference 3D dose array
        threshold: Difference threshold in Gy
        view_angle: Tuple of (elevation, azimuth) for 3D view
        save_path: Path to save figure
        figsize: Figure size
        step_size: Step size for marching cubes
        
    Returns:
        Matplotlib figure
    """
    # Calculate absolute difference
    diff = np.abs(pred_dose - ref_dose)
    
    # Create over and under prediction volumes
    over_pred = np.where(pred_dose > ref_dose + threshold, diff, 0)
    under_pred = np.where(pred_dose < ref_dose - threshold, diff, 0)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot over-prediction in red
    verts, faces = create_isosurface(over_pred, threshold, step_size)
    if verts is not None and len(verts) > 0:
        mesh = Poly3DCollection(
            verts[faces],
            alpha=0.5,
            facecolor='red',
            edgecolor='none',
            label='Over-prediction',
        )
        ax.add_collection3d(mesh)
    
    # Plot under-prediction in blue
    verts, faces = create_isosurface(under_pred, threshold, step_size)
    if verts is not None and len(verts) > 0:
        mesh = Poly3DCollection(
            verts[faces],
            alpha=0.5,
            facecolor='blue',
            edgecolor='none',
            label='Under-prediction',
        )
        ax.add_collection3d(mesh)
    
    # Set axis limits
    ax.set_xlim(0, pred_dose.shape[0])
    ax.set_ylim(0, pred_dose.shape[1])
    ax.set_zlim(0, pred_dose.shape[2])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Dose Difference (threshold: ±{threshold} Gy)', fontsize=12)
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.5, label='Over-prediction'),
        Patch(facecolor='blue', alpha=0.5, label='Under-prediction'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D difference visualization to {save_path}")
    
    return fig


def create_rotating_gif(
    pred_dose: np.ndarray,
    ref_dose: np.ndarray,
    iso_levels: List[float],
    output_path: Path,
    n_frames: int = 36,
    step_size: int = 3,
) -> None:
    """
    Create a rotating GIF animation of the 3D dose comparison.
    
    Args:
        pred_dose: Predicted 3D dose array
        ref_dose: Reference 3D dose array
        iso_levels: Isosurface levels
        output_path: Path to save GIF
        n_frames: Number of frames in rotation
        step_size: Step size for marching cubes
    """
    try:
        from PIL import Image
        import io
    except ImportError:
        print("PIL not available, skipping GIF creation")
        return
    
    frames = []
    
    for i, azim in enumerate(np.linspace(0, 360, n_frames, endpoint=False)):
        fig = plot_3d_comparison(
            pred_dose, ref_dose, iso_levels,
            view_angle=(30, azim),
            step_size=step_size,
        )
        
        # Save frame to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
        
        print(f"Frame {i+1}/{n_frames}", end='\r')
    
    # Save GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    print(f"\nSaved rotating GIF to {output_path}")


def visualize_single_patient(
    patient_id: str,
    pred_dir: Path,
    ref_dir: Path,
    output_dir: Path,
    iso_levels: List[float],
    view_angle: Tuple[float, float],
    step_size: int,
    diff_threshold: float,
    create_gif: bool = False,
) -> bool:
    """
    Generate 3D visualizations for a single patient.
    
    Returns True if successful, False otherwise.
    """
    # Load predicted dose
    pred_path = pred_dir / f"{patient_id}.csv"
    if not pred_path.exists():
        print(f"  Skipping {patient_id}: prediction not found")
        return False
    
    pred_sparse = load_sparse_file(pred_path)
    pred_dose = sparse_to_dense(pred_sparse, PATIENT_SHAPE)
    
    # Load reference dose
    ref_path = ref_dir / patient_id / "dose.csv"
    if not ref_path.exists():
        print(f"  Skipping {patient_id}: reference dose not found")
        return False
    
    ref_sparse = load_sparse_file(ref_path)
    ref_dose = sparse_to_dense(ref_sparse, PATIENT_SHAPE)
    
    # Create visualizations
    try:
        # 3D comparison
        fig = plot_3d_comparison(
            pred_dose, ref_dose,
            iso_levels=iso_levels,
            view_angle=view_angle,
            save_path=output_dir / f"{patient_id}_3d_comparison.png",
            step_size=step_size,
        )
        plt.close(fig)
        
        # 3D difference
        fig = plot_3d_difference(
            pred_dose, ref_dose,
            threshold=diff_threshold,
            view_angle=view_angle,
            save_path=output_dir / f"{patient_id}_3d_difference.png",
            step_size=step_size,
        )
        plt.close(fig)
        
        # Rotating GIF
        if create_gif:
            create_rotating_gif(
                pred_dose, ref_dose,
                iso_levels=iso_levels,
                output_path=output_dir / f"{patient_id}_3d_rotation.gif",
                n_frames=36,
                step_size=step_size + 1,
            )
        
        return True
    except Exception as e:
        print(f"  Error processing {patient_id}: {e}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D visualization of dose predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to model results directory",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Patient ID to visualize (e.g., pt_001). Use --all for batch mode.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="process_all",
        help="Process all patients in the data split",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Data split",
    )
    parser.add_argument(
        "--iso-levels",
        type=float,
        nargs="+",
        default=[20, 40, 60],
        help="Isosurface levels in Gy",
    )
    parser.add_argument(
        "--view-angle",
        type=float,
        nargs=2,
        default=[30, 45],
        help="View angle (elevation, azimuth)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: model-dir/visualizations)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=2,
        help="Marching cubes step size (higher = faster but coarser)",
    )
    parser.add_argument(
        "--create-gif",
        action="store_true",
        help="Create rotating GIF animation",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=5.0,
        help="Threshold for difference visualization (Gy)",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Validate arguments
    if not args.process_all and args.patient is None:
        print("Error: Must specify either --patient or --all")
        return
    
    # Set up paths
    data_dir = Path("provided-data")
    if args.data_split == "validation":
        ref_dir = data_dir / "validation-pats"
    else:
        ref_dir = data_dir / "test-pats"
    
    pred_dir = args.model_dir / "evaluation" / args.data_split / "predictions"
    output_dir = args.output_dir or args.model_dir / "visualizations" / "3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 60}")
    print("3D Dose Visualization")
    print("=" * 60)
    print(f"Isosurface levels: {args.iso_levels} Gy")
    print(f"View angle: elev={args.view_angle[0]}, azim={args.view_angle[1]}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    if args.process_all:
        # Batch mode: process all patients
        if not pred_dir.exists():
            print(f"Error: Predictions not found at {pred_dir}")
            print("Run evaluate_hd_unet.py first to generate predictions.")
            return
        
        # Get all patient predictions
        patient_files = sorted(pred_dir.glob("pt_*.csv"))
        patient_ids = [f.stem for f in patient_files]
        
        print(f"\nProcessing {len(patient_ids)} patients...")
        
        successful = 0
        for i, patient_id in enumerate(patient_ids):
            print(f"\n[{i+1}/{len(patient_ids)}] {patient_id}")
            
            success = visualize_single_patient(
                patient_id=patient_id,
                pred_dir=pred_dir,
                ref_dir=ref_dir,
                output_dir=output_dir,
                iso_levels=args.iso_levels,
                view_angle=tuple(args.view_angle),
                step_size=args.step_size,
                diff_threshold=args.diff_threshold,
                create_gif=args.create_gif,
            )
            if success:
                successful += 1
        
        print(f"\n{'=' * 60}")
        print(f"Completed: {successful}/{len(patient_ids)} patients")
        print(f"Visualizations saved to: {output_dir}")
        print("=" * 60)
    
    else:
        # Single patient mode
        patient_id = args.patient
        print(f"Patient: {patient_id}")
        
        # Check prediction exists
        pred_path = pred_dir / f"{patient_id}.csv"
        if not pred_path.exists():
            print(f"Error: Prediction not found at {pred_path}")
            print("Run evaluate_hd_unet.py first to generate predictions.")
            return
        
        # Load and display info
        pred_sparse = load_sparse_file(pred_path)
        pred_dose = sparse_to_dense(pred_sparse, PATIENT_SHAPE)
        print(f"Loaded predicted dose: shape={pred_dose.shape}, max={pred_dose.max():.1f} Gy")
        
        ref_path = ref_dir / patient_id / "dose.csv"
        if not ref_path.exists():
            print(f"Error: Reference dose not found at {ref_path}")
            return
        
        ref_sparse = load_sparse_file(ref_path)
        ref_dose = sparse_to_dense(ref_sparse, PATIENT_SHAPE)
        print(f"Loaded reference dose: shape={ref_dose.shape}, max={ref_dose.max():.1f} Gy")
        
        # Generate visualizations
        print("\nGenerating 3D comparison...")
        fig = plot_3d_comparison(
            pred_dose, ref_dose,
            iso_levels=args.iso_levels,
            view_angle=tuple(args.view_angle),
            save_path=output_dir / f"{patient_id}_3d_comparison.png",
            step_size=args.step_size,
        )
        plt.close(fig)
        
        print("Generating 3D difference visualization...")
        fig = plot_3d_difference(
            pred_dose, ref_dose,
            threshold=args.diff_threshold,
            view_angle=tuple(args.view_angle),
            save_path=output_dir / f"{patient_id}_3d_difference.png",
            step_size=args.step_size,
        )
        plt.close(fig)
        
        if args.create_gif:
            print("Generating rotating GIF (this may take a minute)...")
            create_rotating_gif(
                pred_dose, ref_dose,
                iso_levels=args.iso_levels,
                output_path=output_dir / f"{patient_id}_3d_rotation.gif",
                n_frames=36,
                step_size=args.step_size + 1,
            )
        
        print(f"\nVisualizations saved to: {output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    main()
