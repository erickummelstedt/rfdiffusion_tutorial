"""
Visualization utilities for protein structures and training metrics.

This module provides functions for visualizing protein structures, training progress,
and evaluation metrics using Matplotlib, Plotly, and py3Dmol.
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Try to import py3Dmol (optional dependency)
try:
    import py3Dmol
    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False


def plot_ramachandran(phi: np.ndarray, psi: np.ndarray, 
                      title: str = "Ramachandran Plot") -> plt.Figure:
    """
    Create a Ramachandran plot of backbone dihedral angles.
    
    Args:
        phi: Phi angles in radians
        psi: Psi angles in radians
        title: Plot title
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Convert to degrees
    phi_deg = np.degrees(phi[~np.isnan(phi)])
    psi_deg = np.degrees(psi[~np.isnan(psi)])
    
    # Create 2D histogram for background
    h, xedges, yedges = np.histogram2d(phi_deg, psi_deg, bins=36)
    ax.contourf(xedges[:-1], yedges[:-1], h.T, levels=10, cmap='Blues', alpha=0.3)
    
    # Scatter plot of points
    ax.scatter(phi_deg, psi_deg, alpha=0.5, s=20)
    
    ax.set_xlabel('Phi (degrees)', fontsize=12)
    ax.set_ylabel('Psi (degrees)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_distance_matrix(distance_matrix: np.ndarray, 
                         title: str = "Contact Map") -> plt.Figure:
    """
    Visualize a protein distance/contact matrix.
    
    Args:
        distance_matrix: Square distance matrix
        title: Plot title
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(distance_matrix, cmap='viridis_r', aspect='auto')
    ax.set_xlabel('Residue Index', fontsize=12)
    ax.set_ylabel('Residue Index', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Distance (Å)', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_structure_3d(coords: np.ndarray, 
                      title: str = "Protein Structure") -> plt.Figure:
    """
    Create a 3D visualization of protein backbone.
    
    Args:
        coords: Coordinates of shape (n_residues, 3) or (n_residues, 3, 3)
        title: Plot title
        
    Returns:
        Matplotlib Figure object
    """
    if coords.ndim == 3:
        # Extract CA coordinates
        coords = coords[:, 1, :]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot backbone as line
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
            'b-', linewidth=2, alpha=0.6)
    
    # Plot CA atoms as spheres
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
              c=np.arange(len(coords)), cmap='rainbow', s=50)
    
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_zlabel('Z (Å)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig


def visualize_pdb_interactive(pdb_file: str, style: str = "cartoon",
                              color: str = "spectrum") -> Optional[object]:
    """
    Create an interactive 3D visualization of a PDB file using py3Dmol.
    
    Args:
        pdb_file: Path to PDB file
        style: Visualization style ('cartoon', 'stick', 'sphere', 'line')
        color: Color scheme ('spectrum', 'chain', 'residue')
        
    Returns:
        py3Dmol view object (if py3Dmol is available)
    """
    if not HAS_PY3DMOL:
        print("py3Dmol not available. Install with: pip install py3Dmol")
        return None
    
    with open(pdb_file, 'r') as f:
        pdb_data = f.read()
    
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    view.setStyle({style: {'color': color}})
    view.zoomTo()
    
    return view


def plot_training_curves(losses: List[float], 
                         metrics: Optional[dict] = None,
                         title: str = "Training Progress") -> plt.Figure:
    """
    Plot training loss and optional metrics over time.
    
    Args:
        losses: List of loss values
        metrics: Optional dictionary of metric name -> values
        title: Plot title
        
    Returns:
        Matplotlib Figure object
    """
    n_plots = 1 + (len(metrics) if metrics else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot loss
    axes[0].plot(losses, linewidth=2)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics
    if metrics:
        for i, (metric_name, values) in enumerate(metrics.items(), start=1):
            axes[i].plot(values, linewidth=2, color='orange')
            axes[i].set_xlabel('Iteration', fontsize=12)
            axes[i].set_ylabel(metric_name, fontsize=12)
            axes[i].set_title(metric_name, fontsize=14)
            axes[i].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def plot_secondary_structure(ss_sequence: str, 
                             title: str = "Secondary Structure") -> plt.Figure:
    """
    Visualize secondary structure as a colored bar.
    
    Args:
        ss_sequence: String of secondary structure codes (H=helix, E=sheet, C=coil)
        title: Plot title
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(15, 2))
    
    # Color mapping
    colors = {'H': 'red', 'E': 'yellow', 'C': 'gray', '-': 'white'}
    color_list = [colors.get(ss, 'gray') for ss in ss_sequence]
    
    # Create colored bars
    ax.bar(range(len(ss_sequence)), [1]*len(ss_sequence), 
           color=color_list, width=1.0, edgecolor='none')
    
    ax.set_xlim(0, len(ss_sequence))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Helix (H)'),
        Patch(facecolor='yellow', label='Sheet (E)'),
        Patch(facecolor='gray', label='Coil (C)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig
