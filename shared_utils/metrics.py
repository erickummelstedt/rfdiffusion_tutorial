"""
Metrics for evaluating protein structure quality and design success.

This module provides functions for calculating various quality metrics,
structural similarity measures, and design validation scores.
"""

from typing import Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray,
                   align: bool = True) -> float:
    """
    Calculate Root Mean Square Deviation between two structures.
    
    Args:
        coords1: First coordinate array (n, 3)
        coords2: Second coordinate array (n, 3)
        align: Whether to align structures before RMSD calculation
        
    Returns:
        RMSD value in Angstroms
    """
    if coords1.shape != coords2.shape:
        raise ValueError("Coordinate arrays must have the same shape")
    
    if align:
        from .structure_utils import align_structures
        coords1_aligned, rmsd = align_structures(coords1, coords2)
        return rmsd
    else:
        diff = coords1 - coords2
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=-1)))
        return rmsd


def calculate_tm_score(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculate TM-score (Template Modeling score) between two structures.
    
    TM-score ranges from 0 to 1, with 1 indicating perfect match.
    Structures with TM-score > 0.5 are generally considered similar.
    
    Args:
        coords1: First coordinate array (n, 3)
        coords2: Second coordinate array (n, 3)
        
    Returns:
        TM-score value
    """
    from .structure_utils import align_structures
    
    n = len(coords1)
    d0 = 1.24 * (n - 15) ** (1/3) - 1.8  # Length-dependent scaling
    
    # Align structures
    coords1_aligned, _ = align_structures(coords1, coords2)
    
    # Calculate distances
    distances = np.sqrt(np.sum((coords1_aligned - coords2)**2, axis=-1))
    
    # Calculate TM-score
    tm_score = np.sum(1 / (1 + (distances / d0)**2)) / n
    
    return tm_score


def calculate_radius_of_gyration(coords: np.ndarray) -> float:
    """
    Calculate radius of gyration (measure of compactness).
    
    Args:
        coords: Coordinate array (n, 3)
        
    Returns:
        Radius of gyration in Angstroms
    """
    center = np.mean(coords, axis=0)
    distances_sq = np.sum((coords - center)**2, axis=-1)
    rg = np.sqrt(np.mean(distances_sq))
    return rg


def calculate_clash_score(coords: np.ndarray, clash_threshold: float = 2.0) -> int:
    """
    Calculate number of atomic clashes (pairs closer than threshold).
    
    Args:
        coords: Coordinate array (n, 3)
        clash_threshold: Distance threshold for clash (Angstroms)
        
    Returns:
        Number of clashes
    """
    from .structure_utils import calculate_distance_matrix
    
    dist_matrix = calculate_distance_matrix(coords)
    
    # Exclude self-interactions and neighbors
    mask = np.triu(np.ones_like(dist_matrix), k=2).astype(bool)
    
    # Count clashes
    clashes = np.sum((dist_matrix < clash_threshold) & mask)
    
    return int(clashes)


def calculate_secondary_structure_content(phi: np.ndarray, 
                                         psi: np.ndarray) -> dict:
    """
    Estimate secondary structure content from backbone angles.
    
    Args:
        phi: Phi angles in radians
        psi: Psi angles in radians
        
    Returns:
        Dictionary with fractions of helix, sheet, and coil
    """
    # Convert to degrees
    phi_deg = np.degrees(phi[~np.isnan(phi)])
    psi_deg = np.degrees(psi[~np.isnan(psi)])
    
    n_residues = len(phi_deg)
    
    # Define regions (simplified)
    # Helix: phi ~ -60, psi ~ -45
    helix_mask = (np.abs(phi_deg + 60) < 30) & (np.abs(psi_deg + 45) < 30)
    
    # Sheet: phi ~ -120, psi ~ 120
    sheet_mask = (np.abs(phi_deg + 120) < 40) & (np.abs(psi_deg - 120) < 40)
    
    n_helix = np.sum(helix_mask)
    n_sheet = np.sum(sheet_mask)
    n_coil = n_residues - n_helix - n_sheet
    
    return {
        'helix': n_helix / n_residues,
        'sheet': n_sheet / n_residues,
        'coil': n_coil / n_residues
    }


def calculate_packing_density(coords: np.ndarray, 
                              radius: float = 10.0) -> float:
    """
    Calculate average packing density (atoms within radius).
    
    Args:
        coords: Coordinate array (n, 3)
        radius: Neighborhood radius in Angstroms
        
    Returns:
        Average number of neighbors per atom
    """
    from .structure_utils import calculate_distance_matrix
    
    dist_matrix = calculate_distance_matrix(coords)
    
    # Count neighbors within radius (excluding self)
    neighbors = np.sum((dist_matrix < radius) & (dist_matrix > 0), axis=1)
    
    avg_density = np.mean(neighbors)
    return avg_density


def calculate_structure_quality_score(coords: np.ndarray, 
                                     phi: Optional[np.ndarray] = None,
                                     psi: Optional[np.ndarray] = None) -> dict:
    """
    Calculate comprehensive structure quality metrics.
    
    Args:
        coords: Coordinate array (n, 3)
        phi: Optional phi angles
        psi: Optional psi angles
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Geometric properties
    metrics['radius_of_gyration'] = calculate_radius_of_gyration(coords)
    metrics['packing_density'] = calculate_packing_density(coords)
    metrics['clash_score'] = calculate_clash_score(coords)
    
    # Secondary structure (if angles provided)
    if phi is not None and psi is not None:
        ss_content = calculate_secondary_structure_content(phi, psi)
        metrics.update(ss_content)
    
    # Overall quality score (simple heuristic)
    # Lower is better for clashes, higher for packing
    metrics['quality_score'] = (
        metrics['packing_density'] / 10.0 - 
        metrics['clash_score'] / 10.0
    )
    
    return metrics


def calculate_interface_rmsd(coords1: np.ndarray, coords2: np.ndarray,
                             interface_residues: np.ndarray) -> float:
    """
    Calculate RMSD only for interface residues.
    
    Args:
        coords1: First coordinate array
        coords2: Second coordinate array
        interface_residues: Boolean mask or indices of interface residues
        
    Returns:
        Interface RMSD
    """
    interface_coords1 = coords1[interface_residues]
    interface_coords2 = coords2[interface_residues]
    
    return calculate_rmsd(interface_coords1, interface_coords2, align=True)


def calculate_designability_score(sequence: str, structure_coords: np.ndarray,
                                  predicted_coords: np.ndarray) -> dict:
    """
    Calculate metrics for design success (sequence -> structure prediction).
    
    Args:
        sequence: Designed amino acid sequence
        structure_coords: Designed structure coordinates
        predicted_coords: Predicted structure from sequence
        
    Returns:
        Dictionary with design metrics
    """
    metrics = {}
    
    # Structure similarity
    metrics['rmsd'] = calculate_rmsd(structure_coords, predicted_coords)
    metrics['tm_score'] = calculate_tm_score(structure_coords, predicted_coords)
    
    # Success criteria (commonly used thresholds)
    metrics['design_success'] = (metrics['rmsd'] < 2.0) and (metrics['tm_score'] > 0.5)
    
    return metrics
