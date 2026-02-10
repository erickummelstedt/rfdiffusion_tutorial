"""
Protein structure utilities for loading, manipulating, and saving protein structures.

This module provides functions for working with PDB files, calculating geometric
properties, and performing common structure manipulations.
"""

from typing import Tuple, Optional, List
import numpy as np
from Bio import PDB
from Bio.PDB import Structure, Chain, Residue, Atom
import logging

logger = logging.getLogger(__name__)


def load_pdb(pdb_file: str) -> Structure.Structure:
    """
    Load a protein structure from a PDB file.
    
    Args:
        pdb_file: Path to PDB file
        
    Returns:
        BioPython Structure object
        
    Example:
        >>> structure = load_pdb("protein.pdb")
        >>> print(f"Loaded {len(list(structure.get_residues()))} residues")
    """
    parser = PDB.PDBParser(QUIET=True)
    structure_id = pdb_file.split("/")[-1].replace(".pdb", "")
    structure = parser.get_structure(structure_id, pdb_file)
    logger.info(f"Loaded structure from {pdb_file}")
    return structure


def save_pdb(structure: Structure.Structure, output_file: str) -> None:
    """
    Save a protein structure to a PDB file.
    
    Args:
        structure: BioPython Structure object
        output_file: Path for output PDB file
    """
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)
    logger.info(f"Saved structure to {output_file}")


def get_backbone_coords(structure: Structure.Structure) -> np.ndarray:
    """
    Extract backbone atom coordinates (N, CA, C) from a structure.
    
    Args:
        structure: BioPython Structure object
        
    Returns:
        Array of shape (n_residues, 3, 3) containing [N, CA, C] coordinates
    """
    coords = []
    for residue in structure.get_residues():
        if residue.id[0] == " ":  # Skip heteroatoms
            try:
                n = residue["N"].get_coord()
                ca = residue["CA"].get_coord()
                c = residue["C"].get_coord()
                coords.append([n, ca, c])
            except KeyError:
                logger.warning(f"Missing backbone atoms in residue {residue.id}")
                continue
    return np.array(coords)


def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise distance matrix between CA atoms.
    
    Args:
        coords: Array of shape (n_residues, 3) or (n_residues, 3, 3)
        
    Returns:
        Distance matrix of shape (n_residues, n_residues)
    """
    if coords.ndim == 3:
        # Extract CA coordinates (middle atom)
        coords = coords[:, 1, :]
    
    # Calculate pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    return distances


def calculate_dihedral(p1: np.ndarray, p2: np.ndarray, 
                       p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Calculate dihedral angle between four points.
    
    Args:
        p1, p2, p3, p4: 3D coordinate arrays
        
    Returns:
        Dihedral angle in radians
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    return np.arctan2(y, x)


def get_phi_psi(structure: Structure.Structure) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate phi and psi backbone dihedral angles.
    
    Args:
        structure: BioPython Structure object
        
    Returns:
        Tuple of (phi_angles, psi_angles) arrays
    """
    residues = list(structure.get_residues())
    n_residues = len(residues)
    
    phi_angles = np.zeros(n_residues)
    psi_angles = np.zeros(n_residues)
    
    for i in range(1, n_residues - 1):
        try:
            # Phi: C(-1) - N - CA - C
            c_prev = residues[i-1]["C"].get_coord()
            n = residues[i]["N"].get_coord()
            ca = residues[i]["CA"].get_coord()
            c = residues[i]["C"].get_coord()
            phi_angles[i] = calculate_dihedral(c_prev, n, ca, c)
            
            # Psi: N - CA - C - N(+1)
            n_next = residues[i+1]["N"].get_coord()
            psi_angles[i] = calculate_dihedral(n, ca, c, n_next)
        except KeyError:
            logger.warning(f"Missing atoms for dihedral calculation at residue {i}")
            phi_angles[i] = np.nan
            psi_angles[i] = np.nan
    
    return phi_angles, psi_angles


def center_structure(coords: np.ndarray) -> np.ndarray:
    """
    Center coordinates at the origin.
    
    Args:
        coords: Coordinate array
        
    Returns:
        Centered coordinates
    """
    center = np.mean(coords.reshape(-1, 3), axis=0)
    return coords - center


def align_structures(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Align mobile structure to target using SVD.
    
    Args:
        mobile: Mobile coordinate array (n, 3)
        target: Target coordinate array (n, 3)
        
    Returns:
        Tuple of (aligned_mobile, rmsd)
    """
    # Center both structures
    mobile_centered = mobile - np.mean(mobile, axis=0)
    target_centered = target - np.mean(target, axis=0)
    
    # Calculate rotation matrix using SVD
    H = mobile_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation
    aligned_mobile = mobile_centered @ R.T
    
    # Calculate RMSD
    rmsd = np.sqrt(np.mean(np.sum((aligned_mobile - target_centered)**2, axis=1)))
    
    # Translate to target center
    aligned_mobile += np.mean(target, axis=0)
    
    return aligned_mobile, rmsd
