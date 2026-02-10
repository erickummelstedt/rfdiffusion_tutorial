"""
Rigid body transformations and frames for protein backbones.

This module implements SE(3) rigid transformations (rotations + translations)
and backbone frame representations used in RFDiffusion.

References:
    - RFDiffusion paper: Supplementary Methods
    - Official code: rfdiffusion/util.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class RigidTransform:
    """
    SE(3) rigid body transformation (rotation + translation).
    
    Represents a transformation T: R^3 -> R^3 defined by:
        T(x) = R @ x + t
    
    where R is a 3x3 rotation matrix and t is a 3D translation vector.
    
    Attributes:
        rotation: (*, 3, 3) rotation matrices
        translation: (*, 3) translation vectors
        
    Examples:
        >>> # Create identity transform
        >>> T = RigidTransform.identity()
        >>> 
        >>> # Apply to points
        >>> points = torch.randn(100, 3)
        >>> transformed = T.apply(points)
        >>> 
        >>> # Compose transforms
        >>> T1 = RigidTransform(R1, t1)
        >>> T2 = RigidTransform(R2, t2)
        >>> T_composed = T1.compose(T2)
    """
    
    def __init__(self, rotation: torch.Tensor, translation: torch.Tensor):
        """
        Initialize rigid transform.
        
        Args:
            rotation: (*, 3, 3) rotation matrices
            translation: (*, 3) translation vectors
        """
        self.rotation = rotation
        self.translation = translation
        
    @staticmethod
    def identity(batch_shape=()) -> 'RigidTransform':
        """
        Create identity transformation.
        
        Args:
            batch_shape: Shape of batch dimensions
            
        Returns:
            Identity RigidTransform
        """
        shape = batch_shape + (3, 3)
        rotation = torch.eye(3).expand(shape)
        translation = torch.zeros(batch_shape + (3,))
        return RigidTransform(rotation, translation)
    
    def compose(self, other: 'RigidTransform') -> 'RigidTransform':
        """
        Compose this transform with another: T_composed = self @ other.
        
        Mathematically: (R1, t1) @ (R2, t2) = (R1 @ R2, R1 @ t2 + t1)
        
        Args:
            other: Another RigidTransform
            
        Returns:
            Composed RigidTransform
        """
        new_rotation = self.rotation @ other.rotation
        new_translation = (self.rotation @ other.translation.unsqueeze(-1)).squeeze(-1) + self.translation
        return RigidTransform(new_rotation, new_translation)
    
    def inverse(self) -> 'RigidTransform':
        """
        Invert transformation.
        
        Mathematically: (R, t)^{-1} = (R^T, -R^T @ t)
        
        Returns:
            Inverted RigidTransform
        """
        inv_rotation = self.rotation.transpose(-2, -1)
        inv_translation = -(inv_rotation @ self.translation.unsqueeze(-1)).squeeze(-1)
        return RigidTransform(inv_rotation, inv_translation)
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to points.
        
        Args:
            points: (*, N, 3) points to transform
            
        Returns:
            Transformed points of same shape
        """
        # Expand rotation and translation for broadcasting
        rotated = (self.rotation @ points.unsqueeze(-1)).squeeze(-1)
        return rotated + self.translation
    
    @staticmethod
    def from_3_points(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> 'RigidTransform':
        """
        Construct frame from three points (e.g., N, CA, C atoms).
        
        Creates a right-handed coordinate system with:
        - Origin at p1
        - x-axis pointing toward p2
        - xy-plane containing p3
        
        Args:
            p1: (*, 3) first point (origin)
            p2: (*, 3) second point (x-axis direction)
            p3: (*, 3) third point (xy-plane)
            
        Returns:
            RigidTransform defining the frame
        """
        # TODO: Implement frame construction from three points
        # Hint: Use Gram-Schmidt orthogonalization
        raise NotImplementedError("To be implemented in notebook 03")
    
    def to_quaternion_translation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert to quaternion + translation representation.
        
        Returns:
            Tuple of (quaternions (*, 4), translations (*, 3))
        """
        # TODO: Convert rotation matrix to quaternion
        # Used for more compact representation and interpolation
        raise NotImplementedError("To be implemented")


class BackboneFrames:
    """
    Backbone frames for a protein structure.
    
    Represents a protein backbone as a series of rigid body frames,
    one per residue. Each frame is defined by the backbone atoms (N, CA, C).
    
    Attributes:
        frames: List of RigidTransform objects, one per residue
        n_residues: Number of residues
        
    Examples:
        >>> # Create from backbone coordinates
        >>> N = torch.randn(100, 3)  # Nitrogen atoms
        >>> CA = torch.randn(100, 3)  # Alpha carbons
        >>> C = torch.randn(100, 3)  # Carbonyl carbons
        >>> backbone = BackboneFrames.from_atoms(N, CA, C)
        >>> 
        >>> # Convert back to atoms
        >>> N_recon, CA_recon, C_recon = backbone.to_atoms()
    """
    
    def __init__(self, frames: RigidTransform):
        """
        Initialize from rigid transforms.
        
        Args:
            frames: RigidTransform with shape (n_residues, ...)
        """
        self.frames = frames
        self.n_residues = frames.rotation.shape[0]
    
    @staticmethod
    def from_atoms(N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor) -> 'BackboneFrames':
        """
        Construct backbone frames from backbone atoms.
        
        Each frame is centered at CA with:
        - Origin at CA
        - Local coordinate system defined by N, CA, C geometry
        
        Args:
            N: (n_residues, 3) nitrogen atom positions
            CA: (n_residues, 3) alpha carbon positions
            C: (n_residues, 3) carbonyl carbon positions
            
        Returns:
            BackboneFrames object
        """
        # TODO: Implement frame construction from backbone atoms
        # This is a key function for converting between representations
        raise NotImplementedError("To be implemented in notebook 03")
    
    def to_atoms(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert frames back to backbone atom coordinates.
        
        Returns:
            Tuple of (N, CA, C) coordinate tensors, each (n_residues, 3)
        """
        # TODO: Reconstruct atoms from frames
        # Should be inverse of from_atoms()
        raise NotImplementedError("To be implemented in notebook 03")
    
    def apply_transform(self, transform: RigidTransform) -> 'BackboneFrames':
        """
        Apply a global transformation to all frames.
        
        Useful for data augmentation (random rotation/translation).
        
        Args:
            transform: Global RigidTransform to apply
            
        Returns:
            Transformed BackboneFrames
        """
        new_frames = transform.compose(self.frames)
        return BackboneFrames(new_frames)
    
    def get_pairwise_distances(self) -> torch.Tensor:
        """
        Compute pairwise CA-CA distances.
        
        Returns:
            Distance matrix of shape (n_residues, n_residues)
        """
        # Extract CA positions (origins of frames)
        ca_coords = self.frames.translation
        diff = ca_coords.unsqueeze(0) - ca_coords.unsqueeze(1)
        distances = torch.sqrt((diff ** 2).sum(-1))
        return distances


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.
    
    Args:
        quaternions: (*, 4) quaternions [w, x, y, z]
        
    Returns:
        Rotation matrices of shape (*, 3, 3)
    """
    # TODO: Implement quaternion to rotation matrix conversion
    # Reference: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    raise NotImplementedError("To be implemented")


def matrix_to_quaternion(matrices: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions.
    
    Args:
        matrices: (*, 3, 3) rotation matrices
        
    Returns:
        Quaternions of shape (*, 4)
    """
    # TODO: Implement rotation matrix to quaternion conversion
    raise NotImplementedError("To be implemented")


def random_rotation(batch_shape=()) -> torch.Tensor:
    """
    Sample random rotation matrices uniformly from SO(3).
    
    Args:
        batch_shape: Shape of batch dimensions
        
    Returns:
        Random rotation matrices of shape (*batch_shape, 3, 3)
    """
    # TODO: Implement uniform sampling from SO(3)
    # Hint: Sample random quaternions and normalize
    raise NotImplementedError("To be implemented")
