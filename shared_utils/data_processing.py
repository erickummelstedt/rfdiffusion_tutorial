"""
Data processing utilities for protein datasets.

This module provides functions for loading, preprocessing, and batching
protein structure data for training and inference.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class ProteinDataset(Dataset):
    """
    PyTorch Dataset for protein structures.
    
    Args:
        pdb_files: List of PDB file paths
        transform: Optional transform to apply to each sample
        max_length: Maximum sequence length (for padding)
    """
    
    def __init__(self, pdb_files: List[str], 
                 transform=None, max_length: Optional[int] = None):
        self.pdb_files = pdb_files
        self.transform = transform
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.pdb_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from .structure_utils import load_pdb, get_backbone_coords
        
        pdb_file = self.pdb_files[idx]
        structure = load_pdb(pdb_file)
        coords = get_backbone_coords(structure)
        
        # Convert to tensor
        coords_tensor = torch.from_numpy(coords).float()
        
        # Pad if needed
        if self.max_length and len(coords_tensor) < self.max_length:
            padding = torch.zeros(self.max_length - len(coords_tensor), *coords_tensor.shape[1:])
            coords_tensor = torch.cat([coords_tensor, padding], dim=0)
            mask = torch.cat([
                torch.ones(len(coords)),
                torch.zeros(self.max_length - len(coords))
            ])
        else:
            mask = torch.ones(len(coords_tensor))
        
        sample = {
            'coords': coords_tensor,
            'mask': mask,
            'pdb_file': pdb_file
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def collate_protein_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching protein structures.
    
    Args:
        batch: List of samples from ProteinDataset
        
    Returns:
        Batched dictionary with padded tensors
    """
    # Find max length in batch
    max_len = max(sample['coords'].shape[0] for sample in batch)
    
    # Pad all samples to max length
    coords_list = []
    mask_list = []
    
    for sample in batch:
        coords = sample['coords']
        mask = sample['mask']
        
        if len(coords) < max_len:
            pad_len = max_len - len(coords)
            coords = torch.cat([coords, torch.zeros(pad_len, *coords.shape[1:])])
            mask = torch.cat([mask, torch.zeros(pad_len)])
        
        coords_list.append(coords)
        mask_list.append(mask)
    
    return {
        'coords': torch.stack(coords_list),
        'mask': torch.stack(mask_list),
        'pdb_files': [s['pdb_file'] for s in batch]
    }


def create_dataloader(pdb_files: List[str], batch_size: int = 8,
                     shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """
    Create a DataLoader for protein structures.
    
    Args:
        pdb_files: List of PDB file paths
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        
    Returns:
        PyTorch DataLoader
    """
    dataset = ProteinDataset(pdb_files)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_protein_batch
    )
    return dataloader


def normalize_coords(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize coordinates to zero mean and unit scale.
    
    Args:
        coords: Coordinate array
        
    Returns:
        Tuple of (normalized_coords, center, scale)
    """
    center = np.mean(coords.reshape(-1, 3), axis=0)
    centered = coords - center
    scale = np.std(centered)
    normalized = centered / scale
    
    return normalized, center, scale


def denormalize_coords(normalized_coords: np.ndarray, 
                      center: np.ndarray, scale: float) -> np.ndarray:
    """
    Denormalize coordinates back to original scale.
    
    Args:
        normalized_coords: Normalized coordinate array
        center: Original center
        scale: Original scale
        
    Returns:
        Denormalized coordinates
    """
    return normalized_coords * scale + center


def add_noise_to_coords(coords: np.ndarray, noise_level: float = 1.0) -> np.ndarray:
    """
    Add Gaussian noise to coordinates.
    
    Args:
        coords: Coordinate array
        noise_level: Standard deviation of noise
        
    Returns:
        Noised coordinates
    """
    noise = np.random.normal(0, noise_level, coords.shape)
    return coords + noise


def augment_structure(coords: np.ndarray) -> np.ndarray:
    """
    Apply random rotation and translation for data augmentation.
    
    Args:
        coords: Coordinate array (n, 3) or (n, 3, 3)
        
    Returns:
        Augmented coordinates
    """
    original_shape = coords.shape
    coords_flat = coords.reshape(-1, 3)
    
    # Random rotation
    angles = np.random.uniform(0, 2*np.pi, 3)
    
    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])
    
    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    
    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx
    
    # Apply rotation
    rotated = coords_flat @ R.T
    
    # Random translation
    translation = np.random.uniform(-5, 5, 3)
    augmented = rotated + translation
    
    return augmented.reshape(original_shape)


def split_dataset(pdb_files: List[str], 
                 train_frac: float = 0.8,
                 val_frac: float = 0.1,
                 seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        pdb_files: List of PDB file paths
        train_frac: Fraction for training
        val_frac: Fraction for validation
        seed: Random seed
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    np.random.seed(seed)
    n = len(pdb_files)
    
    # Shuffle
    indices = np.random.permutation(n)
    
    # Split points
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_files = [pdb_files[i] for i in train_idx]
    val_files = [pdb_files[i] for i in val_idx]
    test_files = [pdb_files[i] for i in test_idx]
    
    logger.info(f"Split dataset: {len(train_files)} train, "
                f"{len(val_files)} val, {len(test_files)} test")
    
    return train_files, val_files, test_files
