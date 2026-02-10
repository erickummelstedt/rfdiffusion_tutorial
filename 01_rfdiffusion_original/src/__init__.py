"""
RFDiffusion implementation package.

This package contains the core components for implementing RFDiffusion
from the Watson et al. (2023) Nature paper.
"""

__version__ = "0.1.0"

from .frames import RigidTransform, BackboneFrames
from .diffusion import SO3Diffusion, DiffusionModel
from .structure_module import StructureModule
from .sampling import sample_unconditional, sample_ddim

__all__ = [
    'RigidTransform',
    'BackboneFrames',
    'SO3Diffusion',
    'DiffusionModel',
    'StructureModule',
    'sample_unconditional',
    'sample_ddim',
]
