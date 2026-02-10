# RFDiffusion Implementation Guide

## Step-by-Step Guide to Recreating the Paper Findings

This guide walks you through implementing RFDiffusion from scratch, based on:

**Paper**: "De novo design of protein structure and function with RFdiffusion"  
**Authors**: Joseph L. Watson et al.  
**Published**: Nature (2023)  
**DOI**: [10.1038/s41586-023-06415-8](https://www.nature.com/articles/s41586-023-06415-8)

---

## 📚 Official Resources

### Original Implementation
- **GitHub**: [RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
- **Paper Supplementary**: [Nature Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06415-8/MediaObjects/41586_2023_6415_MOESM1_ESM.pdf)
- **Weights**: Available through official repository

### Related Repositories
- **RosettaFold**: [RosettaCommons/RosettaFold](https://github.com/RosettaCommons/RosettaFold) - Structure prediction backbone
- **SE(3)-Transformer**: [FabianFuchsML/se3-transformer-public](https://github.com/FabianFuchsML/se3-transformer-public) - Equivariant layers

---

## 🎯 Implementation Roadmap

### Phase 1: Foundations (Weeks 1-2)
Understanding the theoretical and practical foundations.

### Phase 2: Core Components (Weeks 3-4)
Implementing the main architectural components.

### Phase 3: Training Pipeline (Weeks 5-6)
Setting up training infrastructure and loss functions.

### Phase 4: Sampling & Generation (Week 7)
Implementing the diffusion sampling process.

### Phase 5: Conditional Design (Week 8+)
Adding motif scaffolding, symmetry, and binder design.

---

## 📋 Detailed Implementation Steps

## Step 1: Environment Setup

### 1.1 Clone Official Repository (Reference)
```bash
# Clone for reference (don't copy directly)
cd ~/projects
git clone https://github.com/RosettaCommons/RFdiffusion.git
cd RFdiffusion

# Study the structure but implement from scratch for learning
```

### 1.2 Set Up Our Tutorial Environment
```bash
cd /Users/ekummelstedt/le_code_base/rfdiffusion_tutorial
conda activate rfdiffusion_tutorial

# Install additional dependencies
pip install dgl  # Deep Graph Library for SE(3) operations
pip install hydra-core  # Configuration management
pip install wandb  # Optional: experiment tracking
```

### 1.3 Download Pretrained Weights (Optional)
```bash
cd 01_rfdiffusion_original/data
# Follow instructions in official repo to download weights
# We'll train from scratch for learning, but weights useful for validation
```

---

## Step 2: Understanding the Architecture

### 2.1 Read Key Paper Sections
- **Main Text**: Focus on Methods section
- **Extended Data**: Figures showing architecture details
- **Supplementary**: Detailed hyperparameters and training procedures

### 2.2 Study Official Implementation
Key files to understand:
```
RFdiffusion/
├── rfdiffusion/
│   ├── diffusion.py          # Core diffusion logic
│   ├── inference.py          # Sampling/generation
│   ├── training.py           # Training loop
│   └── util.py               # Utilities
├── models/
│   ├── RoseTTAFold_module.py # Structure module
│   └── Track_module.py       # Attention layers
└── config/
    └── inference/            # Configuration files
```

### 2.3 Key Concepts to Master
- [ ] SE(3)-equivariant neural networks
- [ ] Denoising diffusion probabilistic models (DDPM)
- [ ] Rigid body transformations
- [ ] Rotation representations (quaternions, rotation matrices)
- [ ] Backbone frames and internal coordinates

---

## Step 3: Implement Data Structures

### 3.1 Create Rigid Body Frame Representation
**File**: `src/frames.py`

Implement:
```python
class RigidTransform:
    """SE(3) rigid body transformation (rotation + translation)"""
    def __init__(self, rotation, translation):
        pass
    
    def compose(self, other):
        """Compose two transformations"""
        pass
    
    def inverse(self):
        """Invert transformation"""
        pass
    
    def apply(self, points):
        """Apply transformation to points"""
        pass
```

**References**:
- Paper: Supplementary Methods - "Backbone representation"
- Official: `rfdiffusion/util.py` - `Rigid` class

### 3.2 Create Backbone Representation
**File**: `src/backbone.py`

Implement:
```python
class BackboneFrames:
    """Backbone frames for each residue (N, CA, C atoms)"""
    def __init__(self, N, CA, C):
        self.frames = self.compute_frames(N, CA, C)
    
    def compute_frames(self, N, CA, C):
        """Compute rigid body frames from backbone atoms"""
        pass
    
    def to_atoms(self):
        """Convert frames back to atomic coordinates"""
        pass
```

**What to implement**:
- Frame construction from backbone atoms
- Conversion between Cartesian and frame representations
- Validation functions

---

## Step 4: Implement SE(3)-Equivariant Layers

### 4.1 Understand SE(3) Equivariance
**Concept**: Network respects 3D symmetries
- Rotating input → rotated output
- Translating input → translated output

### 4.2 Implement Invariant Point Attention (IPA)
**File**: `src/ipa.py`

Based on AlphaFold2's IPA layer:
```python
class InvariantPointAttention(nn.Module):
    """SE(3)-equivariant attention mechanism"""
    def __init__(self, d_model, n_heads, n_query_points, n_value_points):
        pass
    
    def forward(self, features, frames, mask):
        """
        Args:
            features: (B, N, d_model) - per-residue features
            frames: (B, N, 7) - rigid body frames [rot(4), trans(3)]
            mask: (B, N) - valid residue mask
        """
        pass
```

**References**:
- AlphaFold2 paper: Supplementary for IPA details
- Official RFDiffusion: Uses similar structure module from RosettaFold

### 4.3 Implement Structure Module
**File**: `src/structure_module.py`

```python
class StructureModule(nn.Module):
    """Main structure prediction/generation module"""
    def __init__(self, n_layers=8, d_model=384):
        super().__init__()
        self.layers = nn.ModuleList([
            StructureLayer(d_model) for _ in range(n_layers)
        ])
    
    def forward(self, seq_features, frames, mask):
        """Process through multiple layers of IPA + FFN"""
        pass
```

---

## Step 5: Implement Diffusion Process

### 5.1 Forward Diffusion (Adding Noise)
**File**: `src/diffusion.py`

```python
class SO3Diffusion:
    """Diffusion on SO(3) rotation group"""
    def __init__(self, num_timesteps=1000):
        self.T = num_timesteps
        self.setup_noise_schedule()
    
    def setup_noise_schedule(self):
        """Define beta schedule for noise levels"""
        # Cosine schedule from improved DDPM
        pass
    
    def add_noise(self, frames, t):
        """Add noise to frames at timestep t"""
        # Rotation: noise on SO(3)
        # Translation: Gaussian noise on R^3
        pass
```

### 5.2 Reverse Diffusion (Denoising)
```python
class DiffusionModel(nn.Module):
    """Complete diffusion model"""
    def __init__(self, structure_module, diffusion):
        super().__init__()
        self.structure_module = structure_module
        self.diffusion = diffusion
    
    def forward(self, noisy_frames, t, features, mask):
        """Predict denoised frames"""
        pass
    
    def p_sample_step(self, xt, t, features, mask):
        """Single denoising step"""
        pass
```

**Key Implementation Details**:
- Noise schedule: Use cosine schedule (better than linear)
- SO(3) noise: Sample rotations from IGSO(3) distribution
- Translation noise: Standard Gaussian
- Timestep embedding: Sinusoidal encoding

**References**:
- Paper: "Diffusion process" section in Methods
- Official: `rfdiffusion/diffusion.py`

---

## Step 6: Implement Training

### 6.1 Create Dataset
**File**: `src/dataset.py`

```python
class ProteinStructureDataset(Dataset):
    """Dataset of protein structures for training"""
    def __init__(self, pdb_dir, max_length=128):
        self.structures = self.load_structures(pdb_dir)
    
    def __getitem__(self, idx):
        """Return structure with random augmentation"""
        structure = self.structures[idx]
        # Random rotation/translation
        structure = self.augment(structure)
        return {
            'frames': structure.frames,
            'sequence': structure.sequence,
            'mask': structure.mask
        }
```

### 6.2 Training Loop
**File**: `src/training.py`

```python
def train_diffusion(model, dataloader, optimizer, num_epochs):
    """Training loop for diffusion model"""
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Sample random timestep
            t = torch.randint(0, model.diffusion.T, (batch_size,))
            
            # Add noise
            noisy_frames = model.diffusion.add_noise(batch['frames'], t)
            
            # Predict noise
            pred_noise = model(noisy_frames, t, batch['sequence'], batch['mask'])
            
            # Loss: MSE on noise prediction
            loss = F.mse_loss(pred_noise, true_noise)
            
            # Backprop
            loss.backward()
            optimizer.step()
```

### 6.3 Loss Functions
**File**: `src/losses.py`

Implement:
- **Frame-aligned point error (FAPE)**: Main loss from AlphaFold2
- **Rotation loss**: Geodesic distance on SO(3)
- **Translation loss**: L2 distance
- **Auxiliary losses**: Pairwise distances, angles

**References**:
- Paper: "Training" section
- AlphaFold2: FAPE loss definition

---

## Step 7: Implement Sampling/Generation

### 7.1 Unconditional Generation
**File**: `src/sampling.py`

```python
def sample_unconditional(model, length, num_samples=1, num_steps=200):
    """Generate protein backbones from random noise"""
    # Start from random noise
    frames_T = model.diffusion.sample_noise(num_samples, length)
    
    # Iteratively denoise
    frames = frames_T
    for t in reversed(range(num_steps)):
        frames = model.p_sample_step(frames, t)
    
    return frames
```

### 7.2 DDIM Sampling (Faster)
```python
def sample_ddim(model, length, num_steps=50):
    """Faster sampling using DDIM"""
    # Skip timesteps for faster generation
    pass
```

---

## Step 8: Implement Conditional Generation

### 8.1 Motif Scaffolding
**File**: `src/conditional.py`

```python
def sample_with_motif(model, motif_frames, motif_positions, length):
    """Generate structure around fixed motif"""
    # Fix motif positions during diffusion
    # Generate rest of structure
    pass
```

**Key Features to Implement**:
- Fix motif residues during forward/reverse diffusion
- Inpainting approach
- Guidance on maintaining motif geometry

### 8.2 Symmetric Design
```python
def sample_symmetric(model, length, symmetry='C3'):
    """Generate symmetric oligomer"""
    # Apply symmetry constraints
    # Generate asymmetric unit, replicate with symmetry
    pass
```

**Symmetries to support**:
- Cyclic (C2, C3, C4, etc.)
- Dihedral (D2, D3, D4, etc.)
- Tetrahedral, Octahedral (advanced)

### 8.3 Binder Design
```python
def sample_binder(model, target_structure, length):
    """Design protein that binds to target"""
    # Conditional on target interface
    # Optimize interface contacts
    pass
```

---

## Step 9: Validation & Evaluation

### 9.1 Structure Quality Metrics
**File**: `src/evaluation.py`

Implement metrics:
```python
def evaluate_structure(generated_frames):
    """Compute quality metrics"""
    metrics = {
        'designability': compute_designability(generated_frames),
        'novelty': compute_novelty(generated_frames),
        'diversity': compute_diversity(generated_frames),
        'validity': check_validity(generated_frames),
    }
    return metrics
```

**Metrics to implement**:
- **Designability**: Can sequence fold to structure? (Use AlphaFold2)
- **Novelty**: Distance to nearest training example
- **Diversity**: Pairwise RMSD within generated set
- **Validity**: Clash scores, geometry checks

### 9.2 Compare to Paper Results
Reproduce key results from paper:
- [ ] Table 1: Unconditional generation success rates
- [ ] Figure 2: Motif scaffolding examples
- [ ] Figure 3: Symmetric oligomers
- [ ] Figure 4: Binder design cases

---

## Step 10: Reproduce Paper Experiments

### Experiment 1: Unconditional Generation
**Notebook**: `notebooks/05_unconditional_generation.ipynb`

```python
# Generate 1000 backbones of length 100
backbones = []
for i in range(1000):
    backbone = sample_unconditional(model, length=100)
    backbones.append(backbone)

# Evaluate with ProteinMPNN + AlphaFold2
sequences = design_sequences(backbones)  # ProteinMPNN
predictions = predict_structures(sequences)  # AlphaFold2

# Compute success rate (scRMSD < 2.0 Å)
success_rate = compute_success_rate(backbones, predictions)
print(f"Success rate: {success_rate:.1%}")  # Target: >50%
```

### Experiment 2: Motif Scaffolding
**Notebook**: `notebooks/06_motif_scaffolding.ipynb`

Reproduce paper examples:
- RSV F protein epitope scaffolding
- Metalloprotein active site design
- Enzyme active site grafting

### Experiment 3: Symmetric Design
**Notebook**: `notebooks/07_symmetric_design.ipynb`

Generate symmetric oligomers:
- C2 through C8 symmetry
- D2 through D4 symmetry
- Validate experimental structures match

### Experiment 4: Binder Design
Design binders to targets from paper:
- SARS-CoV-2 spike protein
- PD-L1
- Other therapeutic targets

---

## 📊 Expected Results

### Success Metrics (from Paper)
- **Unconditional**: >50% designability (scRMSD < 2.0 Å)
- **Motif scaffolding**: >40% success with correct motif RMSD < 1.5 Å
- **Symmetric design**: >60% success maintaining symmetry
- **Binder design**: KD values in nM-μM range (experimental validation)

### Computational Requirements
- **Training**: 8x V100 GPUs, ~1 week
- **Inference**: 1x V100 GPU, ~1-10 minutes per design
- **Dataset**: ~10,000-50,000 protein structures from PDB

---

## 🔧 Debugging Tips

### Common Issues

**Issue 1: NaN losses during training**
- Check gradient clipping
- Reduce learning rate
- Verify noise schedule implementation
- Check for invalid rotations (normalize quaternions)

**Issue 2: Generated structures have clashes**
- Increase number of diffusion steps
- Adjust noise schedule
- Add repulsive loss term
- Post-process with relaxation

**Issue 3: Low designability**
- Ensure SE(3) equivariance is correct
- Check frame reconstruction accuracy
- Verify loss function implementation
- Train longer or with more data

**Issue 4: Motif not preserved**
- Increase conditioning strength
- Fix motif earlier in reverse diffusion
- Add explicit motif loss term

---

## 📚 Additional Resources

### Papers to Read
1. **RFDiffusion**: Main paper (Watson et al. 2023)
2. **AlphaFold2**: For structure module architecture (Jumper et al. 2021)
3. **DDPM**: Diffusion models basics (Ho et al. 2020)
4. **Score-based models**: Alternative perspective (Song et al. 2021)
5. **SE(3)-Transformers**: Equivariant networks (Fuchs et al. 2020)

### Code References
- Official RFDiffusion implementation
- AlphaFold2 source code (DeepMind)
- OpenFold (open-source AlphaFold2)
- ProteinMPNN (sequence design)
- ColabFold (accessible structure prediction)

### Tutorials
- Yang Song's blog on diffusion models
- Lil'Log diffusion tutorial
- AlphaFold2 architecture explained
- SE(3) equivariance tutorial

---

## ✅ Checklist

### Core Implementation
- [ ] Rigid body frames (`src/frames.py`)
- [ ] SE(3)-equivariant layers (`src/ipa.py`)
- [ ] Structure module (`src/structure_module.py`)
- [ ] SO(3) diffusion (`src/diffusion.py`)
- [ ] Training loop (`src/training.py`)
- [ ] Sampling functions (`src/sampling.py`)

### Conditional Generation
- [ ] Motif scaffolding
- [ ] Symmetric design
- [ ] Binder design

### Validation
- [ ] Quality metrics
- [ ] Designability pipeline (ProteinMPNN + AF2)
- [ ] Comparison to paper results

### Notebooks
- [ ] 01: Introduction
- [ ] 02: Diffusion basics
- [ ] 03: Protein representation
- [ ] 04: SE(3) equivariance
- [ ] 05: Unconditional generation
- [ ] 06: Motif scaffolding
- [ ] 07: Symmetric design
- [ ] 08: Evaluation

---

## 🎯 Timeline

**Estimated total time**: 8-12 weeks for complete implementation

- **Weeks 1-2**: Foundation and theory
- **Weeks 3-4**: Core architecture
- **Weeks 5-6**: Training pipeline
- **Week 7**: Sampling and basic generation
- **Week 8**: Conditional generation
- **Weeks 9-10**: Validation and reproduction
- **Weeks 11-12**: Refinement and documentation

---

## 📞 Getting Help

- **Official Issues**: [RFdiffusion GitHub Issues](https://github.com/RosettaCommons/RFdiffusion/issues)
- **Paper Questions**: Contact authors (see paper)
- **Tutorial Issues**: Open issue in this repository
- **Community**: RosettaCommons forums, r/bioinformatics

---

**Ready to start?** Begin with [notebooks/01_introduction.ipynb](notebooks/01_introduction.ipynb) 🚀
