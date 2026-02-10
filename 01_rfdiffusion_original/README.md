# Module 1: RFDiffusion Original

## 📄 Paper
**De novo design of protein structure and function with RFdiffusion**  
Watson et al., Nature (2023)  
DOI: [10.1038/s41586-023-06415-8](https://www.nature.com/articles/s41586-023-06415-8)

## 🎯 Learning Objectives

By completing this module, you will understand:

1. **Diffusion Models Fundamentals**
   - Forward and reverse diffusion processes
   - Score-based generative modeling
   - Training and sampling procedures

2. **Protein Structure Representation**
   - Backbone-only representations (N, CA, C atoms)
   - Rotation and translation frames
   - Internal coordinates and rigid body transformations

3. **Conditional Generation**
   - Motif scaffolding
   - Symmetric design
   - Binder design basics

4. **Architecture**
   - SE(3)-equivariant networks
   - Structure module design
   - Noise schedules and sampling strategies

## 📚 Prerequisites

- **Python**: Intermediate level
- **PyTorch**: Basic familiarity with tensors, modules, training loops
- **Linear Algebra**: Rotations, transformations, 3D geometry
- **Protein Structure**: Basic understanding of protein backbone, PDB format
- **Probability**: Understanding of probability distributions, sampling

## 📂 Module Structure

```
01_rfdiffusion_original/
├── notebooks/
│   ├── 01_introduction.ipynb           # Overview and setup
│   ├── 02_diffusion_basics.ipynb       # Diffusion model theory
│   ├── 03_protein_representation.ipynb # Structure encoding
│   ├── 04_se3_equivariance.ipynb       # Geometric deep learning
│   ├── 05_unconditional_generation.ipynb
│   ├── 06_motif_scaffolding.ipynb
│   ├── 07_symmetric_design.ipynb
│   └── 08_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── model.py                        # RFDiffusion architecture
│   ├── diffusion.py                    # Diffusion process
│   ├── se3_transformer.py              # SE(3) layers
│   ├── frames.py                       # Rigid body transformations
│   ├── sampling.py                     # Generation procedures
│   └── losses.py                       # Training objectives
├── data/
│   ├── examples/                       # Example PDB files
│   └── README.md                       # Data sources
├── results/
│   └── .gitkeep
├── tests/
│   ├── test_model.py
│   ├── test_diffusion.py
│   └── test_frames.py
└── README.md                           # This file
```

## 🚀 Getting Started

### 1. Activate Environment
```bash
conda activate rfdiffusion_tutorial
```

### 2. Download Example Data
```bash
cd data
# Instructions in data/README.md
```

### 3. Start with First Notebook
```bash
jupyter notebook notebooks/01_introduction.ipynb
```

## 📖 Recommended Reading Order

1. **01_introduction.ipynb** - Get oriented with the problem and approach
2. **02_diffusion_basics.ipynb** - Learn diffusion model theory
3. **03_protein_representation.ipynb** - Understand how proteins are encoded
4. **04_se3_equivariance.ipynb** - Geometric constraints for proteins
5. **05_unconditional_generation.ipynb** - Generate basic protein backbones
6. **06_motif_scaffolding.ipynb** - Design proteins around functional motifs
7. **07_symmetric_design.ipynb** - Create symmetric assemblies
8. **08_evaluation.ipynb** - Assess quality of designs

## 🔑 Key Concepts

### Diffusion Process
The model learns to denoise protein structures by training on a forward process that gradually adds noise, then learning to reverse this process.

### SE(3) Equivariance
The network respects the symmetries of 3D space - rotating/translating inputs produces rotated/translated outputs.

### Conditional Generation
Generate proteins with specific properties by conditioning the diffusion process on constraints like motifs or symmetry.

## 💡 Exercise Problems

Each notebook includes practice problems. Key exercises:
- Implement a simple 1D diffusion model
- Build rotation-equivariant layers
- Design a motif scaffolding task
- Evaluate generated structures

## 📊 Expected Outcomes

By the end of this module, you should be able to:
- ✅ Explain how diffusion models work for protein design
- ✅ Implement basic components of RFDiffusion
- ✅ Generate novel protein backbones
- ✅ Design proteins with specific structural motifs
- ✅ Evaluate quality of generated structures

## 🔗 Resources

- Original Paper: [Nature link](https://www.nature.com/articles/s41586-023-06415-8)
- RosettaFold: Background on structure prediction
- Score-based generative models: [Yang Song's blog](https://yang-song.net/blog/2021/score/)
- SE(3) Transformers: [Fuchs et al. 2020](https://arxiv.org/abs/2006.10503)

## ⏭️ Next Module

Once you've completed this module, move on to:  
**[Module 2: RFDiffusion All-Atom](../02_rfdiffusion_allatom/)** - Learn all-atom structure generation

## 🐛 Troubleshooting

Common issues and solutions are in [docs/troubleshooting.md](../docs/troubleshooting.md)

## ❓ Questions?

Open an issue with the tag `module-1` for help with this module.
