# Quick Start Guide

Get up and running with the RFDiffusion Tutorial in minutes!

## Prerequisites

- Python 3.9 or higher
- Git
- (Optional but recommended) NVIDIA GPU with CUDA support

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd rfdiffusion_tutorial

# Create and activate environment
conda env create -f environment.yml
conda activate rfdiffusion_tutorial

# Install the package
pip install -e .
```

### Option 2: Pip

```bash
# Clone the repository
git clone <your-repo-url>
cd rfdiffusion_tutorial

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Verify Installation

```python
import torch
import numpy as np
from Bio import PDB
from shared_utils import structure_utils

print("✓ All dependencies installed!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Your First Steps

### 1. Explore the Repository

```bash
# Navigate to first module
cd 01_rfdiffusion_original

# List notebooks
ls notebooks/
```

### 2. Start Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

### 3. Open First Notebook

Navigate to: `01_rfdiffusion_original/notebooks/01_introduction.ipynb`

### 4. Follow Along

Work through the notebooks in order:
1. Introduction and setup
2. Diffusion model basics
3. Protein representation
4. And so on...

## Quick Example

Here's a taste of what you'll learn:

```python
from shared_utils import structure_utils, visualization
import matplotlib.pyplot as plt

# Load a protein structure
structure = structure_utils.load_pdb("data/examples/protein.pdb")

# Get backbone coordinates
coords = structure_utils.get_backbone_coords(structure)

# Visualize
fig = visualization.plot_structure_3d(coords, title="My First Protein")
plt.show()

# Calculate properties
from shared_utils.metrics import calculate_structure_quality_score
quality = calculate_structure_quality_score(coords)
print(f"Structure quality: {quality}")
```

## Learning Path

### Week 1: Foundations
- **Day 1-2**: Protein structure basics ([docs/protein_basics.md](docs/protein_basics.md))
- **Day 3-4**: Diffusion model theory ([docs/diffusion_models.md](docs/diffusion_models.md))
- **Day 5-7**: Module 1 - RFDiffusion Original (notebooks 1-3)

### Week 2: Core Implementation
- **Day 8-10**: Module 1 (notebooks 4-6)
- **Day 11-12**: Module 1 (notebooks 7-8)
- **Day 13-14**: Practice and exercises

### Week 3: Advanced Topics
- **Day 15-17**: Module 2 - All-Atom RFDiffusion
- **Day 18-21**: Module 3 - Biomolecular Interactions

### Week 4+: Specialization
- Module 4: Protein Binders
- Module 5: Multi-Domain Proteins
- Your own projects!

## Common Commands

### Run Tests
```bash
pytest tests/
```

### Format Code
```bash
black shared_utils/
```

### Check Style
```bash
flake8 shared_utils/
```

### Update Dependencies
```bash
conda env update -f environment.yml
# or
pip install -U -r requirements.txt
```

## Troubleshooting

Having issues? Check [docs/troubleshooting.md](docs/troubleshooting.md)

Common quick fixes:

**Can't import shared_utils?**
```bash
pip install -e .
```

**Jupyter kernel dies?**
```bash
# Reduce batch size or use fewer data
```

**CUDA out of memory?**
```python
torch.cuda.empty_cache()
# or reduce batch_size
```

## Getting Help

1. Check [troubleshooting guide](docs/troubleshooting.md)
2. Search existing [issues](https://github.com/yourusername/rfdiffusion_tutorial/issues)
3. Ask in [discussions](https://github.com/yourusername/rfdiffusion_tutorial/discussions)
4. Create new issue with details

## Resources

### In This Repo
- [README.md](README.md) - Main overview
- [docs/protein_basics.md](docs/protein_basics.md) - Protein structure primer
- [docs/diffusion_models.md](docs/diffusion_models.md) - Diffusion model theory
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute

### External Resources
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [BioPython Tutorial](http://biopython.org/DIST/docs/tutorial/Tutorial.html)
- [PDB File Format](https://www.wwpdb.org/documentation/file-format)
- [Rosetta Commons](https://www.rosettacommons.org/)

## What's Next?

After installation:

1. ✅ Read [protein_basics.md](docs/protein_basics.md) if you're new to protein structure
2. ✅ Read [diffusion_models.md](docs/diffusion_models.md) for diffusion model background
3. ✅ Start [Module 1](01_rfdiffusion_original/) - Begin your protein design journey!

## Tips for Success

🎯 **Focus**: Work through modules in order
  
📓 **Practice**: Run all code cells, experiment with parameters

🤔 **Think**: Try exercises before looking at solutions

👥 **Collaborate**: Discuss with others learning the same material

🔄 **Iterate**: Revisit earlier modules as you learn more

📝 **Document**: Keep notes on insights and questions

## Community

Join the protein design community:
- Share your designs
- Ask questions
- Help others learn
- Contribute improvements

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

**Ready to start?** Head to [Module 1: RFDiffusion Original](01_rfdiffusion_original/) 🚀
