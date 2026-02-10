# RFDiffusion Tutorial: Learning Protein Design from David Baker's Lab

A comprehensive, hands-on learning repository for understanding and recreating the groundbreaking work in protein design from David Baker's laboratory, focusing on RFDiffusion and related methods.

## 🎯 Purpose

This repository is designed as a **progressive learning resource** where each paper is treated as an individual module. Starting from the basics, you'll build your understanding of diffusion models for protein design, eventually mastering advanced concepts like all-atom design, biomolecular interactions, and multi-domain protein engineering.

## 📚 Learning Path

Work through the modules in order for the best learning experience:

### 1️⃣ [RFDiffusion Original](01_rfdiffusion_original/)
**Paper**: [De novo design of protein structure and function with RFdiffusion](https://www.nature.com/articles/s41586-023-06415-8) (Nature 2023)

**What you'll learn:**
- Fundamentals of diffusion models for protein structure generation
- Backbone-level protein design
- Conditional generation and motif scaffolding
- Basic protein structure representations

**Prerequisites**: Python, basic PyTorch, protein structure basics

### 2️⃣ [RFDiffusion All-Atom](02_rfdiffusion_allatom/)
**Paper**: [Generative design of de novo proteins based on secondary-structure constraints using an attention-based diffusion model](https://www.nature.com/articles/s41586-025-09248-9) (Nature 2024)

**What you'll learn:**
- All-atom structure generation
- Secondary structure constraints
- Attention mechanisms in protein design
- Side-chain prediction and optimization

**Prerequisites**: Complete Module 1

### 3️⃣ [Biomolecular Interactions](03_biomolecular_interactions/)
**Paper**: [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://www.science.org/doi/10.1126/science.adr8063) (Science 2024)

**What you'll learn:**
- Protein-protein interaction prediction
- Protein-ligand binding
- Multi-chain structure modeling
- Interface design principles

**Prerequisites**: Modules 1-2

### 4️⃣ [High-Affinity Protein Binders](04_protein_binders/)
**Paper**: [De novo design of high-affinity protein binders to bioactive helical peptides](https://dx.doi.org/10.1126/science.adu2454) (Science 2024)

**What you'll learn:**
- Binder design strategies
- Affinity optimization
- Peptide targeting
- Experimental validation approaches

**Prerequisites**: Modules 1-3

### 5️⃣ [Multi-Domain Proteins](05_multi_domain_proteins/)
**Paper**: [Automated design of multi-domain proteins via large language models and structure prediction](https://www.nature.com/articles/s41589-025-01929-w) (Nature Chemical Biology 2025)

**What you'll learn:**
- Multi-domain architecture design
- LLM integration for protein design
- Domain assembly strategies
- Complex protein engineering

**Prerequisites**: All previous modules

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/erickummelstedt/rfdiffusion_tutorial.git
cd rfdiffusion_tutorial
```

### 2. Set Up Environment

#### Option A: Conda (Recommended)
Best for scientific computing with PyTorch, CUDA, and complex dependencies:

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate rfdiffusion_tutorial

# Install the package in editable mode
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

#### Option B: Python venv
For a lighter setup using standard Python virtual environments:

```bash
# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate
# Or on Windows: .venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

**Note**: The `.venv` directory is already in `.gitignore`

### 3. Start Learning
```bash
# Navigate to the first module
cd 01_rfdiffusion_original

# Launch Jupyter
jupyter lab
# Or: jupyter notebook

# Open: notebooks/01_introduction.ipynb
```

## 📁 Repository Structure

```
rfdiffusion_tutorial/
├── 01_rfdiffusion_original/       # Original RFDiffusion paper
│   ├── README.md                  # Module overview and objectives
│   ├── notebooks/                 # Jupyter tutorials
│   ├── src/                       # Implementation code
│   ├── data/                      # Example datasets
│   ├── results/                   # Generated structures
│   └── tests/                     # Validation tests
├── 02_rfdiffusion_allatom/        # All-atom RFDiffusion
├── 03_biomolecular_interactions/  # Interaction prediction
├── 04_protein_binders/            # Binder design
├── 05_multi_domain_proteins/      # Multi-domain engineering
├── shared_utils/                  # Common utilities
│   ├── structure_utils.py         # PDB manipulation
│   ├── visualization.py           # Plotting and 3D viz
│   ├── metrics.py                 # Evaluation metrics
│   └── data_processing.py         # Data handling
├── docs/                          # Additional documentation
│   ├── protein_basics.md          # Protein structure primer
│   ├── diffusion_models.md        # Diffusion model theory
│   └── troubleshooting.md         # Common issues
├── environment.yml                # Conda environment
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
└── README.md                      # This file
```

## 🔧 Requirements

### Core Dependencies
- Python 3.9+
- PyTorch 2.0+
- NumPy, Pandas, Matplotlib
- BioPython
- PyMOL or py3Dmol for visualization
- Jupyter Notebook/Lab

### Optional
- CUDA-capable GPU (highly recommended)
- AlphaFold weights (for comparison)
- Rosetta (for energy calculations)

## 📖 Learning Resources

Each module includes:
- 📓 **Interactive Notebooks**: Step-by-step tutorials with explanations
- 💻 **Source Code**: Clean, documented implementations
- 📊 **Example Data**: Pre-processed datasets for immediate use
- ✅ **Tests**: Validation against published results
- 📝 **Exercises**: Practice problems to reinforce learning

## 🎓 Who Is This For?

This repository is ideal for:
- Graduate students studying computational biology
- Researchers new to protein design
- Machine learning practitioners interested in structural biology
- Anyone wanting to understand RFDiffusion from first principles

## 🤝 Contributing

Contributions welcome! Whether it's:
- Bug fixes or improvements
- Additional examples or tutorials
- Better documentation
- New exercise problems

Please open an issue or submit a pull request.

## 📄 License

This educational resource is provided for learning purposes. Please cite the original papers when using concepts or methods from this repository.

## 🙏 Acknowledgments

This repository is built to learn from the incredible work of:
- David Baker and the RosettaCommons team
- Authors of all papers included in this tutorial
- The broader protein design and machine learning communities

## 📧 Questions?

Open an issue or check the [troubleshooting guide](docs/troubleshooting.md).

---

**Start your protein design journey today!** Begin with [Module 1: RFDiffusion Original](01_rfdiffusion_original/)
