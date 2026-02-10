# Troubleshooting Guide

Common issues and solutions for the RFDiffusion Tutorial.

## Installation Issues

### CUDA/PyTorch Issues

**Problem**: PyTorch not detecting GPU
```
RuntimeError: CUDA not available
```

**Solutions**:
1. Check CUDA installation:
```bash
nvidia-smi
nvcc --version
```

2. Install correct PyTorch version:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. Verify PyTorch installation:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

### BioPython Issues

**Problem**: BioPython PDB parser warnings
```
PDBConstructionWarning: WARNING: Chain A is discontinuous
```

**Solution**: These warnings are usually harmless. To suppress:
```python
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
```

### py3Dmol Visualization

**Problem**: Interactive visualization not showing in Jupyter
```
py3Dmol view not rendering
```

**Solutions**:
1. Enable widget support:
```bash
jupyter nbextension enable --py widgetsnbextension
```

2. Use alternative:
```python
# If py3Dmol doesn't work, use PyMOL or static matplotlib
from shared_utils.visualization import plot_structure_3d
fig = plot_structure_3d(coords)
```

## Data Issues

### Missing PDB Files

**Problem**: Example data not found
```
FileNotFoundError: data/examples/protein.pdb
```

**Solution**: Download example structures:
```bash
cd 01_rfdiffusion_original/data
# Download from RCSB PDB
wget https://files.rcsb.org/download/1UBQ.pdb -O examples/ubiquitin.pdb
```

### Memory Issues

**Problem**: Out of memory during data loading
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size:
```python
dataloader = create_dataloader(files, batch_size=4)  # Instead of 32
```

2. Use CPU for data loading:
```python
dataloader = DataLoader(dataset, num_workers=0)  # Reduce workers
```

3. Clear cache:
```python
torch.cuda.empty_cache()
```

## Model Issues

### NaN Losses

**Problem**: Loss becomes NaN during training
```
Loss: nan
```

**Solutions**:
1. Check learning rate (may be too high):
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower LR
```

2. Add gradient clipping:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

3. Check input normalization:
```python
coords = normalize_coords(coords)  # Normalize data
```

### Slow Training

**Problem**: Training very slow

**Solutions**:
1. Profile code:
```python
import torch.autograd.profiler as profiler

with profiler.profile(use_cuda=True) as prof:
    output = model(input)
    
print(prof.key_averages().table())
```

2. Use mixed precision:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

3. Optimize data loading:
```python
dataloader = DataLoader(
    dataset, 
    batch_size=32,
    num_workers=4,  # Parallel loading
    pin_memory=True,  # Faster CPU->GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

## Structure Generation Issues

### Invalid Structures

**Problem**: Generated structures have clashes or bad geometry

**Solutions**:
1. Check clash score:
```python
from shared_utils.metrics import calculate_clash_score
clashes = calculate_clash_score(coords)
if clashes > 10:
    print("Warning: Many clashes detected")
```

2. Adjust sampling parameters:
```python
# Use more diffusion steps
structure = sample(model, num_steps=200)  # Instead of 50

# Adjust temperature
structure = sample(model, temperature=0.8)  # Lower = less noise
```

3. Post-process with relaxation (if Rosetta available):
```bash
# Energy minimization
rosetta_scripts -s generated.pdb -parser:protocol relax.xml
```

### Poor Design Success

**Problem**: Designed sequences don't fold to target structure

**Solutions**:
1. Validate with structure prediction:
```python
# Use AlphaFold or ESMFold to predict from sequence
predicted_coords = predict_structure(sequence)
rmsd = calculate_rmsd(designed_coords, predicted_coords)
print(f"RMSD: {rmsd:.2f} Å")  # Should be < 2.0
```

2. Check designability metrics:
```python
from shared_utils.metrics import calculate_structure_quality_score
quality = calculate_structure_quality_score(coords, phi, psi)
print(quality)
```

## Notebook Issues

### Kernel Dies

**Problem**: Jupyter kernel crashes

**Solutions**:
1. Check memory usage:
```python
import psutil
print(f"RAM: {psutil.virtual_memory().percent}%")
```

2. Restart kernel and clear outputs:
```
Kernel → Restart & Clear Output
```

3. Run cells separately instead of "Run All"

### Import Errors

**Problem**: Cannot import shared_utils
```
ModuleNotFoundError: No module named 'shared_utils'
```

**Solutions**:
1. Install package in development mode:
```bash
cd /path/to/rfdiffusion_tutorial
pip install -e .
```

2. Add to Python path:
```python
import sys
sys.path.append('/path/to/rfdiffusion_tutorial')
from shared_utils import structure_utils
```

## Environment Issues

### Conda Environment

**Problem**: Wrong Python version or packages

**Solutions**:
1. Recreate environment:
```bash
conda env remove -n rfdiffusion_tutorial
conda env create -f environment.yml
conda activate rfdiffusion_tutorial
```

2. Update packages:
```bash
conda update --all
pip install --upgrade -r requirements.txt
```

### Conflicting Dependencies

**Problem**: Package version conflicts

**Solution**: Use conda-forge channel:
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install <package>
```

## Performance Issues

### CPU vs GPU

**Problem**: Not sure if GPU is being used

**Check**:
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check during training
model = model.to(device)
inputs = inputs.to(device)

# Monitor GPU usage
# In terminal: watch -n 1 nvidia-smi
```

### Slow Inference

**Problem**: Generation takes too long

**Solutions**:
1. Use fewer diffusion steps:
```python
structure = sample(model, num_steps=50)  # Instead of 200
```

2. Batch multiple samples:
```python
structures = sample_batch(model, batch_size=8)
```

3. Use FP16:
```python
model = model.half()  # Use float16
```

## Getting Help

If these solutions don't work:

1. **Check Logs**: Look at error messages carefully
2. **Search Issues**: Check GitHub issues in official repos
3. **Ask Community**: Post on discussion forums
4. **Open Issue**: Create issue in this repo with:
   - Python version
   - Package versions (pip list)
   - Error message
   - Minimal reproducible example

## Additional Resources

- [PyTorch Troubleshooting](https://pytorch.org/docs/stable/notes/faq.html)
- [BioPython FAQ](https://biopython.org/wiki/FAQ)
- [RosettaCommons Forums](https://www.rosettacommons.org/forum)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/biopython)

## Reporting Bugs

When reporting issues, include:

```python
import sys
import torch
import numpy as np
import Bio

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"BioPython: {Bio.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```
