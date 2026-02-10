# Running RFDiffusion Tutorial on Google Colab

This guide shows how to run the tutorial notebooks on Google Colab with free GPU access.

## 🚀 Quick Start

### Option 1: Direct Colab Links (Easiest)
Click these links to open notebooks directly in Colab:

- [01 Introduction](https://colab.research.google.com/github/erickummelstedt/rfdiffusion_tutorial/blob/main/01_rfdiffusion_original/notebooks/01_introduction.ipynb)
- More notebooks coming soon...

### Option 2: Manual Setup

1. **Open Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Enable GPU**:
   - Click `Runtime` → `Change runtime type`
   - Select `T4 GPU` (free tier)
   - Click `Save`

3. **Clone Repository**:
```python
# Run this in first cell
!git clone https://github.com/erickummelstedt/rfdiffusion_tutorial.git
%cd rfdiffusion_tutorial
```

4. **Install Dependencies**:
```python
# Run in second cell
!pip install -q biopython biotite matplotlib seaborn plotly einops
!pip install -e .
```

5. **Verify Setup**:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 📝 Colab Setup Cell

Add this cell at the top of any notebook when running on Colab:

```python
# Colab Setup Cell - Run this first!
import sys
import os

# Check if running on Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("🔧 Setting up Colab environment...")
    
    # Clone repo if not already present
    if not os.path.exists('rfdiffusion_tutorial'):
        !git clone https://github.com/erickummelstedt/rfdiffusion_tutorial.git
        %cd rfdiffusion_tutorial
    
    # Install dependencies
    !pip install -q biopython biotite matplotlib seaborn plotly einops timm
    !pip install -q -e .
    
    # Verify GPU
    import torch
    print(f"\n✅ Setup complete!")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running locally - skipping Colab setup")
```

## 💡 Tips for Using Colab

### GPU Access
- **Free tier**: ~12 hours of T4 GPU per session
- **Colab Pro**: Longer sessions, better GPUs (V100/A100)
- Sessions timeout after inactivity

### Saving Work
- **Download notebooks**: `File` → `Download` → `.ipynb`
- **Save to Drive**: `File` → `Save a copy in Drive`
- **Commit to GitHub**: Use GitHub integration

### File Management
```python
# Upload files from your computer
from google.colab import files
uploaded = files.upload()

# Download results
files.download('results/generated_structure.pdb')

# Mount Google Drive (persist data)
from google.colab import drive
drive.mount('/content/drive')
```

### Data Persistence
Colab VMs are temporary! Options:
- **Google Drive**: Mount and save to Drive
- **GitHub**: Commit results back to repo
- **Download**: Save important files locally

## 🎯 Workflow Example

```python
# 1. Setup (run once per session)
!git clone https://github.com/erickummelstedt/rfdiffusion_tutorial.git
%cd rfdiffusion_tutorial
!pip install -q -e .

# 2. Enable GPU
# Runtime → Change runtime type → T4 GPU

# 3. Run your code
from src.diffusion import DiffusionModel
model = DiffusionModel(...)

# 4. Save results to Drive
from google.colab import drive
drive.mount('/content/drive')
torch.save(model.state_dict(), '/content/drive/MyDrive/rfdiffusion/model.pth')
```

## 🔄 Updating Repository

```python
# Pull latest changes
%cd rfdiffusion_tutorial
!git pull origin main

# Reinstall if packages changed
!pip install -q -e .
```

## ⚡ Performance Tips

### Maximize GPU Utilization
```python
# Check GPU memory
import torch
print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# Clear cache if needed
torch.cuda.empty_cache()
```

### Batch Processing
```python
# Use larger batches on GPU
batch_size = 32 if torch.cuda.is_available() else 4
```

### Mixed Precision (Faster Training)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
```

## 🐛 Troubleshooting

### Issue: GPU Not Available
```python
# Check runtime type
# Runtime → Change runtime type → Select GPU
```

### Issue: Session Disconnected
```python
# Reconnect and rerun setup cells
# Your files are lost unless saved to Drive!
```

### Issue: Out of Memory
```python
# Reduce batch size
# Clear GPU cache
torch.cuda.empty_cache()
```

### Issue: Packages Not Found
```python
# Reinstall
!pip install --upgrade -e .
```

## 📊 Monitoring Resources

```python
# Check GPU usage
!nvidia-smi

# Check RAM
!free -h

# Check disk space
!df -h
```

## 🔗 Useful Colab Features

### Code Snippets
- Click `</> Insert` → `Code snippets`
- Snippets for common tasks

### Forms
```python
#@title Training Configuration
learning_rate = 0.001 #@param {type:"number"}
batch_size = 32 #@param {type:"slider", min:1, max:128}
use_gpu = True #@param {type:"boolean"}
```

### Shell Commands
```python
!ls -la  # Run shell commands
%cd /path  # Change directory
```

## 📱 Mobile Access

You can even run on your phone/tablet:
1. Open Colab on mobile browser
2. Full notebook editing
3. Code execution (uses Colab servers)

## 💰 Cost Comparison

| Option | GPU | Cost | Best For |
|--------|-----|------|----------|
| Colab Free | T4 | Free | Learning, small models |
| Colab Pro | V100/A100 | $10/mo | Regular use |
| Paperspace | Various | ~$0.50/hr | Flexible |
| AWS/GCP | Various | ~$1-3/hr | Production |

## ✅ Ready to Start!

1. Open [Google Colab](https://colab.research.google.com)
2. Click [this link](https://colab.research.google.com/github/erickummelstedt/rfdiffusion_tutorial/blob/main/01_rfdiffusion_original/notebooks/01_introduction.ipynb)
3. Enable GPU (Runtime → Change runtime type)
4. Run the setup cell
5. Start learning! 🚀

## 📚 Resources

- [Colab Quickstart](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colab Pro Features](https://colab.research.google.com/signup)
- [GPU Tips](https://colab.research.google.com/notebooks/gpu.ipynb)
