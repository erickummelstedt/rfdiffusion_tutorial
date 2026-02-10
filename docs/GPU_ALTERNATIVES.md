# GPU Alternatives to Google Colab

Having trouble with Colab? Here are reliable alternatives for GPU access.

## 🆓 Free Options

### 1. Kaggle Notebooks (Recommended)

**Why:** Most stable free GPU option, 30 hours/week

**Setup:**
1. Go to [kaggle.com](https://www.kaggle.com)
2. Create free account
3. Click "Create" → "New Notebook"
4. Settings → Accelerator → GPU T4 x2
5. Add code cell:

```python
!git clone https://github.com/erickummelstedt/rfdiffusion_tutorial.git
%cd rfdiffusion_tutorial
!pip install -q biopython biotite matplotlib seaborn plotly einops
!pip install -q -e .

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Limits:** 30 GPU hours/week, 12 hour sessions

---

### 2. Lightning AI Studio

**Why:** 22 free GPU hours/month, modern interface

**Setup:**
1. Go to [lightning.ai/studios](https://lightning.ai/studios)
2. Sign up (GitHub login works)
3. Create new Studio
4. Select GPU instance (free tier)
5. Terminal opens → run:

```bash
git clone https://github.com/erickummelstedt/rfdiffusion_tutorial.git
cd rfdiffusion_tutorial
pip install -e .
jupyter lab
```

**Limits:** 22 GPU hours/month

---

### 3. AWS SageMaker Studio Lab

**Why:** Completely free, no credit card needed

**Setup:**
1. Request account at [studiolab.sagemaker.aws](https://studiolab.sagemaker.aws)
2. Wait for approval (usually 1-3 days)
3. Once approved, start GPU runtime
4. Open terminal and run:

```bash
git clone https://github.com/erickummelstedt/rfdiffusion_tutorial.git
cd rfdiffusion_tutorial
conda env create -f environment_simple.yml
conda activate rfdiffusion_tutorial
pip install -e .
```

**Limits:** 4 hour sessions, need to request account

---

## 💰 Paid Options (Affordable)

### 4. Vast.ai (Cheapest - ~$0.20/hr)

**Why:** Cheapest option, marketplace of GPUs

**Setup:**
1. Go to [vast.ai](https://vast.ai)
2. Create account, add $5-10 credit
3. Browse GPUs → Filter by PyTorch image
4. Rent instance with Jupyter
5. Open Jupyter, create terminal:

```bash
git clone https://github.com/erickummelstedt/rfdiffusion_tutorial.git
cd rfdiffusion_tutorial
pip install -e .
```

**Cost:** $0.15-0.40/hour depending on GPU
**Best for:** Long training runs

---

### 5. RunPod (~$0.30-0.50/hr)

**Why:** Easy to use, reliable

**Setup:**
1. Go to [runpod.io](https://runpod.io)
2. Sign up, add credit
3. Deploy pod → Select "Jupyter Notebook" template
4. Choose GPU (RTX 3090 or 4090)
5. Once started, click "Connect" → JupyterLab
6. Terminal:

```bash
git clone https://github.com/erickummelstedt/rfdiffusion_tutorial.git
cd rfdiffusion_tutorial
pip install -e .
```

**Cost:** $0.30-0.80/hour
**Best for:** Reliability and ease of use

---

### 6. Paperspace Gradient

**Why:** Free tier + easy paid upgrades

**Setup:**
1. Go to [paperspace.com/gradient](https://www.paperspace.com/gradient)
2. Create account
3. Create new Notebook
4. Select GPU tier (Free or Paid)
5. Terminal:

```bash
git clone https://github.com/erickummelstedt/rfdiffusion_tutorial.git
cd rfdiffusion_tutorial
pip install -e .
```

**Cost:** Free tier (limited) or $8/month + $0.50/hr

---

### 7. Lambda Labs (~$0.50-1.10/hr)

**Why:** High-end GPUs, reliable infrastructure

**Setup:**
1. Go to [lambdalabs.com/service/gpu-cloud](https://lambdalabs.com/service/gpu-cloud)
2. Create account
3. Launch instance with PyTorch
4. SSH in and run:

```bash
git clone https://github.com/erickummelstedt/rfdiffusion_tutorial.git
cd rfdiffusion_tutorial
pip install -e .
jupyter lab --allow-root
```

**Cost:** $0.50-1.10/hour
**Best for:** Serious training

---

## 🏠 Local Mac with MPS

If you have Apple Silicon (M1/M2/M3):

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")

# Use MPS for acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
```

**Performance:** Not as fast as NVIDIA GPUs, but usable for learning
**Cost:** Free!

---

## 📊 Comparison

| Platform | Cost | GPU Hours | Best For |
|----------|------|-----------|----------|
| **Kaggle** | Free | 30/week | Learning, experiments |
| **Lightning AI** | Free | 22/month | Small projects |
| **SageMaker Lab** | Free | Unlimited* | Long-term learning |
| **Vast.ai** | ~$0.20/hr | Pay as you go | Cost-conscious training |
| **RunPod** | ~$0.40/hr | Pay as you go | Easy setup + reliability |
| **Paperspace** | $8/mo base | Varies | Regular use |
| **Lambda Labs** | ~$0.80/hr | Pay as you go | Professional work |

*4 hour sessions, but can restart

---

## 🎯 Recommendations

**Just learning?** → **Kaggle** (best free option)

**Quick experiments?** → **Lightning AI** or **Kaggle**

**Serious training?** → **Vast.ai** (cheapest) or **RunPod** (easier)

**Professional work?** → **Lambda Labs** or **RunPod**

**Have Apple Silicon?** → Try **local MPS** first

---

## 🔧 Universal Setup Script

Once you have GPU access anywhere, run this:

```bash
# Clone repo
git clone https://github.com/erickummelstedt/rfdiffusion_tutorial.git
cd rfdiffusion_tutorial

# Install dependencies
pip install biopython biotite matplotlib seaborn plotly einops torch

# Install package
pip install -e .

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

---

## 💡 Tips

1. **Save your work frequently** - Cloud sessions can disconnect
2. **Use Git** - Commit and push results regularly
3. **Download important files** - Don't rely on cloud storage
4. **Monitor costs** - Set billing alerts on paid platforms
5. **Start small** - Test with free tiers before paying

---

## 🆘 Still Having Issues?

If all cloud options fail:
1. Check if your institution has GPU servers
2. Consider renting a GPU desktop on shadow.tech
3. Use CPU for learning (slower but works)
4. Join the Discord/Slack for shared resources
