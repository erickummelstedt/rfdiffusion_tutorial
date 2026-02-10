# Diffusion Models for Protein Design

An introduction to diffusion models and their application to protein structure generation.

## What Are Diffusion Models?

Diffusion models are a class of generative models that learn to generate data by reversing a gradual noising process.

### Key Idea
1. **Forward Process**: Gradually add noise to data until it becomes pure noise
2. **Reverse Process**: Learn to remove noise step-by-step to generate data

This is analogous to:
- Watching ink diffuse in water (forward)
- Reversing the process to recover the original drop (reverse)

## Mathematical Framework

### Forward Diffusion Process

Starting with data $\mathbf{x}_0$, we add Gaussian noise over $T$ timesteps:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$

where $\beta_t$ is the noise schedule.

At any timestep $t$:
$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

where $\bar{\alpha}_t = \prod_{i=1}^t (1-\beta_i)$

### Reverse Diffusion Process

Learn to reverse the process:
$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))$$

The neural network $\theta$ predicts the mean (and optionally variance).

### Training Objective

Train by predicting the noise:
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)\|^2\right]$$

where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ is the noise added.

## Score-Based Perspective

Alternative formulation using score functions:

The **score** is the gradient of the log probability:
$$\nabla_\mathbf{x} \log p(\mathbf{x})$$

**Score matching**: Train network to estimate the score:
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_t}\left[\|\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) - s_\theta(\mathbf{x}_t, t)\|^2\right]$$

## Application to Proteins

### Challenges
1. **3D Structure**: Proteins are 3D objects, not images
2. **Invariances**: Must respect rotation/translation
3. **Constraints**: Bond lengths, angles, chirality
4. **Variable Length**: Proteins have different numbers of residues

### RFDiffusion Approach

**Representation**: 
- Backbone frames (rigid body transformations)
- SE(3)-equivariant network

**Diffusion in SE(3)**:
- Position: Standard Gaussian diffusion in $\mathbb{R}^3$
- Orientation: Diffusion on SO(3) rotation group

**Architecture**:
- Structure module from RosettaFold
- Attention mechanisms
- Equivariant layers

### Sampling Procedure

```python
# Start with random noise
x_T = sample_noise()

# Iteratively denoise
for t in reversed(range(T)):
    # Predict noise
    noise_pred = model(x_t, t)
    
    # Remove noise
    x_{t-1} = denoise_step(x_t, noise_pred, t)

# Final structure
structure = x_0
```

## Conditional Generation

Generate proteins with specific properties:

### Motif Scaffolding
Fix certain residues, generate the rest:
- Conditional on motif positions
- Network learns to scaffold around motif

### Symmetric Design
Generate symmetric oligomers:
- Apply symmetry constraints during diffusion
- Ensure consistent symmetric copies

### Binder Design
Generate protein that binds to target:
- Conditional on target structure
- Optimize interface contacts

## Advantages of Diffusion Models

1. **Stable Training**: Easier than GANs
2. **High Quality**: Generate diverse, realistic samples
3. **Controllable**: Easy to add conditions
4. **Flexible**: Can model complex distributions

## Implementation Details

### Noise Schedule
Choice of $\beta_t$ matters:
- **Linear**: $\beta_t = \beta_{min} + (\beta_{max} - \beta_{min}) \frac{t}{T}$
- **Cosine**: Smoother schedule
- **Learned**: Train the schedule

### Timestep Encoding
Embed timestep $t$ into network:
- Sinusoidal encoding
- Learned embedding
- Affects conditioning

### Network Architecture
Key components:
- Encoder: Extract features
- Processor: Transform features
- Decoder: Predict noise/score

## Comparison to Other Methods

### GANs (Generative Adversarial Networks)
- **Pros**: Fast generation
- **Cons**: Unstable training, mode collapse

### VAEs (Variational Autoencoders)
- **Pros**: Fast, interpretable latent space
- **Cons**: Blurry samples, limited capacity

### Diffusion Models
- **Pros**: High quality, stable training
- **Cons**: Slower generation (many steps)

### For Proteins
Diffusion models excel because:
- Handle 3D geometry naturally
- Easy to incorporate physical constraints
- Flexible conditioning mechanisms

## Key Papers

1. **DDPM** (Ho et al. 2020): Denoising Diffusion Probabilistic Models
2. **Score-Based** (Song et al. 2021): Score-Based Generative Modeling
3. **Improved DDPM** (Nichol & Dhariwal 2021): Improvements
4. **RFDiffusion** (Watson et al. 2023): Application to proteins

## Practical Tips

### Training
- Start with simple unconditional model
- Gradually add conditioning
- Monitor sample quality, not just loss
- Use validation sampling frequently

### Sampling
- More timesteps = better quality (but slower)
- Temperature controls diversity
- Guidance strength for conditioning

### Debugging
- Visualize forward process
- Check if network predicts noise correctly
- Monitor RMSD to training data
- Validate physical properties

## Code Example

```python
import torch

class SimpleDiffusion:
    def __init__(self, T=1000):
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, x0, t):
        """Forward process"""
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t]
        xt = torch.sqrt(alpha_bar_t) * x0 + \
             torch.sqrt(1 - alpha_bar_t) * noise
        return xt, noise
    
    def denoise_step(self, xt, noise_pred, t):
        """Reverse process step"""
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        
        # Predict x0
        x0_pred = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / \
                  torch.sqrt(alpha_bar_t)
        
        # Sample x_{t-1}
        if t > 0:
            noise = torch.randn_like(xt)
            x_prev = torch.sqrt(alpha_t) * x0_pred + \
                     torch.sqrt(1 - alpha_t) * noise
        else:
            x_prev = x0_pred
        
        return x_prev
```

## Next Steps

Ready to apply diffusion models to proteins:
1. **[Module 1: RFDiffusion](../01_rfdiffusion_original/)**: Start here
2. Implement simple 1D diffusion model
3. Build up to protein structures
4. Experiment with conditioning

## Resources

- [Yang Song's Blog](https://yang-song.net/blog/2021/score/): Excellent intuition
- [Lil'Log](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): Comprehensive overview
- [Hugging Face Tutorial](https://huggingface.co/blog/annotated-diffusion): Code walkthrough
