# Module 2: RFDiffusion All-Atom

## 📄 Paper
**Generative design of de novo proteins based on secondary-structure constraints using an attention-based diffusion model**  
Krishna et al., Nature (2024)  
DOI: [10.1038/s41586-025-09248-9](https://www.nature.com/articles/s41586-025-09248-9)

## 🎯 Learning Objectives

By completing this module, you will understand:

1. **All-Atom Representations**
   - Moving beyond backbone-only models
   - Side-chain conformations and rotamers
   - All-atom diffusion processes

2. **Secondary Structure Control**
   - Conditioning on secondary structure elements
   - Alpha-helix and beta-sheet design
   - Loop regions and turns

3. **Attention Mechanisms**
   - Attention for protein structure
   - Pair representations
   - Cross-attention for conditioning

4. **Advanced Sampling**
   - All-atom structure generation
   - Maintaining physical constraints
   - Side-chain packing

## 📚 Prerequisites

- **✅ Module 1**: Complete understanding of RFDiffusion basics
- **Attention Mechanisms**: Transformers, self-attention
- **Protein Chemistry**: Amino acid side chains, rotamers
- **Advanced PyTorch**: Custom layers, attention implementations

## 📂 Module Structure

```
02_rfdiffusion_allatom/
├── notebooks/
│   ├── 01_from_backbone_to_allatom.ipynb
│   ├── 02_side_chain_representations.ipynb
│   ├── 03_attention_mechanisms.ipynb
│   ├── 04_secondary_structure_control.ipynb
│   ├── 05_allatom_generation.ipynb
│   └── 06_evaluation_validation.ipynb
├── src/
│   ├── __init__.py
│   ├── allatom_model.py
│   ├── attention_layers.py
│   ├── side_chain_packing.py
│   ├── secondary_structure.py
│   └── constraints.py
├── data/
│   └── examples/
├── results/
└── tests/
```

## 🚀 Getting Started

### 1. Ensure Module 1 is Complete
You should understand backbone diffusion before starting this module.

### 2. Review Attention Mechanisms
If needed, review transformer architectures and attention.

### 3. Start Learning
```bash
jupyter notebook notebooks/01_from_backbone_to_allatom.ipynb
```

## 🔑 Key Concepts

### All-Atom Modeling
Generate complete protein structures including side chains, not just backbones.

### Secondary Structure Conditioning
Control the formation of helices, sheets, and loops during generation.

### Attention for Structure
Use attention mechanisms to capture long-range interactions between residues.

## 📊 Expected Outcomes

- ✅ Generate all-atom protein structures
- ✅ Design proteins with specified secondary structure
- ✅ Implement attention-based structure models
- ✅ Validate all-atom designs

## ⏭️ Next Module

**[Module 3: Biomolecular Interactions](../03_biomolecular_interactions/)**

## ❓ Questions?

Open an issue with the tag `module-2` for help with this module.
