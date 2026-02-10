# Module 1 Data Directory

This directory contains example data and instructions for obtaining training/testing datasets.

## Directory Structure

```
data/
├── examples/           # Small example PDB files for testing
├── training/          # Training dataset (download separately)
├── validation/        # Validation structures
└── results/           # Generated structures (output)
```

## Example Data

### Included Examples
Small PDB files are included for immediate testing:
- `1ubq.pdb` - Ubiquitin (76 residues) - Good test case
- `5trv.pdb` - Small designed protein
- Coming soon: Additional curated examples

### Download Examples
```bash
cd examples

# Download from RCSB PDB
wget https://files.rcsb.org/download/1UBQ.pdb -O 1ubq.pdb
wget https://files.rcsb.org/download/5TRV.pdb -O 5trv.pdb

# Download additional test structures
# Small proteins (< 100 residues) are best for learning
```

## Training Data

### Official RFDiffusion Training Set
The official RFDiffusion model was trained on structures from the PDB with specific filtering criteria.

**Filtering Criteria** (from paper):
- Resolution < 2.5 Å
- Length: 50-500 residues  
- Sequence identity < 30% (clustered)
- Quality checks (R-free, clashes, geometry)

### Download Training Data

#### Option 1: Use Official Dataset
```bash
# Follow instructions from official RFDiffusion repo
# https://github.com/RosettaCommons/RFdiffusion
```

#### Option 2: Create Custom Dataset
```bash
# Download from RCSB PDB
# Use our filtering script

cd training/
python ../../scripts/download_pdb_dataset.py \
    --resolution 2.5 \
    --min_length 50 \
    --max_length 500 \
    --output_dir ./structures

# This will download and filter appropriate structures
```

#### Option 3: Use Preprocessed Dataset
Several preprocessed datasets available:
- **PDB Snapshot**: Structures as of specific date
- **CATH**: Domain-level structures
- **SCOPe**: Classification-based structures

```bash
# Example: Download CATH dataset
wget http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz
tar -xzf cath-dataset-nonredundant-S40.pdb.tgz
```

## Validation Data

Use separate structures not in training set:
- CASP14/15 targets
- Recently deposited structures
- De novo designed proteins for comparison

```bash
cd validation/

# Download CASP targets
# Or use recent PDB deposits
```

## Data Processing

### Convert PDB to Features

```python
from shared_utils import structure_utils

# Load PDB
structure = structure_utils.load_pdb('examples/1ubq.pdb')

# Extract backbone
coords = structure_utils.get_backbone_coords(structure)

# Convert to frames (for training)
from src.frames import BackboneFrames
frames = BackboneFrames.from_atoms(coords[:, 0], coords[:, 1], coords[:, 2])
```

### Create Training Dataset

```python
from src.dataset import ProteinStructureDataset

# Create dataset from directory of PDB files
dataset = ProteinStructureDataset(
    pdb_dir='data/training/structures',
    max_length=128
)

print(f"Dataset size: {len(dataset)}")
```

## Data Statistics

### Expected Training Set Size
- Structures: ~10,000-50,000
- Total residues: ~5-20 million
- Disk space: ~5-50 GB (depending on preprocessing)

### Recommended Splits
- Training: 90%
- Validation: 5%
- Test: 5%

## Quality Control

Before using structures, verify:
- [ ] Resolution ≤ 2.5 Å
- [ ] No large gaps in sequence
- [ ] Reasonable B-factors
- [ ] No unresolved regions
- [ ] Passing geometry checks

Use provided quality control script:
```bash
python scripts/check_structure_quality.py --pdb examples/1ubq.pdb
```

## Data Augmentation

During training, apply augmentation:
- Random rotation (SE(3) augmentation)
- Random translation
- Random cropping (for long proteins)
- Coordinate noise (small)

See `src/dataset.py` for implementation.

## Citing Data Sources

If using PDB structures, cite:
```
Berman, H.M., et al. (2000) The Protein Data Bank. 
Nucleic Acids Research, 28: 235-242.
```

For CATH:
```
Sillitoe, I., et al. (2021) CATH: increased structural coverage of functional space.
Nucleic Acids Research, 49: D266-D273.
```

## Troubleshooting

### Issue: PDB Files Not Loading
- Check file format (PDB vs mmCIF)
- Verify file is not corrupted
- Try different BioPython parser

### Issue: Out of Memory
- Reduce batch size
- Use shorter sequences
- Filter by length more strictly

### Issue: Slow Data Loading
- Preprocess and cache structures
- Use memory-mapped files
- Parallel data loading

## Next Steps

1. Download example structures
2. Test loading and processing
3. Download full training set (if training)
4. Proceed to training notebooks

## Resources

- **RCSB PDB**: https://www.rcsb.org/
- **PDB File Format**: https://www.wwpdb.org/documentation/file-format
- **CATH Database**: http://www.cathdb.info/
- **SCOPe**: https://scop.berkeley.edu/
