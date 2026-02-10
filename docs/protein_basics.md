# Protein Structure Basics

A primer for understanding protein structure fundamentals needed for this tutorial.

## Protein Structure Hierarchy

### Primary Structure
The linear sequence of amino acids connected by peptide bonds.
- 20 standard amino acids
- Sequence determines 3D structure
- Written N-terminus to C-terminus

### Secondary Structure
Local structural patterns stabilized by hydrogen bonds:

**α-Helix**
- Right-handed spiral
- 3.6 residues per turn
- Stabilized by i to i+4 hydrogen bonds
- Characterized by φ ≈ -60°, ψ ≈ -45°

**β-Sheet**
- Extended strands
- Can be parallel or antiparallel
- Inter-strand hydrogen bonds
- Characterized by φ ≈ -120°, ψ ≈ 120°

**Loops/Turns**
- Connect helices and sheets
- More flexible regions
- Often functionally important

### Tertiary Structure
3D arrangement of entire protein chain:
- Hydrophobic core
- Charged surface
- Binding sites
- Active sites

### Quaternary Structure
Assembly of multiple protein chains:
- Homo- or hetero-oligomers
- Symmetry (C2, C3, D2, etc.)
- Functional assemblies

## Key Concepts

### Backbone Atoms
Every residue has three main chain atoms:
- **N**: Nitrogen (amide)
- **CA**: Alpha carbon (central)
- **C**: Carbonyl carbon

### Side Chains
Variable group attached to CA:
- 20 different types
- Determine properties
- Not included in backbone-only models

### Dihedral Angles

**φ (Phi)**: C(i-1) - N(i) - CA(i) - C(i)
**ψ (Psi)**: N(i) - CA(i) - C(i) - N(i+1)
**ω (Omega)**: CA(i) - C(i) - N(i+1) - CA(i+1) (usually ≈ 180°)

### Ramachandran Plot
Plot of φ vs ψ angles:
- Shows allowed conformations
- Identifies structural problems
- Region clusters indicate secondary structure

## PDB File Format

PDB (Protein Data Bank) files contain atomic coordinates:

```
ATOM      1  N   MET A   1      20.154  29.699   5.276  1.00 49.05           N
ATOM      2  CA  MET A   1      21.379  29.691   4.498  1.00 49.05           C
ATOM      3  C   MET A   1      21.197  30.431   3.181  1.00 49.05           C
```

Fields:
- Record type (ATOM)
- Atom serial number
- Atom name
- Residue name (3-letter code)
- Chain ID
- Residue number
- X, Y, Z coordinates (Angstroms)
- Occupancy
- B-factor (temperature factor)
- Element symbol

## Geometric Representations

### Cartesian Coordinates
Direct (x, y, z) positions:
- Absolute positions in space
- Simple but not invariant to rotation/translation

### Internal Coordinates
Bonds, angles, dihedrals:
- More compact representation
- Captures local geometry
- Used in many prediction methods

### Frames/Rigid Bodies
Rotation and translation frames:
- Position and orientation
- SE(3) group elements
- Used in RFDiffusion

## Structure Quality Metrics

### RMSD (Root Mean Square Deviation)
Measures average atomic distance between structures:
- RMSD < 1.0 Å: Very similar
- RMSD < 2.0 Å: Good match
- RMSD > 3.0 Å: Different structures

### TM-Score
Length-normalized similarity metric:
- Range [0, 1]
- TM > 0.5: Similar fold
- Less sensitive to length than RMSD

### Clash Score
Number of unrealistic atomic overlaps:
- Atoms too close together
- Indicates modeling errors
- Should be minimized

### Packing Density
How tightly atoms are packed:
- Core should be well-packed
- Surface more loosely packed
- Indicator of stability

## Protein Design Considerations

### Designability
Not all structures can be designed:
- Must be stable
- Must fold
- Must avoid aggregation

### Folding Funnel
Energy landscape should favor native state:
- Native state is low energy
- Few competing states
- Cooperative folding

### Sequence-Structure Relationship
Sequence determines structure:
- But mapping is many-to-one
- Multiple sequences fold to same structure
- Design problem: find good sequence

## Resources

- **PDB**: [rcsb.org](https://www.rcsb.org/)
- **ProteInS**: Structural Classification of Proteins
- **PyMOL**: Visualization software
- **BioPython**: Python tools for structural biology

## Next Steps

With this foundation, you're ready to start:
1. **[Module 1: RFDiffusion Original](../01_rfdiffusion_original/)**
2. Explore PDB files
3. Practice calculating structural properties
