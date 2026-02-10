# RFDiffusion Tutorial Workspace Instructions

## Project Overview
This workspace is dedicated to recreating and learning from David Baker's RFDiffusion papers. Each paper is treated as an individual learning module with progressive difficulty.

## Workspace Structure
- Each paper has its own folder with notebooks, source code, data, and results
- Shared utilities are in `shared_utils/`
- Documentation is in `docs/`
- Progress from basic to advanced concepts

## Development Guidelines
- Use conda environment for dependency management
- Follow PyTorch best practices for neural network implementations
- Document all protein structure manipulations clearly
- Include visualization for protein structures
- Test implementations against published results where possible

## Coding Standards
- Use type hints for all functions
- Include docstrings following NumPy style
- Keep notebooks focused on learning objectives
- Separate reusable code into modules
- Use logging instead of print statements for debugging

## Paper-Specific Instructions
- Start with RFDiffusion 1 before moving to more advanced papers
- Each module should be self-contained but can import from shared_utils
- Include references to original papers in all implementations
- Provide clear explanations for mathematical concepts
