# Contributing to RFDiffusion Tutorial

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (Python version, OS, etc.)

### Code Contributions

1. **Fork the repository**
```bash
git clone https://github.com/yourusername/rfdiffusion_tutorial.git
cd rfdiffusion_tutorial
```

2. **Create a branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
```bash
pytest tests/
```

5. **Commit with clear messages**
```bash
git commit -m "Add: Clear description of your changes"
```

6. **Push and create pull request**
```bash
git push origin feature/your-feature-name
```

## Coding Standards

### Python Style

Follow PEP 8 with these specifics:

```python
# Use type hints
def function_name(param: str, count: int = 0) -> bool:
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        param: Description of param
        count: Description of count
        
    Returns:
        Description of return value
    """
    pass

# Docstrings: NumPy style
# Line length: 88 characters (Black default)
# Imports: stdlib, third-party, local (separated by blank lines)
```

### Code Organization

```
module/
├── __init__.py          # Package initialization
├── core.py              # Core functionality
├── utils.py             # Utility functions
└── tests/               # Tests mirroring structure
    └── test_core.py
```

### Naming Conventions

- **Functions/methods**: `lowercase_with_underscores`
- **Classes**: `CapitalizedWords`
- **Constants**: `ALL_CAPS_WITH_UNDERSCORES`
- **Private**: `_leading_underscore`

## Documentation Standards

### Module Documentation

Every module should have:
```python
"""
Brief module description.

Longer explanation of what the module provides and when to use it.
"""
```

### Function Documentation

```python
def calculate_metric(coords: np.ndarray, threshold: float = 2.0) -> float:
    """
    Calculate a quality metric from coordinates.
    
    This function computes... [detailed explanation]
    
    Args:
        coords: Array of shape (n, 3) containing atomic coordinates
        threshold: Distance threshold for metric calculation
        
    Returns:
        Calculated metric value
        
    Raises:
        ValueError: If coords is not 2D array
        
    Examples:
        >>> coords = np.random.rand(100, 3)
        >>> metric = calculate_metric(coords)
        >>> print(f"Metric: {metric:.2f}")
    """
```

### Notebook Standards

Notebooks should:
1. Have clear section headers with markdown
2. Include learning objectives at the top
3. Explain concepts before code
4. Have inline comments for complex code
5. Include visualizations
6. End with exercises or next steps

Example structure:
```markdown
# Notebook Title

## Learning Objectives
- Objective 1
- Objective 2

## Background
[Explanation]

## Implementation
[Code with explanations]

## Exercises
1. Exercise 1
2. Exercise 2

## Next Steps
[What to do next]
```

## Testing

### Writing Tests

Use pytest:
```python
def test_function_name():
    """Test that function_name works correctly."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_name(input_data)
    
    # Assert
    assert result.shape == (10, 3)
    assert np.allclose(result.mean(), 0.0, atol=1e-6)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_structure_utils.py

# Run with coverage
pytest --cov=shared_utils tests/
```

## Types of Contributions

### 🐛 Bug Fixes
- Fix incorrect implementations
- Resolve errors or crashes
- Improve error handling

### 📚 Documentation
- Improve existing docs
- Add examples
- Fix typos
- Add tutorials

### ✨ New Features
- New modules/papers
- Additional utilities
- Enhanced visualizations

### 🔬 Examples
- Real protein examples
- Case studies
- Worked problems

### 🎨 Notebooks
- New tutorial notebooks
- Interactive examples
- Visualization improvements

## Module Contribution Template

When adding a new paper/module:

```
XX_module_name/
├── README.md                    # Module overview
├── notebooks/
│   ├── 01_introduction.ipynb
│   ├── 02_theory.ipynb
│   └── 03_implementation.ipynb
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── utils.py
├── data/
│   ├── examples/
│   └── README.md
├── results/
│   └── .gitkeep
└── tests/
    └── test_model.py
```

README.md should include:
- Paper reference
- Learning objectives
- Prerequisites
- Key concepts
- Resources
- Next steps

## Review Process

1. **Automated checks** run on pull requests:
   - Code style (Black, flake8)
   - Tests pass
   - Documentation builds

2. **Manual review** by maintainers:
   - Code quality
   - Documentation clarity
   - Test coverage
   - Educational value

3. **Feedback incorporated** by contributor

4. **Merge** when approved

## Communication

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Pull Requests**: For code contributions

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in relevant documentation

## Questions?

Feel free to open an issue with the `question` label or start a discussion!

## Code of Conduct

Be respectful, constructive, and welcoming. We're all here to learn!

---

Thank you for contributing to protein design education! 🧬
