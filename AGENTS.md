# AGENTS.md - Guidelines for AI Coding Agents

This document provides instructions for AI coding agents working on the YouTube Video Views Prediction project.

## Project Overview

A Python-based machine learning project for predicting YouTube video performance using historical data and video metadata. Uses UV for dependency management, scikit-learn for ML, and Jupyter notebooks for analysis.

## Build/Run Commands

```bash
# Setup virtual environment and install dependencies
uv sync

# Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install new dependencies
uv add <package_name>

# Run Python scripts
uv run python script.py
# or with activated env:
python script.py

# Run Jupyter notebooks
uv run jupyter notebook

# Run specific notebook
uv run jupyter notebook notebook.ipynb
```

## Testing

```bash
# Run a single Python test file
python -m pytest tests/test_module.py -v

# Run a specific test function
python -m pytest tests/test_module.py::test_function_name -v

# Run all tests in a directory
python -m pytest tests/ -v

# Run tests matching a pattern
python -m pytest -k "test_name_pattern" -v
```

## Code Style Guidelines

### Imports
- Group imports: stdlib, third-party, local (each group separated by blank line)
- Use absolute imports within `utils/` package
- Handle optional dependencies gracefully with try/except blocks
- Example:
```python
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
```

### Formatting
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use double quotes for strings (unless single quotes avoid escaping)
- Add blank lines between function definitions (2 lines between classes)

### Types
- Use type hints for all function parameters and return values
- Use `from typing import` for complex types (Optional, Union, Dict, List, Tuple, Any)
- Example: `def process_text(text: Union[str, None]) -> str:`

### Naming Conventions
- Functions: `snake_case` (e.g., `calculate_vif`, `filter_outliers`)
- Classes: `PascalCase` (e.g., `TextPreprocessor`, `TextVisualizer`)
- Constants: `UPPER_CASE` (e.g., `MATPLOTLIB_AVAILABLE`, `SUPPORTED_MODELS`)
- Private methods/vars: `_leading_underscore` (e.g., `_validate_dependencies`)
- Module-level private vars: `_UPPER_CASE` for constants

### Error Handling
- Use try/except blocks for operations that may fail
- Provide descriptive error messages with context
- Use `raise ValueError` for invalid parameters
- Use `raise ImportError` for missing optional dependencies with installation instructions
- Fail gracefully for data operations

### Docstrings
- Use NumPy-style docstrings
- Include: Description, Parameters, Returns, Examples
- Example:
```python
def feature_scaling(data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    Scale features using specified method.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe to scale
    method : str, default='standard'
        Scaling method ('standard', 'minmax', 'robust')
    
    Returns:
    --------
    pd.DataFrame
        Scaled dataframe
    
    Examples:
    ---------
    >>> df_scaled = feature_scaling(df, method='minmax')
    """
```

### Code Organization
- Use unicode box drawing characters for section headers (╔═╗║╚╝)
- Group related functions with descriptive comments
- Keep functions focused and under 50 lines when possible
- Use `__all__` in `__init__.py` to define public API

### Project Structure
```
E:\Personal Projects\Youtube Video Views Prediction/
├── utils/                    # Reusable utility modules
│   ├── __init__.py          # Public API exports
│   ├── preprocessing.py     # Data preprocessing functions
│   ├── text_preprocessing.py # NLP text processing
│   ├── regression_evals_and_tuning.py  # ML model tuning
│   ├── visualization.py     # Plotting functions
│   ├── feature_selection.py # Feature selection utilities
│   ├── statistics.py        # Statistical analysis
│   └── preprocess_youtube_data.py  # Domain-specific preprocessing
├── data/                    # Data files
├── output/                  # Generated outputs/plots
├── reference/               # Reference materials
├── notebook.ipynb           # Main analysis notebook
├── pyproject.toml           # Project configuration
└── requirements.txt         # Dependencies
```

## Dependencies

Key packages (managed via UV):
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning
- xgboost - Gradient boosting
- matplotlib, seaborn - Visualization
- wordcloud, textblob - NLP
- gradio - Web interface
- deep-translator, langid - Translation
- category-encoders - Categorical encoding

## Guidelines

1. Always validate DataFrame columns exist before accessing
2. Use copy() when modifying DataFrames to avoid side effects
3. Preserve datetime columns during transformations
4. Support both training and inference modes (fit/transform pattern)
5. Handle missing values explicitly
6. Use type hints for all public functions
7. Add docstrings with examples for public APIs
8. Keep imports at top of file, grouped by source
9. Avoid committing data files, outputs, or notebooks with outputs
10. Use random_state=42 for reproducibility in ML code
