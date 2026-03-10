# Build BERT From Scratch

A comprehensive self-learning project to implement BERT from scratch using PyTorch.

## Project Overview

This project provides a step-by-step implementation of BERT (Bidirectional Encoder Representations from Transformers), designed for learning purposes following a modular, task-based approach.

## Learning Path

| Task | Topic |
|------|-------|
| 1 | PyTorch Foundations |
| 2 | Self-Attention Mechanism |
| 3 | Multi-Head Attention |
| 4 | Transformer Encoder |
| 5 | Positional Embeddings |
| 6 | BERT Embeddings & Encoder |
| 7 | BERT Pretraining (MLM + NSP) |
| 8 | BERT Finetuning - Classification |
| 9 | BERT Finetuning - QA |
| 10 | Inference & Model Serving |
| 11 | Production Optimization |
| 12 | Advanced Exercises |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/autokernel.git
cd autokernel

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Then navigate to `tasks/task-01-pytorch-basics/notebook.ipynb` to begin!

## Project Structure

```
autokernel/
├── tasks/           # Learning modules (12 tasks)
│   └── task-*/
│       ├── overview.md
│       └── notebook.ipynb
├── src/             # Modular Python code
│   └── bert/
│       ├── attention.py
│       ├── transformer.py
│       ├── embeddings.py
│       └── bert.py
├── exercises/       # Additional exercises
├── reference.qmd    # API documentation
└── _quarto.yml     # Site configuration
```

## Features

- **Production-ready code** with type hints and docstrings
- **12 comprehensive tasks** from basics to advanced
- **Notebook-friendly** with Jupyter/Quarto
- **Modular architecture** for easy extension
- **Exercises** to test understanding

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Jupyter

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a PR.
