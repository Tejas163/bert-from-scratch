# Build BERT From Scratch

A comprehensive self-learning project to implement BERT (Bidirectional Encoder Representations from Transformers) from the ground up using PyTorch.

## Overview

This project provides a step-by-step implementation of BERT, designed for learning purposes following a modular, task-based approach. Each task builds upon the previous one, guiding you from PyTorch fundamentals to production-ready BERT implementations.

## What You'll Build

By completing this project, you will have implemented:

- ✅ **PyTorch Fundamentals** - Tensors, autograd, nn.Module, DataLoader
- ✅ **Self-Attention Mechanism** - Query, Key, Value matrices
- ✅ **Multi-Head Attention** - Parallel attention heads
- ✅ **Transformer Encoder** - Feed-forward networks, residual connections, LayerNorm
- ✅ **Positional Embeddings** - Sinusoidal and learned
- ✅ **BERT Architecture** - Embeddings + Encoder + Pooler
- ✅ **Pretraining** - Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
- ✅ **Finetuning** - Classification and Question Answering
- ✅ **Inference & Serving** - REST API with FastAPI
- ✅ **Production Optimization** - ONNX export, quantization

## Learning Path

| Task | Topic | Estimated Time |
|------|-------|----------------|
| 1 | PyTorch Foundations | 2 hours |
| 2 | Self-Attention Mechanism | 2 hours |
| 3 | Multi-Head Attention | 2 hours |
| 4 | Transformer Encoder | 2 hours |
| 5 | Positional Embeddings | 1 hour |
| 6 | BERT Embeddings & Encoder | 2 hours |
| 7 | BERT Pretraining (MLM + NSP) | 3 hours |
| 8 | BERT Finetuning - Classification | 2 hours |
| 9 | BERT Finetuning - QA | 2 hours |
| 10 | Inference & Model Serving | 2 hours |
| 11 | Production Optimization | 2 hours |
| 12 | Advanced Exercises | 4+ hours |

## Quick Start

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0+
- 8GB+ RAM (for model training)

### Installation

```bash
# Clone the repository
git clone https://github.com/Tejas163/bert-from-scratch.git
cd bert-from-scratch

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

```bash
# Launch Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Navigate to `tasks/task-01-pytorch-basics/notebook.ipynb` to begin!

### Building the Documentation Site

This project uses Quarto to generate a documentation website:

```bash
# Install Quarto (https://quarto.org/docs/download/)

# Preview the site locally
quarto preview

# Build the site
quarto render
```

## Project Structure

```
bert-from-scratch/
├── tasks/                      # Learning modules (12 tasks)
│   ├── task-01-pytorch-basics/
│   │   ├── overview.md        # Task explanation
│   │   └── notebook.ipynb     # Interactive notebook
│   ├── task-02-self-attention/
│   ├── task-03-multi-head-attention/
│   ├── task-04-transformer-encoder/
│   ├── task-05-positional-embeddings/
│   ├── task-06-bert-embeddings-encoder/
│   ├── task-07-bert-pretraining/
│   ├── task-08-bert-classification/
│   ├── task-09-bert-qa/
│   ├── task-10-inference-serving/
│   ├── task-11-production-optimization/
│   └── task-12-advanced-exercises/
├── src/                      # Production-ready Python code
│   └── bert/
│       ├── attention.py      # Self & Multi-head attention
│       ├── transformer.py    # Transformer encoder
│       ├── embeddings.py     # BERT embeddings
│       └── bert.py          # Complete BERT models
├── exercises/                # Additional exercises
├── reference.qmd            # API documentation
├── index.qmd                # Home page
├── _quarto.yml              # Site configuration
├── pyproject.toml           # Python project config
└── requirements.txt         # Dependencies
```

## Features

- **12 Comprehensive Tasks** - From basics to advanced
- **Interactive Notebooks** - Learn by doing
- **Production-Ready Code** - Type hints, docstrings, modular design
- **Exercises** - Test your understanding
- **API Reference** - Complete documentation
- **GitHub Pages** - Published documentation site

## Usage Examples

### Using the BERT Model

```python
import torch
from src.bert import BERTForClassification

# Create model
model = BERTForClassification(
    vocab_size=30000,
    num_labels=2,
    d_model=256,
    num_heads=4,
    num_layers=4,
    d_ff=1024
)

# Forward pass
token_ids = torch.randint(0, 30000, (2, 128))
logits = model(token_ids)
print(f"Output shape: {logits.shape}")
```

### Training

```python
from torch.utils.data import DataLoader
from src.bert import BERTForPretraining

model = BERTForPretraining(vocab_size=30000)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(3):
    for batch in dataloader:
        loss, _, _ = model(**batch)
        loss.backward()
        optimizer.step()
```

## Advanced Exercises

After completing all tasks, try:

1. **HuggingFace Compatibility** - Make your BERT load pretrained weights
2. **RoBERTa** - Remove NSP, use dynamic masking
3. **ALBERT** - Parameter sharing, factorized embeddings
4. **Custom Extensions** - Different activations, pooling strategies

## Documentation

- **Live Site**: https://tejas163.github.io/bert-from-scratch/
- **API Reference**: See `reference.qmd`

## Requirements

### Core Dependencies
- torch>=2.0.0
- numpy>=1.24.0
- pandas>=2.0.0
- tqdm>=4.65.0

### Data Processing
- datasets>=2.14.0
- tokenizers>=0.13.0

### Model Training
- transformers>=4.30.0
- accelerate>=0.20.0
- tensorboard>=2.13.0

### Jupyter & Notebooks
- jupyter>=1.0.0
- ipykernel>=6.0.0
- nbconvert>=7.0.0

### Testing & Linting
- pytest>=7.0.0
- black>=23.0.0
- ruff>=0.0.270

### Serving & API
- fastapi>=0.100.0
- uvicorn>=0.23.0

### Optimization
- onnx>=1.14.0
- onnxruntime>=1.15.0

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License

## Acknowledgments

- Based on the original BERT paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Inspired by the Transformer architecture from "Attention Is All You Need"

## Support

- Open an issue for bugs or questions
- Check the notebooks for detailed explanations
- Review the API reference for code documentation

---

**Happy Learning!** 🚀
