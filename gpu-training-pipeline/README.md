# GPU Training Pipeline

A modular and configurable deep learning training pipeline for text classification tasks, optimized for GPU training with PyTorch and Hugging Face Transformers.

## Overview

This pipeline provides a flexible framework for training transformer-based models on text classification tasks. It features a clean, modular architecture with configuration management using Hydra, making it easy to experiment with different models, datasets, and training parameters.

## Key Features

- **Modular Architecture**: Separated components for data loading, model definition, training, and utilities
- **Hydra Configuration**: Hierarchical configuration system for easy experiment management
- **GPU Support**: Optimized for CUDA-enabled GPU training with gradient accumulation
- **Docker Support**: Containerized setup with NVIDIA CUDA runtime for reproducible environments
- **Weights & Biases Integration**: Optional logging and experiment tracking
- **Flexible Model Support**: Easy integration with Hugging Face transformers

## Project Structure

```
gpu-training-pipeline/
├── train.py                 # Main training script
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration for GPU training
├── configs/                # Hydra configuration files
│   ├── config.yaml        # Main configuration
│   ├── data/              # Dataset configurations (e.g., IMDB)
│   ├── model/             # Model configurations (e.g., DistilBERT)
│   ├── trainer/           # Training configurations (CPU/GPU)
│   └── task/              # Task-specific configurations
├── scripts/               # Utility scripts
│   └── train_local.sh    # Local training script
└── src/                   # Source code
    ├── dataloaders/      # Data loading and preprocessing
    ├── models/           # Model architectures
    ├── trainers/         # Training loop implementation
    └── utils/            # Utility functions (logging, seeding)
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Hydra-core
- Accelerate
- Optional: Weights & Biases for experiment tracking

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Run training with default configuration:
```bash
python train.py
```

### Custom Configuration

Override specific parameters:
```bash
python train.py model=distilbert data=imdb trainer=gpu trainer.epochs=5
```

### Docker Training

Build and run with Docker:
```bash
docker build -t gpu-training-pipeline .
docker run --gpus all gpu-training-pipeline
```

## Configuration

The pipeline uses Hydra for hierarchical configuration management. Key configuration groups:

- **model**: Model architecture (e.g., DistilBERT, BERT, RoBERTa)
- **data**: Dataset configuration (name, batch size, preprocessing)
- **trainer**: Training parameters (learning rate, epochs, device)
- **task**: Task-specific settings (classification, regression, etc.)

Example configuration override:
```bash
python train.py \
    model.model_name=bert-base-uncased \
    data.batch_size=32 \
    trainer.learning_rate=2e-5 \
    trainer.epochs=3
```

## Components

### Data Loaders
- Text classification data loading with tokenization
- Support for Hugging Face datasets
- Configurable batch size and preprocessing

### Models
- Base model abstraction
- Text classification model with customizable heads
- Support for freezing base model layers

### Trainers
- Training loop with evaluation
- Gradient accumulation support
- Learning rate scheduling with warmup
- Checkpointing and logging

### Utilities
- Deterministic seeding for reproducibility
- Weights & Biases integration
- Rich console output

## Extending the Pipeline

### Adding a New Dataset
1. Create a new YAML file in `configs/data/`
2. Specify dataset name, fields, and preprocessing parameters

### Adding a New Model
1. Create a new YAML file in `configs/model/`
2. Optionally extend base model classes in `src/models/`

### Custom Training Logic
Extend the `Trainer` class in `src/trainers/trainer.py` to implement custom training procedures.

## License

See the main repository for license information.
