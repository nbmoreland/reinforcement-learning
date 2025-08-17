# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a PyTorch-based reinforcement learning and deep learning repository focusing on image classification and Q-learning implementations. The codebase contains multiple neural network architectures for the Food101 dataset classification task and a Q-learning implementation for grid-based environments.

## Architecture and Structure

### Core Components

1. **Neural Network Models** - Multiple CNN architectures for Food101 classification:
   - `basic_cnn.py`: Basic CNN with sequential feature extraction and linear classifier
   - `convolution_net.py`: Advanced CNN with global average pooling and optional dropout
   - `regularization.py`: CNN with extensive dropout and data augmentation for regularization
   - `transfer_learning.py`: Framework for transfer learning with pre-trained models

2. **Reinforcement Learning**:
   - `q_learning.py`: Grid-world Q-learning implementation with Action, Agent, and Environment classes

3. **Utilities**:
   - `helper.py`: Contains `AverageMeter` for tracking metrics and `accuracy()` for computing top-k precision

### Model Architecture Patterns

All CNN models follow a consistent pattern:
- **Feature extraction**: Convolutional layers with ReLU activations and pooling
- **Classification head**: Linear layers mapping features to 101 Food101 classes
- **Training loop**: Uses Adam optimizer, CrossEntropyLoss, and TensorBoard logging

### Key Design Decisions

- Models use Food101 dataset (101 food categories) with 95/5 or 80/20 train/validation splits
- Batch size of 100 across all models
- Learning rate of 1e-2 with Adam optimizer
- TensorBoard integration for training visualization
- Data stored in `./food101data` directory

## Common Development Commands

Since this is a collection of standalone training scripts without a unified runner:

```bash
# Run individual training scripts (requires Jupyter environment for tqdm.notebook)
python basic_cnn.py         # Train basic CNN model
python convolution_net.py   # Train advanced CNN with global pooling
python regularization.py    # Train CNN with regularization techniques
python transfer_learning.py # Run transfer learning pipeline

# For Q-learning (requires environment data file)
python q_learning.py  # Run after setting up environment file

# Monitor training with TensorBoard
tensorboard --logdir=runs/
```

## Important Considerations

1. **Dependencies**: Scripts use PyTorch, torchvision, tqdm (notebook version), and tensorboard. Ensure these are installed.

2. **Dataset**: Food101 dataset will be automatically downloaded on first run to `./food101data`

3. **Device**: Scripts automatically detect and use CUDA if available, otherwise CPU

4. **Training Duration**: Each model trains for 50 epochs by default, which can be time-consuming

5. **Memory Requirements**: Food101 is a large dataset (101,000 images). Ensure sufficient memory for data loading.

6. **Q-Learning Environment**: The `q_learning.py` script expects a CSV file defining the grid world. The file format should contain:
   - 'X' for obstacles
   - 'I' for initial state
   - Numeric values for terminal states
   - '.' for empty cells

## Code Patterns to Follow

When modifying or extending the models:
- Maintain the existing class structure with `__init__()` and `forward()` methods
- Use consistent naming: `Food101` for model classes
- Follow the training loop pattern with AverageMeter for metrics tracking
- Add TensorBoard logging for new metrics
- Keep hyperparameters as module-level variables for easy tuning