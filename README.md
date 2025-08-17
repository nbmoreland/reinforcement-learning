# Deep Learning & Reinforcement Learning Repository

A collection of PyTorch implementations for image classification using CNNs and reinforcement learning with Q-learning. This repository demonstrates various deep learning techniques including basic CNNs, regularization methods, and transfer learning approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Models](#models)
  - [Basic CNN](#basic-cnn)
  - [Advanced Convolution Network](#advanced-convolution-network)
  - [Regularized CNN](#regularized-cnn)
  - [Transfer Learning](#transfer-learning)
  - [Q-Learning Agent](#q-learning-agent)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Monitoring](#monitoring)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Author](#author)

## 🎯 Overview

This repository contains implementations of various neural network architectures for the Food101 dataset classification task (101 food categories) and a Q-learning implementation for grid-world navigation. The project explores different approaches to improve model performance through architectural changes, regularization techniques, and transfer learning.

### Key Features

- Multiple CNN architectures with increasing complexity
- Dropout and data augmentation for regularization
- Transfer learning framework for leveraging pre-trained models
- Q-learning implementation for reinforcement learning tasks
- TensorBoard integration for real-time training visualization
- Modular code structure for easy experimentation

## 📁 Project Structure

```
reinforcement-learning/
│
├── basic_cnn.py           # Basic CNN with simple architecture
├── convolution_net.py     # Advanced CNN with global average pooling
├── regularization.py      # CNN with extensive regularization techniques
├── transfer_learning.py   # Transfer learning implementation
├── q_learning.py          # Q-learning for grid-world environments
├── helper.py              # Utility functions for metrics and accuracy
├── CLAUDE.md             # Documentation for Claude Code AI assistant
└── README.md             # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM (for dataset loading)
- 10GB+ free disk space (for dataset download)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/reinforcement-learning.git
cd reinforcement-learning
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm tensorboard numpy
```

For CPU-only installation:
```bash
pip install torch torchvision torchaudio
pip install tqdm tensorboard numpy
```

## 🤖 Models

### Basic CNN
**File:** `basic_cnn.py`

A straightforward CNN implementation with:
- 2 convolutional layers (3→6→16 channels)
- MaxPooling after each conv layer
- 3 fully connected layers (7744→120→84→101)
- ReLU activations throughout
- Input size: 100x100 images
- Best for: Learning CNN basics, baseline performance

**Architecture:**
```
Input (3x100x100) → Conv(6) → ReLU → MaxPool → Conv(16) → ReLU → MaxPool 
→ Flatten → FC(120) → FC(84) → FC(101) → Output
```

### Advanced Convolution Network
**File:** `convolution_net.py`

Enhanced CNN with modern techniques:
- Global average pooling instead of flatten
- Optional dropout (20% rate)
- Larger input size: 224x224
- Class-wise convolution for final layer
- Best for: Better generalization, reduced overfitting

**Architecture:**
```
Input (3x224x224) → [Dropout] → Conv(6) → ReLU → Conv(16) → ReLU 
→ Conv(101, 1x1) → Global Average Pool → Output
```

### Regularized CNN
**File:** `regularization.py`

Heavily regularized model to prevent overfitting:
- Multiple dropout layers throughout the network
- Extensive data augmentation:
  - RandAugment
  - Random rotation (±30°)
  - Random cropping
  - Horizontal flipping
- 80/20 train/validation split (vs 95/5 in others)
- Best for: Small datasets, preventing overfitting

**Data Augmentation Pipeline:**
```python
- RandAugment (automatic augmentation policies)
- RandomRotation(30)
- RandomResizedCrop(100)
- RandomHorizontalFlip()
```

### Transfer Learning
**File:** `transfer_learning.py`

Framework for fine-tuning pre-trained models:
- Supports any torchvision pre-trained model
- Learning rate scheduling
- Model checkpointing for best weights
- Separate train/validate functions
- Best for: Achieving highest accuracy, limited training data

**Features:**
- Automatic best model selection
- Time tracking for training duration
- Flexible architecture (can use ResNet, VGG, EfficientNet, etc.)

### Q-Learning Agent
**File:** `q_learning.py`

Reinforcement learning implementation for grid-world navigation:
- Epsilon-greedy exploration strategy
- Stochastic environment (10% chance of unintended turns)
- Temporal difference learning
- Supports arbitrary grid layouts

**Components:**
- `Action`: Represents movement directions (up, down, left, right)
- `Agent`: Executes actions and tracks position
- `Environment`: Grid world with obstacles and rewards
- Q-value update with learning rate decay

## 💻 Usage

### Training Image Classification Models

1. **Basic CNN Training:**
```bash
python basic_cnn.py
```

2. **Advanced CNN with Global Pooling:**
```bash
python convolution_net.py
```

3. **CNN with Regularization:**
```bash
python regularization.py
```

4. **Transfer Learning:**
```bash
python transfer_learning.py
```

### Running Q-Learning

1. Create an environment file (e.g., `grid_world.csv`):
```
I,.,.,.,10
.,X,X,.,.,
.,.,X,.,.
.,.,.,.,-10
```
Where:
- `I` = Initial position
- `X` = Obstacle
- `.` = Empty cell
- Numbers = Terminal states with rewards

2. Modify `q_learning.py` to load your environment:
```python
env = Environment('grid_world.csv', ntr=-0.04)  # ntr = non-terminal reward
```

3. Run the Q-learning agent:
```bash
python q_learning.py
```

## 📊 Dataset

### Food101 Dataset
- **Size:** 101,000 images (1,000 per class)
- **Classes:** 101 food categories
- **Download:** Automatic on first run
- **Storage:** `./food101data/` directory
- **Split:** 
  - Training: 75,750 images (750 per class)
  - Testing: 25,250 images (250 per class)

The dataset includes diverse food categories such as pizza, sushi, hamburger, etc. Images vary in quality, lighting, and presentation, making it a challenging real-world dataset.

## 📈 Training

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 100 | Number of samples per batch |
| Learning Rate | 0.01 | Adam optimizer learning rate |
| Epochs | 50 | Training iterations |
| Optimizer | Adam | Adaptive learning rate optimizer |
| Loss Function | CrossEntropyLoss | Multi-class classification loss |

### Training Tips

1. **GPU Usage:** Models automatically detect and use CUDA if available
2. **Memory Management:** Reduce batch size if encountering OOM errors
3. **Early Stopping:** Monitor validation loss and stop if it plateaus
4. **Learning Rate:** Consider reducing if loss becomes unstable

## 📊 Monitoring

### TensorBoard Integration

All models log metrics to TensorBoard:

1. Start TensorBoard:
```bash
tensorboard --logdir=runs/
```

2. Open browser at `http://localhost:6006`

3. Available metrics:
   - Training Loss
   - Training Accuracy
   - Validation Loss (transfer_learning.py)
   - Validation Accuracy (transfer_learning.py)

### Logged Directories
- `runs/cnn/` - Basic CNN metrics
- `runs/convolution_net/` - Advanced CNN metrics
- `runs/regularization/` - Regularized model metrics
- `runs/transfer_learning/` - Transfer learning metrics

## 📊 Results

### Expected Performance

| Model | Top-1 Accuracy | Training Time | Parameters |
|-------|---------------|---------------|------------|
| Basic CNN | ~15-20% | ~2 hours | ~950K |
| Advanced CNN | ~20-25% | ~3 hours | ~11K |
| Regularized CNN | ~18-22% | ~4 hours | ~950K |
| Transfer Learning | ~60-70%* | ~1 hour | Varies |

*With pre-trained ResNet or similar

### Interpreting Results

- **Low accuracy initially:** Food101 is challenging; even 20% is significant (random = 1%)
- **Overfitting signs:** Training accuracy >> validation accuracy
- **Underfitting signs:** Both accuracies remain low after many epochs
- **Good convergence:** Smooth loss decrease, accuracies gradually improving

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - Reduce batch size (e.g., 50 or 25)
   - Use smaller input dimensions
   - Enable gradient checkpointing

2. **Dataset Download Fails:**
   - Check internet connection
   - Manually download from [Food101 website](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
   - Place in `./food101data/`

3. **ImportError with tqdm.notebook:**
   - Replace `from tqdm.notebook import tqdm` with `from tqdm import tqdm`
   - Install notebook support: `pip install ipywidgets`

4. **Slow Training:**
   - Ensure CUDA is being used: Check for "cuda" in device output
   - Reduce image size for faster processing
   - Use data parallel for multi-GPU: `model = nn.DataParallel(model)`

5. **Poor Convergence:**
   - Try different learning rates (1e-3, 1e-4)
   - Implement learning rate scheduling
   - Check data augmentation isn't too aggressive

### Performance Optimization

1. **Enable Mixed Precision Training:**
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

2. **Use Multiple Workers for Data Loading:**
```python
DataLoader(..., num_workers=4, pin_memory=True)
```

3. **Gradient Accumulation for Large Batches:**
```python
accumulation_steps = 4
for i, (input, target) in enumerate(dataloader):
    loss = criterion(model(input), target)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- New model architectures
- Performance improvements
- Documentation updates
- Additional RL algorithms

### Development Guidelines

1. Follow existing code structure
2. Add docstrings for new functions
3. Update README for significant changes
4. Test on both CPU and GPU
5. Ensure TensorBoard logging works

## 👤 Author

**Nicholas Moreland**  
Date: 05/02/2024

---

*This project is part of coursework in deep learning and reinforcement learning. Feel free to use for educational purposes.*