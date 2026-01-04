# Flax NNX ResNet-18 on ImageNet

## Executive Summary

This report presents the results of a ResNet-18 model trained on the ImageNet-1K
dataset using the new Flax NNX api. The model achieved a **Top-1 Accuracy of
69.0%** on the validation set with limited augmentation and standard 1-crop
validation. We carefully align settings to official training recipes, except for
data augmentations.

## Training Configuration

### Hyperparameters

- **Total Steps**: 450,000
- **Batch Size**: 256
- **Training Epochs**: ~90 epochs
- **Optimizer**: SGD with momentum
- **Data Augmentation**: RandomResizedCrop + Normalisation

### Data Pipeline

- **Training Data**: `data/in1k-train-*.ar` (ArrayRecord Format)
- **Validation Data**: `data/in1k-validation-*.ar` (ArrayRecord Format)

### Evaluation Schedule

- **Evaluation Frequency**: Every 10,000 steps
- **Logging Frequency**: Every 64 steps
- **Reported Metrics**: Top-1 Accuracy

## Performance Results

### Final Evaluation Metrics

| Metric             | Value     |
| ------------------ | --------- |
| **Top-1 Accuracy** | **69.0%** |

### Training Progress

- **Final Training Loss**: ~1.2
- **Final Training Top-1**: ~70-72%
- **Convergence**: Stable convergence observed throughout training

## Technical Implementation

### Key Components

1. **NNX Module System**: Modern Flax NNX API for module definition
2. **Logging**: nnlogging library for experiment tracking
3. **Mixed Precision**: bfloat16 for improved performance

## Comparison with Benchmarks

### ResNet-18 Standard Performance

| Source                  | Top-1     | Notes                   |
| ----------------------- | --------- | ----------------------- |
| **This Implementation** | **69.0%** | Flax NNX implementation |
| PyTorch baseline        | 69.8%     | Extra augmentations     |
| Common Benchmarks       | 68-71%    |                         |

## Reference

![imagenet-arrayrecord-scripts](https://codeberg.org/amas127/mlscripts/src/commit/40147809eccea716100c924e71ac4d0517ba0844/imagenet-1k-resize-flatbuffers-arrayrecord)

![torch-resnet-reproduction](https://torch.ch/blog/2016/02/04/resnets.html)
