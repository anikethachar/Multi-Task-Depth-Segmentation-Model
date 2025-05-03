# Multi-Task-Depth-Segmentation-Model
# Multi-Task Depth Segmentation Model

This project implements a multi-task deep learning model for depth estimation and semantic segmentation using a **MobileNetV3** backbone. The model simultaneously predicts depth maps and semantic segmentation labels from input images. The code is implemented in **PyTorch** and is designed for training on a custom dataset containing images, depth maps, and segmentation masks.

## Key Features

- **Multi-task learning**: Simultaneous prediction of depth maps and segmentation masks from a single input image.
- **MobileNetV3 Backbone**: Utilizes MobileNetV3 as the feature extractor to ensure efficient and lightweight model architecture.
- **Custom Dataset Support**: Compatible with custom datasets containing paired images, depth maps, and segmentation masks.
- **Mixed Precision Training**: Leveraging PyTorch's AMP (Automatic Mixed Precision) for faster training with reduced memory usage.
- **Checkpointing**: Save model checkpoints after each epoch for easy recovery and continued training.

## Requirements

- Python 3.x
- PyTorch 1.8+ (with CUDA support if using a GPU)
- torchvision
- numpy
- tqdm
- Pillow (PIL)

To install the necessary dependencies, you can create a virtual environment and install the required libraries using:

```bash
pip install -r requirements.txt
