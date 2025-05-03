# Multi-Task Depth Estimation and Semantic Segmentation Model

This repository implements a multi-task deep learning model designed for depth estimation and semantic segmentation. The model uses a MobileNetV3 backbone to ensure an efficient, lightweight architecture. It simultaneously predicts depth maps and segmentation labels from input images. The implementation is built with PyTorch, and it is designed for training on custom datasets containing RGB images, depth maps, and segmentation masks.

## Key Features

- **Multi-Task Learning**: Simultaneously predicts depth maps and semantic segmentation labels from a single input image, enhancing performance through shared feature extraction.
- **Efficient Backbone**: Uses MobileNetV3 for a lightweight architecture suitable for resource-constrained environments.
- **Custom Dataset Support**: Compatible with datasets containing RGB images, depth maps, and segmentation masks.
- **Mixed Precision Training**: Leverages PyTorch's AMP to speed up training and reduce memory usage.
- **Model Checkpointing**: Automatically saves checkpoints after each epoch for easy recovery.

## Requirements

- Python 3.x
- PyTorch 1.8+ (with CUDA support if using GPU)
- torchvision
- numpy
- tqdm
- Pillow (PIL)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Directory Structure
data/
    ├── folder1/
    │   ├── photo/          # RGB images (.jpg)
    │   ├── depth/          # Depth maps (.png, values scaled by 1000)
    │   └── instance/       # Segmentation masks (.png with integer class labels)
    ├── folder2/
    │   ├── photo/
    │   ├── depth/
    │   └── instance/
    └── ...

## Training
```bash
python depth_segmentation_model.py \
    --data_dir <path_to_data> \
    --epochs <num_epochs> \
    --batch_size <batch_size> \
    --lr <learning_rate> \
    --input_size <height> <width> \
    --num_workers <num_workers>
```

Example Command
```bash
python depth_segmentation_model.py \
    --data_dir data \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-4 \
    --input_size 256 256 \
    --num_workers 4
```

Resume Training
```bash
python depth_segmentation_model.py ... --resume checkpoints/checkpoint_epoch_10.pth
```
The final trained model is saved as multitask_model_final.pth for inference.















