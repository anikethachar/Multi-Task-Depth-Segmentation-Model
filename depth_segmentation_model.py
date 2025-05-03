import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ----------------------------
# 1. Multi-Task Model
# ----------------------------
class MultiTaskModel(nn.Module):
    def __init__(self, backbone='mobilenet_v3_small', num_classes=21):
        super().__init__()
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.encoder = base_model.features

        # Depth decoder
        self.depth_decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2),
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        )

        # Segmentation decoder
        self.segmentation_decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        features = self.encoder(x)
        depth = self.depth_decoder(features)
        segmentation = self.segmentation_decoder(features)
        return depth, segmentation


# ----------------------------
# 2. Multi-Task Loss
# ----------------------------
def multitask_loss(pred_depth, true_depth, pred_seg, true_seg):
    true_depth_resized = nn.functional.interpolate(true_depth, size=pred_depth.shape[2:], mode='bilinear', align_corners=False)
    true_seg_resized = nn.functional.interpolate(true_seg.unsqueeze(1).float(), size=pred_seg.shape[2:], mode='nearest').squeeze(1).long()

    depth_loss = nn.L1Loss()(pred_depth, true_depth_resized)
    seg_loss = nn.CrossEntropyLoss()(pred_seg, true_seg_resized)
    return depth_loss + seg_loss, depth_loss, seg_loss


# ----------------------------
# 3. Custom Dataset
# ----------------------------
class DepthSegDataset(Dataset):
    def __init__(self, root_dir, transform=None, input_size=(256, 256)):
        self.image_paths = []
        self.depth_paths = []
        self.seg_paths = []
        self.input_size = input_size

        subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
        for sub in subfolders:
            photo_files = sorted(glob.glob(os.path.join(sub, "photo", "*.jpg")))
            depth_files = sorted(glob.glob(os.path.join(sub, "depth", "*.png")))
            seg_files = sorted(glob.glob(os.path.join(sub, "instance", "*.png")))

            if not (len(photo_files) == len(depth_files) == len(seg_files)):
                raise ValueError(f"Mismatch in number of files in {sub}")

            self.image_paths.extend(photo_files)
            self.depth_paths.extend(depth_files)
            self.seg_paths.extend(seg_files)

        self.img_transform = transform or transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])
        self.resize = transforms.Resize(self.input_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.img_transform(img)

        depth = Image.open(self.depth_paths[idx])
        depth_np = np.array(depth).astype(np.float32) / 1000.0
        depth_np = np.clip(depth_np, 0, 10)
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)

        seg = Image.open(self.seg_paths[idx])
        seg_resized = self.resize(seg)
        seg_tensor = torch.from_numpy(np.array(seg_resized)).long()

        return img, depth_tensor, seg_tensor


# ----------------------------
# 4. Training Function
# ----------------------------
def train(model, dataloader, optimizer, device, scaler):
    model.train()
    total_loss = 0

    for images, depths, segs in tqdm(dataloader, desc="Training"):
        images, depths, segs = images.to(device), depths.to(device), segs.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred_depths, pred_segs = model(images)
            loss, d_loss, s_loss = multitask_loss(pred_depths, depths, pred_segs, segs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Avg Loss: {avg_loss:.4f} | Depth Loss: {d_loss.item():.4f} | Seg Loss: {s_loss.item():.4f}")
    return avg_loss


# ----------------------------
# 5. Find Max Segmentation Label
# ----------------------------
def find_max_seg_label(dataset):
    max_label = 0
    for _, _, seg in tqdm(dataset, desc="Scanning labels"):
        current_max = seg.max().item()
        if current_max > max_label:
            max_label = current_max
    return max_label


# ----------------------------
# 6. Get Latest Checkpoint
# ----------------------------
def get_latest_checkpoint():
    checkpoints = glob.glob("checkpoint_epoch_*.pth")
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return checkpoints[-1]


# ----------------------------
# 7. Main Entry
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--input_size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DepthSegDataset(root_dir=args.data_dir, input_size=tuple(args.input_size))
    max_class = find_max_seg_label(dataset)
    print(f"Detected max segmentation label: {max_class}")

    model = MultiTaskModel(num_classes=max_class + 1).to(device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0

    # Load from checkpoint
    resume_path = args.resume or get_latest_checkpoint()
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed training from epoch {start_epoch} using {resume_path}")

    # Create a folder to store checkpoints
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        avg_loss = train(model, dataloader, optimizer, device, scaler)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    # Save the final model
    final_model_path = "multitask_model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved at {final_model_path}")


if __name__ == '__main__':
    main()
