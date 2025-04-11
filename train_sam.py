import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import scipy.ndimage as ndimage

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss

from transformers import SamModel, SamProcessor

from dataset import NuInSegDataset
from arg_parser import arguments_parser

import matplotlib.pyplot as plt

def get_nucleus_bboxes(binary_mask):
    """
    binary_mask: np.array of shape (H,W), with values 0 or 1
    returns: list of bounding boxes [ [x0,y0,x1,y1], [x0,y0,x1,y1], ... ]
    where x0 < x1 and y0 < y1
    """
    labeled, num_labels = ndimage.label(binary_mask) 
    bboxes = []
    for label_idx in range(1, num_labels + 1):
        ys, xs = np.where(labeled == label_idx)
        if ys.size == 0 or xs.size == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        # SAM typically uses (x0, y0, x1, y1)
        bboxes.append([float(x0), float(y0), float(x1), float(y1)])
    return bboxes


def get_nucleus_single_mask(binary_mask, labeled_mask, label_idx):
    """
    Return a single-object mask for the connected component 'label_idx'.
    binary_mask: (H,W) in {0,1}
    labeled_mask: (H,W) in {0..num_labels}, from ndimage.label
    label_idx: the integer label for the nucleus
    Output shape: (1,H,W)
    """
    single_mask = (labeled_mask == label_idx).astype(np.float32) #(H*W)
    return single_mask[None, ...]

def visualize_mask_and_prediction(image, gt_mask, pred_mask, alpha=0.5):
    """
    Visualize the original image, ground truth mask, and predicted mask.
    image: (H, W, 3) or (3, H, W)
    gt_mask: (H, W) or (1, H, W)
    pred_mask: (H, W) or (1, H, W)
    alpha: transparency for overlay
    """
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.detach().cpu().numpy()
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.detach().cpu().numpy()

    # If channel-first (C, H, W), convert to (H, W, C)
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = image.transpose(1,2,0)
    if gt_mask.ndim == 3 and gt_mask.shape[0] == 1:
        gt_mask = gt_mask[0]
    if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask[0]

    plt.figure()
    plt.imshow(image)
    plt.title("Original Image")
    plt.savefig("./plot/original_image.png")

    plt.figure()
    plt.imshow(image)
    plt.imshow(gt_mask, alpha=alpha)
    plt.title("Ground Truth Mask Overlay")
    plt.savefig("./plot/gt_mask_overlay.png")

    plt.figure()
    plt.imshow(image)
    plt.imshow(pred_mask, alpha=alpha)
    plt.title("Predicted Mask Overlay")
    plt.savefig("./plot/pred_mask_overlay.png")

def dice_loss(pred, target):
    """
    pred: [B, 1, H, W] after sigmoid, in [0..1]
    target: [B, 1, H, W] in {0,1}
    """
    smooth = 1e-6
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_score = (2.0 * intersection + smooth) / (pred_flat.sum() + 
                target_flat.sum() + smooth)
    return 1.0 - dice_score


def build_grid_points(mask_np, grid_step=8):
    """
    mask_np: (H, W) in {0, 1}
    grid_step: sampling step for the grid
    Returns:
      points: (num_points, 2), each row = [y, x]
      labels: (num_points,), each label = 0 or 1
    """
    H, W = mask_np.shape
    ys = np.arange(0, H, grid_step)
    xs = np.arange(0, W, grid_step)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    grid_points = np.stack([yy.flatten(), xx.flatten()], axis=-1)  # (N, 2)
    point_labels = mask_np[grid_points[:,0], grid_points[:,1]]  # in {0,1}
    grid_points = grid_points.astype(np.float32)  # Cast to float
    point_labels = point_labels.astype(np.int32) 
    return grid_points, point_labels

def train_one_epoch(model, processor, dataloader, criterion_bce, criterion_dce, optimizer, device, 
                    writer, epoch, grid_step=8):
    train_dice_metric = DiceMetric(include_background=False, reduction="mean")
    epoch_loss = 0.0

    for batch_idx, (images, masks) in enumerate(tqdm(dataloader, 
                                    desc=f"Train Epoch {epoch}", leave=False)):
        # if images.shape[0] != 1:
        #     raise ValueError("This example code assumes batch_size=1 for multi-point prompts.")

        images = images.to(device)  # shape [1, 3, H, W]
        masks = masks.to(device)    # shape [1, 1, H, W]

        img_np = images[0].permute(1,2,0).detach().cpu().numpy()  # shape (H, W, 3)
        mask_np = masks[0, 0].detach().cpu().numpy()              # shape (H, W)

        grid_points, point_labels = build_grid_points(mask_np, grid_step=grid_step)

        if len(grid_points) == 0:
            continue

        encoded_inputs = processor(
            images=[img_np],
            input_points=[grid_points.tolist()],     # Must be list of list of [y,x]
            input_labels=[point_labels.tolist()],    # Must be list of list of 0/1
            return_tensors="pt"
        ).to(device)

        outputs = model(**encoded_inputs, multimask_output=False)
        pred_masks = outputs.pred_masks  # shape [1, 1, 256, 256]

        # Ensure shape is [1, 1, H, W] for interpolation
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)
        elif pred_masks.dim()  == 5:
            pred_masks = pred_masks.squeeze(1) 

        H, W = mask_np.shape
        upscaled_mask = F.interpolate(
            pred_masks,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )  # shape [1, 1, H, W]

        pred_prob = torch.sigmoid(upscaled_mask)  
        d_loss = dice_loss(pred_prob, masks)        
        bce_loss = criterion_bce(pred_prob, masks)
        loss = 0.5 * d_loss + 0.5 * bce_loss
        # loss = d_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    print(f"[Epoch {epoch+1}] Train Dice Loss: {epoch_loss:.4f}")
    writer.add_scalar('Train/Loss', epoch_loss, epoch)


def validate(model, processor, dataloader, device, writer, epoch, grid_step=8):
    model.eval()
    val_dice = 0.0
    num_images = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Valid", leave=False):
            if images.shape[0] != 1:
                raise ValueError("This example code assumes batch_size=1 for \
                                 multi-point prompts.")

            images = images.to(device)  # [1, 3, H, W]
            masks = masks.to(device)    # [1, 1, H, W]

            img_np = images[0].permute(1,2,0).cpu().numpy()
            mask_np = masks[0, 0].cpu().numpy()

            grid_points, point_labels = build_grid_points(mask_np, 
                                                         grid_step=grid_step)
            if len(grid_points) == 0:
                continue

            encoded_inputs = processor(
                images=[img_np],
                input_points=[grid_points.tolist()],
                input_labels=[point_labels.tolist()],
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded_inputs, multimask_output=False)
            pred_masks = outputs.pred_masks  # [1, 1, 256, 256]

            # Ensure shape is [1, 1, H, W] for interpolation
            if pred_masks.dim() == 3:
                pred_masks = pred_masks.unsqueeze(1)
            elif pred_masks.dim()  == 5:
                pred_masks = pred_masks.squeeze(1)  

            H, W = mask_np.shape
            upscaled_mask = F.interpolate(
                pred_masks,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )
            pred_prob = torch.sigmoid(upscaled_mask)

            pred_bin = (pred_prob > 0.5).float()
            single_dice = 1.0 - dice_loss(pred_bin, masks).item()  
            val_dice += single_dice
            num_images += 1

    if num_images > 0:
        val_dice /= num_images
    print(f"Validation Dice Score: {val_dice:.4f}")
    writer.add_scalar('Validation/Dice', val_dice, epoch)
    return val_dice

def save_model(model, optimizer, epoch, best_val_dice, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_dice': best_val_dice,
    }, checkpoint_path)


def main():
    args = arguments_parser()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Basic transform (example)
    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Resize((1024, 1024)),
    ])

    # Datasets
    train_dataset = NuInSegDataset(
        data_dir=args.data_dir,
        train=True,
        val_size=args.val_size,
        test_size=args.test_size,
        transform=transform
    )
    val_dataset = NuInSegDataset(
        data_dir=args.data_dir,
        train=False,
        val_size=args.val_size,
        test_size=args.test_size,
        transform=transform
    )
    test_dataset = NuInSegDataset(
        data_dir=args.data_dir,
        train=False,
        val_size=args.val_size,
        test_size=args.test_size,
        transform=transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                                    drop_last=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                    drop_last=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                    drop_last=True, num_workers=2)

    # Initialize model + processor
    # model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    # processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    # model.load_state_dict(torch.load('/projects/ovcare/users/elahe_ranjbari/SAM/bmeg591_nuclei_segmentation/runs/sam_experiment/sam_e49.pth')['model_state_dict'])
    
    model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")

    print(model)

    # Freeze encoders, unfreeze mask decoder
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.prompt_encoder.parameters():
        param.requires_grad = False
    for name, param in model.mask_decoder.named_parameters():
        param.requires_grad = True

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dce = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    best_val_dice = 0.0
    model.to(device)
    model.train()
    for epoch in range(args.epochs):
        train_one_epoch(
            model, processor, train_dataloader, criterion_bce, criterion_dce, optimizer, device, 
            writer, epoch, grid_step=16  
        )

        val_dice = validate(
            model, processor, val_dataloader, device, writer, epoch,
            grid_step=16
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_model(model, optimizer, epoch, best_val_dice,
                       os.path.join(args.log_dir, f'best_sam_e{epoch}.pth'))

        if epoch % args.save_every == 0:
            save_model(model, optimizer, epoch, best_val_dice,
                       os.path.join(args.log_dir, f'sam_e{epoch}.pth'))

    writer.close()
    save_model(model, optimizer, args.epochs - 1, best_val_dice,
               os.path.join(args.log_dir, 'last_sam.pth'))


if __name__ == "__main__":
    main()