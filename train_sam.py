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
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import  DiceCELoss

from transformers import SamModel, SamProcessor

from dataset import NuInSegDataset
from arg_parser import arguments_parser

import matplotlib.pyplot as plt

def build_grid_points(mask_np, grid_step=32):
    H, W = mask_np.shape
    ys = np.arange(0, H, grid_step)
    xs = np.arange(0, W, grid_step)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    grid_points = np.stack([yy.flatten(), xx.flatten()], axis=-1)
    point_labels = mask_np[grid_points[:, 0], grid_points[:, 1]]
    point_labels[point_labels == 0] = -1
    return grid_points.astype(np.float32), point_labels.astype(np.int32)

def generate_random_point_prompts(mask_np, num_pos_points=1, f=1):
    labeled_mask, num_instances = ndimage.label(mask_np)   # [H, W] -> [H, W]

    pos_points = []
    for i in range(1, num_instances + 1):
        y_coords, x_coords = np.where(labeled_mask == i)
        points = np.column_stack((y_coords, x_coords))
        
        # Always include the center of the instance
        center_y, center_x = np.mean(points, axis=0).astype(int)
        center_point = np.array([[center_y, center_x]])
        
        # Select other points randomly
        if num_pos_points > 1:
            # Ensure we don't select the center point again
            points = points[(points[:, 0] != center_y) | (points[:, 1] != center_x)]
            # Randomly select points from the remaining points
            selected_indices = np.random.choice(len(points), size=min(len(points), num_pos_points - 1), replace=False)
            selected_points = points[selected_indices]
        else:
            selected_points = np.empty((0, 2), dtype=int)
        
        pos_points.append(np.vstack([center_point, selected_points]))
    pos_points = np.vstack(pos_points)  # Combine all selected points
    
    # Select background points where mask_np == 0 (background area)
    neg_points = np.argwhere(mask_np == 0)  # Get indices of background pixels
    num_neg_points = len(pos_points) * f
    selected_background_points = neg_points[np.random.choice(neg_points.shape[0], size=num_neg_points, replace=False)]

    # Combine nuclei centers and background points
    point_prompts = np.vstack([pos_points, selected_background_points])
    point_labels = np.array([1] * len(pos_points) + [-1] * num_neg_points)  # 1 for nuclei centers, -1 for background points

    return point_prompts.astype(np.float32), point_labels.astype(np.int32)

def generate_bbx_prompts(mask_np):
    labeled_mask, num_instances = ndimage.label(mask_np)
    bboxes = []
    for i in range(1, num_instances + 1):
        ys, xs = np.where(labeled_mask == i)
        if ys.size == 0 or xs.size == 0:
            continue
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        bboxes.append([x0, y0, x1, y1])
    return np.array(bboxes).astype(np.float32)


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

def visualize(image, gt_mask, pred_mask, save_dir, file_name, points=None, point_labels=None, bbxes=None, alpha=0.5):
    image = image.permute(1, 2, 0).cpu().numpy()
    gt_mask = gt_mask.squeeze().cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))  # Adjusted figure size for better legibility on paper

    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Image", fontsize=16)
    plt.axis("off")

    # Ground truth mask
    plt.subplot(1, 4, 2)
    plt.imshow(image)
    plt.imshow(gt_mask, alpha=alpha)
    plt.title("GT Mask", fontsize=16)
    plt.axis("off")

    # Image with grid points
    if points is not None:
        plt.subplot(1, 4, 3)
        plt.imshow(image)
        plt.scatter(points[:, 1], points[:, 0], c=point_labels, s=1, cmap='cool')
        plt.title("Prompts", fontsize=16)
        plt.axis("off")
    
    if bbxes is not None:
        plt.subplot(1, 4, 3)
        plt.imshow(image)
        for bbx in bbxes:
            plt.gca().add_patch(plt.Rectangle((bbx[0], bbx[1]), bbx[2]-bbx[0], bbx[3]-bbx[1], edgecolor='red', facecolor='none'))
        plt.title("Bounding Boxes", fontsize=16)
        plt.axis("off")

    # Predicted mask
    plt.subplot(1, 4, 4)
    plt.imshow(image)
    plt.imshow(pred_mask, alpha=alpha)
    plt.title("Predicted Mask", fontsize=16)
    plt.axis("off")

    plt.tight_layout()
    name = file_name.replace(".png", ".svg")
    plt.savefig(f"{save_dir}/{name}", bbox_inches='tight', format='svg', dpi=500)
    plt.close()
    return None

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


def train_one_epoch(model, processor, dataloader, criterion_bce, criterion_dce, optimizer, device, 
                    writer, epoch, grid_step=8, prompt_type='grid', num_pos_points=6):
    epoch_loss = 0.0

    for batch_idx, (images, masks, image_path) in enumerate(tqdm(dataloader, 
                                    desc=f"Train Epoch {epoch}", leave=False)):
        # if images.shape[0] != 1:
        #     raise ValueError("This example code assumes batch_size=1 for multi-point prompts.")

        # Determine the number of instances in the mask
        num_instances = ndimage.label(masks.squeeze().numpy())[1]
        if num_instances > 100:
            # Skip due to memory constraints
            print(f"Skipping image {image_path[0]} due to too many instances: {num_instances}")
            continue
        
        # Generate prompts
        points, point_labels, bbxes = None, None, None
        if prompt_type == "grid":
            points, point_labels = build_grid_points(masks.squeeze().numpy(), grid_step=grid_step)
        elif prompt_type == "random":
            points, point_labels = generate_random_point_prompts(masks.squeeze().numpy(), num_pos_points=num_pos_points, f=3)
        elif prompt_type == "bbx":
            bbxes = generate_bbx_prompts(masks.squeeze().numpy())
        else:
            raise ValueError("Unsupported prompt type. Choose 'grid', 'random', or 'bbx'.")

        #  Skip if no prompts are provided
        if points is None and bbxes is None:
            continue
        
        encoded = processor(
            images=[images.squeeze(0).permute(1, 2, 0).numpy()],
            input_points=[points.tolist()] if points is not None else None,
            input_labels=[point_labels.tolist()] if point_labels is not None else None,
            input_boxes=[bbxes.tolist()] if bbxes is not None else None,
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**encoded, multimask_output=False)
        pred_masks = outputs.pred_masks  # [1, 1, 256, 256]
        
        pred_masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            encoded["original_sizes"],
            encoded["reshaped_input_sizes"],
            binarize=False
        )[0].float()    # [N, 1, H, W]

        # Combine all instances into one in case of bounding box prompts
        pred_masks = pred_masks.max(dim=0, keepdim=True).values # [N, 1, H, W] -> [1, 1, H, W]

        d_loss = criterion_dce(pred_masks, masks)      
        bce_loss = criterion_bce(pred_masks, masks)
        # loss = 0.5 * d_loss + 0.5 * bce_loss
        loss = d_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        torch.cuda.empty_cache()

    epoch_loss /= len(dataloader)
    print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}")
    writer.add_scalar('Train/Loss', epoch_loss, epoch)


def validate(model, processor, dataloader, device, writer, epoch, model_name, grid_step=8, vis_dir=None, prompt_type="grid", num_pos_points=6):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False)

    with torch.no_grad():
        for idx, (image, mask, image_path) in enumerate(tqdm(dataloader, desc="Zero-shot Testing")):
            if image.shape[0] != 1:
                raise ValueError("Batch size must be 1")

            image = image.to(device)        # [1, 3, H, W]

            img_np = image[0].permute(1, 2, 0).cpu().numpy()
            mask_np = mask[0, 0].numpy()    # [1, 1, H, W] -> [H, W]

            # Generate prompts
            points, point_labels, bbxes = None, None, None
            if prompt_type == "grid":
                points, point_labels = build_grid_points(mask_np, grid_step=grid_step)
            elif prompt_type == "random":
                points, point_labels = generate_random_point_prompts(mask_np, num_pos_points=num_pos_points, f=3)
            elif prompt_type == "bbx":
                bbxes = generate_bbx_prompts(mask_np)
            else:
                raise ValueError("Unsupported prompt type. Choose 'grid', 'random', or 'bbx'.")

            #  Skip if no prompts are provided
            if points is None and bbxes is None:
                continue
            
            encoded = processor(
                images=[img_np],
                input_points=[points.tolist()] if points is not None else None,
                input_labels=[point_labels.tolist()] if point_labels is not None else None,
                input_boxes=[bbxes.tolist()] if bbxes is not None else None,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**encoded, multimask_output=False)
            
            pred_masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                encoded["original_sizes"].cpu(),
                encoded["reshaped_input_sizes"].cpu(),
            )[0].float()    # [1, 1, H, W]

            # Combine the predicted masks into a single binary mask. This is especially necessary for bounding box prompts
            pred_masks = torch.any(pred_masks, dim=0, keepdim=True).float()  # [N, 1, H, W] -> [1, 1, H, W]

            dice_metric(y_pred=pred_masks, y=mask)
            hd95_metric(y_pred=pred_masks, y=mask)

            if vis_dir:
                visualize(image[0], mask[0], pred_masks[0], vis_dir+'/'+model_name+'/'+prompt_type, image_path[0], points, point_labels, bbxes)

    avg_dice = dice_metric.aggregate().item() * 100
    avg_hd95 = hd95_metric.aggregate().item()

    print(f"Average Zero-shot Dice Score: {avg_dice:.2f}")
    print(f"Average Zero-shot HD95 Score: {avg_hd95:.2f}")    
    return avg_dice, avg_hd95

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

    if args.model_name.lower() == "sam":
        ckpt = "facebook/sam-vit-huge"
    elif args.model_name.lower() == "medsam":
        ckpt = "flaviagiammarino/medsam-vit-base"
    else:
        ckpt = "facebook/sam-vit-base"
    
    print(f"Using model weights from {ckpt}")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge" , do_rescale=False, do_resize=False)


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
    log_dir = os.path.join(args.log_dir, args.model_name, args.prompt_type)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    for epoch in range(args.epochs):
        train_one_epoch(
            model, processor, train_dataloader, criterion_bce, criterion_dce, optimizer, device, 
            writer, epoch, grid_step=32, prompt_type=args.prompt_type, num_pos_points=args.num_pos_points
        )

        val_dice, _ = validate(
            model, processor, val_dataloader, device, writer, epoch, args.model_name,
            grid_step=32, prompt_type=args.prompt_type, num_pos_points=args.num_pos_points
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_model(model, optimizer, epoch, best_val_dice,
                       os.path.join(log_dir, f'best_sam_best.pth'))

        if epoch % args.save_every == 0:
            save_model(model, optimizer, epoch, best_val_dice,
                       os.path.join(log_dir, f'sam_last.pth'))

    writer.close()
    save_model(model, optimizer, args.epochs - 1, best_val_dice,
               os.path.join(log_dir, 'last_sam.pth'))


if __name__ == "__main__":
    main()