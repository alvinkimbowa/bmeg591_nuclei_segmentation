import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from monai.metrics import DiceMetric
from transformers import SamModel, SamProcessor

from dataset import NuInSegDataset 
from arg_parser import arguments_parser  


def build_grid_points(mask_np, grid_step=32):
    H, W = mask_np.shape
    ys = np.arange(0, H, grid_step)
    xs = np.arange(0, W, grid_step)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    grid_points = np.stack([yy.flatten(), xx.flatten()], axis=-1)
    point_labels = mask_np[grid_points[:, 0], grid_points[:, 1]]
    return grid_points.astype(np.float32), point_labels.astype(np.int32)


def visualize(image, gt_mask, pred_mask, save_dir, idx, alpha=0.5):
    image = image.permute(1, 2, 0).cpu().numpy()
    gt_mask = gt_mask.squeeze().cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)

    plt.imsave(f"{save_dir}/image_{idx}.png", image)
    
    plt.figure()
    plt.imshow(image)
    plt.imshow(gt_mask, alpha=alpha)
    plt.title("GT Mask")
    plt.savefig(f"{save_dir}/gt_{idx}.png")
    plt.close()

    plt.figure()
    plt.imshow(image)
    plt.imshow(pred_mask, alpha=alpha)
    plt.title("Predicted Mask")
    plt.savefig(f"{save_dir}/pred_{idx}.png")
    plt.close()

def test_sam_zeroshot(model, processor, dataloader, device, grid_step=32, vis_dir=None):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    total_dice = 0.0
    num_samples = 0

    with torch.no_grad():
        for idx, (image, mask) in enumerate(tqdm(dataloader, desc="Zero-shot Testing")):
            if image.shape[0] != 1:
                raise ValueError("Batch size must be 1")

            image = image.to(device)        # [1, 3, H, W]
            mask = mask.to(device)          # [1, 1, H, W]

            img_np = image[0].permute(1, 2, 0).cpu().numpy()
            mask_np = mask[0, 0].cpu().numpy()

            grid_points, point_labels = build_grid_points(mask_np, grid_step=grid_step)

            if len(grid_points) == 0:
                continue

            encoded = processor(
                images=[img_np],
                input_points=[grid_points.tolist()],
                input_labels=[point_labels.tolist()],
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded, multimask_output=False)
            pred_mask = outputs.pred_masks  # [1, 1, 256, 256]

            if pred_mask.dim() == 3:
                pred_mask = pred_mask.unsqueeze(1)  # [1,1,H,W]
            elif pred_mask.dim() == 5:
                pred_mask = pred_mask.squeeze(1)

            H, W = mask_np.shape
            pred_up = F.interpolate(
                pred_mask,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )
            pred_prob = torch.sigmoid(pred_up)
            pred_bin = (pred_prob > 0.5).float()

            dice = 1.0 - dice_loss(pred_bin, mask).item()
            total_dice += dice
            num_samples += 1

            if vis_dir:
                visualize(image[0], mask[0], pred_bin[0], vis_dir, idx)

    avg_dice = total_dice / num_samples
    print(f"Average Zero-shot Dice Score: {avg_dice:.4f}")


def dice_loss(pred, target):
    smooth = 1e-6
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def main():
    args = arguments_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Resize((1024, 1024)),
    ])

    test_dataset = NuInSegDataset(
        data_dir=args.data_dir,
        train=False,
        val_size=args.val_size,
        test_size=args.test_size,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    test_sam_zeroshot(
        model=model,
        processor=processor,
        dataloader=test_loader,
        device=device,
        grid_step=32,
        vis_dir="./plot"  
    )

if __name__ == "__main__":
    main()
