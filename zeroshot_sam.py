import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from monai.metrics import DiceMetric, HausdorffDistanceMetric
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
    point_labels[point_labels == 0] = -1
    return grid_points.astype(np.float32), point_labels.astype(np.int32)


def visualize(image, gt_mask, pred_mask, save_dir, idx, points=None, point_labels=None,alpha=0.5):
    image = image.permute(1, 2, 0).cpu().numpy()
    gt_mask = gt_mask.squeeze().cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(20, 5))

    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    # Ground truth mask
    plt.subplot(1, 4, 2)
    plt.imshow(image)
    plt.imshow(gt_mask, alpha=alpha)
    plt.title("GT Mask")
    plt.axis("off")

    # Image with grid points
    plt.subplot(1, 4, 3)
    plt.imshow(image)
    plt.scatter(points[:, 1], points[:, 0], c=point_labels, s=1, cmap='cool')
    plt.title("Prompts")
    plt.axis("off")

    # Predicted mask
    plt.subplot(1, 4, 4)
    plt.imshow(image)
    plt.imshow(pred_mask, alpha=alpha)
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/visualization_{idx}.png", bbox_inches='tight')
    plt.close()
    return None

def test_sam_zeroshot(model, processor, dataloader, device, grid_step=32, vis_dir=None):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False)

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

            dice_metric(y_pred=pred_bin, y=mask)
            hd95_metric(y_pred=pred_bin, y=mask)

            if vis_dir:
                visualize(image[0], mask[0], pred_masks[0], vis_dir, idx, points, point_labels)

    avg_dice = dice_metric.aggregate().item() * 100
    avg_hd95 = hd95_metric.aggregate().item()

    print(f"Average Zero-shot Dice Score: {avg_dice:.2f}")
    print(f"Average Zero-shot HD95 Score: {avg_hd95:.2f}")

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
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge" , do_rescale=False, do_resize=False)

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
