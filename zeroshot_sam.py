import os
import torch
import numpy as np
from scipy import ndimage
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

def test_sam_zeroshot(model, processor, dataloader, device, grid_step=32, vis_dir=None, prompt_type="grid"):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False)

    with torch.no_grad():
        for idx, (image, mask) in enumerate(tqdm(dataloader, desc="Zero-shot Testing")):
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
                points, point_labels = generate_random_point_prompts(mask_np)
            else:
                raise ValueError("Unsupported prompt type. Choose 'grid' or 'random'.")

            #  Skip if no prompts are provided
            if len(points) == 0 and len(point_labels) == 0 and len(bbxes) == 0:
                continue
            
            encoded = processor(
                images=[img_np],
                input_points=[points.tolist()] if points is not None else None,
                input_labels=[point_labels.tolist()] if point_labels is not None else None,
                input_boxes=[bbxes.tolist()] if bbxes is not None else None,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**encoded, multimask_output=False)
            
            pred_masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                encoded["original_sizes"].cpu(),
                encoded["reshaped_input_sizes"].cpu(),
            )[0].float()    # [1, 1, H, W]

            dice_metric(y_pred=pred_masks, y=mask)
            hd95_metric(y_pred=pred_masks, y=mask)

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
