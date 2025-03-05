import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from monai.metrics import DiceMetric
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import random

from unet import UNet
from dataset import NuInSegDataset

def arguments_parser():
    parser = argparse.ArgumentParser(description="Training script for segmentation")
    # Adding arguments for command line execution
    parser.add_argument('--data_dir', type=str, default='/home/ultrai/datasets/NuInsSeg', help='Path to the dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size for training and evaluation')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--val_size', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--test_size', type=float, default=0.1, help='Fraction of data to use for testing')
    parser.add_argument('--log_dir', type=str, default='./runs/unet_experiment', help='Directory to store TensorBoard logs')
    parser.add_argument('--resume', type=str, default='', help='Path to the checkpoint to resume training')
    parser.add_argument('--save_every', type=int, default=1, help='Save model checkpoint every n epochs')
    parser.add_argument('--visualize_every', type=int, default=1, help='Log predictions to TensorBoard every n epochs')
    return parser.parse_args()

def train_one_epoch(model, dataloader, criterion, optimizer, device, writer, epoch):
    model.train()
    running_loss = 0.0
    train_dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    for images, masks in tqdm(dataloader, desc='Training Steps', leave=False):
        images, masks = images.to(device), masks.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Calculate loss
        loss = criterion(outputs, masks)
        running_loss += loss.item()
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        # Calculate dice metric
        train_dice_metric(y_pred=outputs > 0.5, y=masks)
    
    train_dice = train_dice_metric.aggregate().item()
    train_dice_metric.reset()
    
    # Log the loss and dice metric to TensorBoard
    writer.add_scalar('Train/Loss', running_loss / len(dataloader), epoch)
    writer.add_scalar('Train/Dice', train_dice, epoch)
    print(f"Train Loss: {running_loss / len(dataloader):.4f}, Train Dice: {train_dice:.4f}")

def validate(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    val_loss = 0.0
    val_dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation Steps', leave=False):
            images, masks = images.to(device), masks.to(device)
            # Forward pass
            outputs = model(images)
            # Calculate loss
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            # Calculate dice metric
            val_dice_metric(y_pred=outputs > 0.5, y=masks)
    
    val_dice = val_dice_metric.aggregate().item()
    val_dice_metric.reset()
    
    # Log the loss and dice metric to TensorBoard
    writer.add_scalar('Val/Loss', val_loss / len(dataloader), epoch)
    writer.add_scalar('Val/Dice', val_dice, epoch)
    print(f"Val Loss: {val_loss / len(dataloader):.4f}, Val Dice: {val_dice:.4f}")
    
    return val_dice

def visualize_predictions(model, dataloader, device, writer, epoch):
    model.eval()
    
    images_list, masks_list, pred_masks_list = [], [], []
    for _ in range(4):  # Select 8 images
        random_batch = random.choice(list(dataloader))
        images, masks = random_batch
        idx = random.randint(0, images.shape[0] - 1)
        images_list.append(images[idx])
        masks_list.append(masks[idx])
        
    images = torch.stack(images_list).to(device)
    masks = torch.stack(masks_list).to(device)
    
    with torch.no_grad():
        outputs = torch.sigmoid(model(images))
        pred_masks = (outputs > 0.5).float()
    
    # Convert single channel mask and predicted mask to 3 channels
    images = images.detach().cpu()
    masks = masks.repeat(1, 3, 1, 1).detach().cpu()
    pred_masks = pred_masks.repeat(1, 3, 1, 1).detach().cpu()
    
    # Prepare a list to hold the images, masks, and predictions for each row
    combined = []
    for i in range(len(images_list)):
        combined.append(images[i].unsqueeze(0))         # Image
        combined.append(masks[i].unsqueeze(0))      # Mask
        combined.append(pred_masks[i].unsqueeze(0)) # Predicted mask

    # Concatenate the list into a single tensor
    combined = torch.cat(combined, dim=0)
    
    # Create the grid with 3 images per row (img, mask, pred)
    grid = vutils.make_grid(combined, nrow=3)

    writer.add_image(f'Predictions/epoch_{epoch}', grid, epoch)

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_dice = checkpoint['best_val_dice']
        print(f"Resuming training from epoch {epoch}, Best Validation Dice: {best_val_dice:.4f}")
        return model, optimizer, epoch, best_val_dice
    else:
        print("No checkpoint found, starting from scratch.")
        return model, optimizer, 0, 0.0

def main():
    args = arguments_parser()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Define transformations
    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Resize((512, 512)),
    ])
    
    # Create datasets for train, validation, and test sets
    train_dataset = NuInSegDataset(args.data_dir, train=True, val_size=args.val_size, test_size=args.test_size, transform=transform)
    val_dataset = NuInSegDataset(args.data_dir, train=False, val_size=args.val_size, test_size=args.test_size, transform=transform)
    test_dataset = NuInSegDataset(args.data_dir, train=False, val_size=args.val_size, test_size=args.test_size, transform=transform)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Initialize the model, loss function, and optimizer
    model = UNet(n_channels=3, n_classes=1).to(device) # Adjust channels as per dataset

    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with logits loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Training loop
    ckpt = args.resume if args.resume else os.path.join(args.log_dir, 'last_unet.pth')
    model, optimizer, start_epoch, best_val_dice = load_checkpoint(model, optimizer, ckpt)
    best_val_dice = 0.0
    for epoch in tqdm(range(start_epoch, args.epochs), desc='Epoch'):
        train_one_epoch(model, train_dataloader, criterion, optimizer, device, writer, epoch)
        val_dice = validate(model, val_dataloader, criterion, device, writer, epoch)

        # Log predictions to TensorBoard
        if epoch % args.visualize_every == 0:
            visualize_predictions(model, val_dataloader, device, writer, epoch)

        # Save the model if validation dice is better
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
            }, os.path.join(args.log_dir, 'best_unet.pth'))
        
        # Save model checkpoint every n epochs
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
            }, os.path.join(args.log_dir, 'last_unet.pth'))

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_dice': best_val_dice,
    }, os.path.join(args.log_dir, 'final_unet.pth'))

if __name__ == "__main__":
    main()