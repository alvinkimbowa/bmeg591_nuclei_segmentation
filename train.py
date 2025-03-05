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

from unet import UNet
from dataset import NuInSegDataset

def arguments_parser():
    parser = argparse.ArgumentParser(description="Training script for segmentation")
    
    # Adding arguments for command line execution
    parser.add_argument('--data_dir', type=str, default='/home/ultrai/datasets/NuInsSeg', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--val_size', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--test_size', type=float, default=0.1, help='Fraction of data to use for testing')
    parser.add_argument('--log_dir', type=str, default='./runs/unet_experiment', help='Directory to store TensorBoard logs')


    return parser.parse_args()

def main(args):
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
    best_val_dice = 0.0
    best_epoch = 0
    for epoch in tqdm(range(args.epochs), desc='Epochs'):
        # Initialize Dice Metric
        train_dice_metric = DiceMetric(include_background=False, reduction="mean")
        val_dice_metric = DiceMetric(include_background=False, reduction="mean")

        model.train()
        running_loss = 0.0
        
        for images, masks in tqdm(train_dataloader, desc='steps', leave=False):
            images, masks = images.to(device), masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            train_dice_metric(y_pred=outputs, y=masks)

            running_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        train_dice = train_dice_metric.aggregate().item()
        train_dice_metric.reset()

        # Log the loss to TensorBoard
        writer.add_scalar('Loss/train', running_loss / len(train_dataloader), epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        print(f"Train Loss: {running_loss / len(train_dataloader)}")
        print(f"Train Dice: {train_dice}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_dataloader:
                images, masks = images.to(device), masks.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice_metric(y_pred=outputs, y=masks)

        val_dice = val_dice_metric.aggregate().item()
        val_dice_metric.reset()

        # Log the validation loss to TensorBoard
        writer.add_scalar('Loss/val', val_loss / len(val_dataloader), epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        print(f"Val Loss: {val_loss / len(val_dataloader)}")
        print(f"Val Dice: {val_dice}")

        # Save the model if validation dice is better
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_unet.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'last_unet.pth'))

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.log_dir, 'final_unet.pth'))

if __name__ == "__main__":
    # Parse the arguments
    args = arguments_parser()

    # Call the main function to start the process
    main(args)
