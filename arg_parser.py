import argparse


def arguments_parser():
    parser = argparse.ArgumentParser(description="Training script for segmentation")
    # Adding arguments for command line execution
    parser.add_argument('--data_dir', type=str, default='./NuInsSeg', help='Path to the dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size for training and evaluation')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--val_size', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--test_size', type=float, default=0.1, help='Fraction of data to use for testing')
    parser.add_argument('--log_dir', type=str, default='./runs/sam_experiment', help='Directory to store TensorBoard logs')
    parser.add_argument('--resume', type=str, default='', help='Path to the checkpoint to resume training')
    parser.add_argument('--save_every', type=int, default=1, help='Save model checkpoint every n epochs')
    parser.add_argument('--visualize_every', type=int, default=1, help='Log predictions to TensorBoard every n epochs')
    parser.add_argument('--vis_dir', type=str, default='./plot', help='Directory to save visualizations')
    parser.add_argument('--model_name', type=str, default='sam-vit-huge', help='Model name for SAM')
    parser.add_argument('--num_pos_points', type=int, default=6, help='Number of positive points within the nucleus')
    return parser.parse_args()
