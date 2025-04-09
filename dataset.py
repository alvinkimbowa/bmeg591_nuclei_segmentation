import os
import numpy as np
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import tv_tensors

class NuInSegDataset(Dataset):
    """
    A custom dataset loader for the NuInSeg dataset.
    This class loads images and their corresponding binary masks from the specified directory.
    It supports splitting the dataset into training, validation, and test sets.

    Args:
        data_dir (str): Path to the root directory containing the dataset.
        train (bool): If True, loads the training dataset. If False, loads the validation or test dataset.
        val_size (float): Fraction of the dataset to use for validation. Default is 0.2.
        test_size (float): Fraction of the dataset to use for testing. Default is 0.1.
        transform (callable, optional): A function/transform to apply to the image and mask pairs.
    
    Attributes:
        img_paths (list): List of image file paths (depending on the split).
        mask_paths (list): List of mask file paths (depending on the split).
    """

    def __init__(self, data_dir, train=True, val_size=0.2, test_size=0.1, transform=None):
        """
        Initializes the NuInSegDataset class. It reads images and corresponding masks, and splits
        the dataset into training, validation, and test sets based on the given `val_size` and `test_size`.

        Args:
            data_dir (str): Path to the root directory containing the dataset.
            train (bool): If True, loads the training dataset. If False, loads the validation or test dataset.
            val_size (float): Fraction of the dataset to use for validation. Default is 0.2.
            test_size (float): Fraction of the dataset to use for testing. Default is 0.1.
            transform (callable, optional): A function/transform to apply to the image and mask pairs.
        """

        self.data_dir = data_dir
        self.transform = transform
        
        # Get image and mask paths
        self.img_paths = glob(f"{data_dir}/*/tissue images/*.png")
        self.mask_paths = glob(f"{data_dir}/*/mask binary without border/*.png")
        
        # First, split into training (80%) and remaining (20%) dataset
        print(len(self.img_paths), len(self.mask_paths))
    
        self.train_img_paths, remaining_img_paths, self.train_mask_paths, remaining_mask_paths = train_test_split(
            self.img_paths, self.mask_paths, test_size=(val_size + test_size), random_state=19, shuffle=True
        )
        
        # Now, split the remaining 20% into validation and test sets (50% each)
        val_size = val_size / (val_size + test_size)  # Adjust validation size to be proportionate
        self.val_img_paths, self.test_img_paths, self.val_mask_paths, self.test_mask_paths = train_test_split(
            remaining_img_paths, remaining_mask_paths, test_size=val_size, random_state=19, shuffle=True
        )
        
        # Ensure images and masks match
        self.sanity_check()

        # Select the dataset depending on whether we need training, validation, or test set
        if train:
            self.img_paths = self.train_img_paths
            self.mask_paths = self.train_mask_paths
        elif val_size:
            self.img_paths = self.val_img_paths
            self.mask_paths = self.val_mask_paths
        else:
            self.img_paths = self.test_img_paths
            self.mask_paths = self.test_mask_paths

    def __len__(self):
        return len(self.img_paths)
    
    def sanity_check(self):
        """
        Checks for duplicates between the training, validation, and test sets.
        """

        assert np.all([img_path.split('/')[-1] == mask_path.split('/')[-1] for img_path, mask_path in zip(self.train_img_paths, self.train_mask_paths)])
        assert np.all([img_path.split('/')[-1] == mask_path.split('/')[-1] for img_path, mask_path in zip(self.val_img_paths, self.val_mask_paths)])
        assert np.all([img_path.split('/')[-1] == mask_path.split('/')[-1] for img_path, mask_path in zip(self.test_img_paths, self.test_mask_paths)])

        assert set(self.train_img_paths).isdisjoint(set(self.val_img_paths))
        assert set(self.train_img_paths).isdisjoint(set(self.test_img_paths))
        assert set(self.val_img_paths).isdisjoint(set(self.test_img_paths))

        assert len(self.train_img_paths) + len(self.val_img_paths) + len(self.test_img_paths) == len(self.img_paths)
        assert len(self.train_mask_paths) + len(self.val_mask_paths) + len(self.test_mask_paths) == len(self.mask_paths)
        assert len(self.train_img_paths) == len(self.train_mask_paths)
        assert len(self.val_img_paths) == len(self.val_mask_paths)
        assert len(self.test_img_paths) == len(self.test_mask_paths)
        
        return None

    def __getitem__(self, idx):
        """
        Retrieves the image and its corresponding mask at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple (image, mask), where:
                - image (torch.Tensor): The image in CHW format.
                - mask (torch.Tensor): The binary mask in CHW format.
        """
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load images
        image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255
        mask = np.array(Image.open(mask_path), dtype=np.float32) / 255
        
        image = image.transpose(2, 0, 1)
        mask = mask[np.newaxis, :, :]

        image = tv_tensors.Image(image)
        mask = tv_tensors.Mask(mask)

        if self.transform:
            image, mask = self.transform((image, mask))
        
        # print(image.min(), image.max(), mask.min(), mask.max(),flush=True)
        
        assert image.min() >= 0 and image.max() <= 1
        # assert np.all(np.isin(image, [0, 1]))
        assert np.all(np.isin(mask, [0, 1]))

        return image, mask, os.path.basename(img_path)

