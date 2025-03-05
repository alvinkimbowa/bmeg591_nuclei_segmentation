from glob import glob

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import tv_tensors

class NuInSegDataset(Dataset):
    def __init__(self, data_dir, train=True, val_size=0.2, test_size=0.1, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Get image and mask paths
        self.img_paths = glob(f"{data_dir}/*/tissue images/*.png")
        self.mask_paths = glob(f"{data_dir}/*/mask binary without border/*.png")
        
        # First, split into training (80%) and remaining (20%) dataset
        self.train_img_paths, remaining_img_paths, self.train_mask_paths, remaining_mask_paths = train_test_split(
            self.img_paths, self.mask_paths, test_size=(val_size + test_size), random_state=19, shuffle=True
        )
        
        # Now, split the remaining 20% into validation and test sets (50% each)
        val_size = val_size / (val_size + test_size)  # Adjust validation size to be proportionate
        self.val_img_paths, self.test_img_paths, self.val_mask_paths, self.test_mask_paths = train_test_split(
            remaining_img_paths, remaining_mask_paths, test_size=val_size, random_state=19, shuffle=True
        )
        
        # Ensure images and masks match
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

    def __getitem__(self, idx):
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
        
        assert image.min() >= 0 and image.max() <= 1
        assert np.all(np.isin(mask, [0, 1]))

        return image, mask

