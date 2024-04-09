import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

class KittiDataset(Dataset):
    """
    Dataset class for KITTI dataset.

    Parameters
    ----------
    filenames_file : str
        Path to the file containing the filenames of the images to load.
    mode : str
        Mode in which the dataset is used. One of 'train', 'test', 'online_eval'.
    args : argparse.Namespace
        Arguments including all settings.
    transform : callable, optional
        Optional transform to be applied on a sample.

    Attributes
    ----------
    filenames : list
        List of filenames to load.
    mode : str
        Current mode of dataset usage.
    transform : callable
        Transform to apply to each sample.
    args : argparse.Namespace
        Collection of arguments/settings.
    """
    def __init__(self, filenames_file, mode, args, transform=None):
        self.mode = mode
        self.args = args
        self.transform = transform

        with open(filenames_file, 'r') as f:
            self.filenames = f.readlines()

    def __len__(self):
        """
        Returns the size of the dataset.
        
        Returns
        -------
        int
            Total number of samples in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Retrieves the dataset item at the specified index.
        
        Parameters
        ----------
        idx : int
            Index of the item to retrieve.
        
        Returns
        -------
        dict
            A dictionary containing the image, depth map, focal length, and optionally
            a flag indicating the presence of valid depth.
        """
        sample_path = self.filenames[idx].split()
        image_path = os.path.join(self.args.data_path, sample_path[0])
        depth_path = os.path.join(self.args.gt_path, sample_path[1]) if len(sample_path) > 1 else None
        focal = float(sample_path[2])

        image = self.load_image(image_path)
        depth = self.load_depth(depth_path) if depth_path else None

        sample = {'image': image, 'depth': depth, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_path):
        """
        Loads the image from the specified path.
        
        Parameters
        ----------
        image_path : str
            Path to the image file.
        
        Returns
        -------
        np.ndarray
            The loaded image as a NumPy array.
        """
        image = Image.open(image_path)
        return np.asarray(image, dtype=np.float32) / 255.0

    def load_depth(self, depth_path):
        """
        Loads the depth map from the specified path.
        
        Parameters
        ----------
        depth_path : str
            Path to the depth map file.
        
        Returns
        -------
        np.ndarray
            The loaded depth map as a NumPy array.
        """
        depth = Image.open(depth_path)
        depth = np.asarray(depth, dtype=np.float32)
        if self.args.dataset == 'nyu':
            depth /= 1000.0  # Convert to meters for NYU dataset
        else:
            depth /= 256.0  # Adjust scale for KITTI dataset
        return np.expand_dims(depth, axis=2)

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.

    Parameters
    ----------
    mode : str
        Current mode of dataset usage.

    Notes
    -----
    Normalizes the image using predefined mean and standard deviation.
    """

    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        """
        Applies the transformation to the given sample.
        
        Parameters
        ----------
        sample : dict
            A sample containing 'image' and 'depth' keys.
        
        Returns
        -------
        dict
            The modified sample with the image and depth converted to tensors.
        """
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        sample['image'] = image
        if 'depth' in sample and sample['depth'] is not None:
            sample['depth'] = self.to_tensor(sample['depth'])

        return sample

    def to_tensor(self, pic):
        """
        Convert a numpy.ndarray or PIL.Image.Image to tensor.

        Parameters
        ----------
        pic : numpy.ndarray or PIL.Image.Image
            Image to be converted to tensor.

        Returns
        -------
        torch.FloatTensor
            Image converted to tensor.
        """
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()
        
        # Handle PIL Image
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        if pic.mode == 'YCbCr':
            nchannel = 3
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        return img.float().div(255).transpose(0, 1).transpose(0, 2).contiguous()





##  Check if it works

# import matplotlib.pyplot as plt
# import argparse
# from kitti_dataset import KittiDataset, ToTensor

# def visualize_sample(sample):
#     """
#     Visualizes a single sample from the dataset.

#     Parameters
#     ----------
#     sample : dict
#         A sample containing 'image', 'depth', and 'focal' keys.
#     """
#     image = sample['image'].numpy().transpose(1, 2, 0)
#     depth = sample['depth'].numpy().squeeze() if sample['depth'] is not None else None

#     plt.figure(figsize=(10, 5))

#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     plt.title('Image')
#     plt.axis('off')

#     if depth is not None:
#         plt.subplot(1, 2, 2)
#         plt.imshow(depth, cmap='inferno')
#         plt.title('Depth Map')
#         plt.axis('off')

#     plt.show()

# def main():
#     # Parse arguments
#     parser = argparse.ArgumentParser(description='Test KITTI Dataset Loader')
#     parser.add_argument('--data_path', type=str, default='./data/kitti/images', help='Path to the images')
#     parser.add_argument('--gt_path', type=str, default='./data/kitti/depth', help='Path to the ground truth depth maps')
#     parser.add_argument('--dataset', type=str, default='kitti', choices=['kitti', 'nyu'], help='Dataset name')
#     parser.add_argument('--filenames_file', type=str, default='./data/kitti/filenames.txt', help='Path to the file containing the filenames')

#     args = parser.parse_args()

#     # Initialize the dataset and data loader
#     dataset = KittiDataset(
#         filenames_file=args.filenames_file,
#         mode='train',
#         args=args,
#         transform=ToTensor('train')
#     )

#     # Let's visualize the first few samples in the dataset
#     for i in range(min(len(dataset), 3)):  # Visualize the first 3 samples
#         sample = dataset[i]
#         visualize_sample(sample)

# if __name__ == '__main__':
#     main()
