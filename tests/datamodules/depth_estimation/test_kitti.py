import unittest
import os
import shutil
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
#from your_dataset_file import KittiDataset, ToTensor 

class TestKittiDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create mock data for testing
        cls.test_dir = 'test_data'
        cls.images_dir = os.path.join(cls.test_dir, 'images')
        cls.depth_dir = os.path.join(cls.test_dir, 'depth')
        os.makedirs(cls.images_dir, exist_ok=True)
        os.makedirs(cls.depth_dir, exist_ok=True)

        # Create dummy images and depth files
        cls.file_names = ['image_01.png', 'image_02.png']
        for file_name in cls.file_names:
            image = Image.new('RGB', (640, 480), color = 'red')
            depth = Image.new('L', (640, 480), color = 'white')  # Single channel image

            image.save(os.path.join(cls.images_dir, file_name))
            depth.save(os.path.join(cls.depth_dir, file_name))

        # Create a filenames file
        with open(os.path.join(cls.test_dir, 'filenames.txt'), 'w') as f:
            for file_name in cls.file_names:
                f.write(f"images/{file_name} depth/{file_name} 718.856\n")

        # Mock arguments
        cls.args = {
            'data_path': cls.images_dir,
            'gt_path': cls.depth_dir,
            'dataset': 'kitti',
        }

    @classmethod
    def tearDownClass(cls):
        # Remove the directory after the test
        shutil.rmtree(cls.test_dir)

    def test_init_and_len(self):
        # Test initialization and __len__
        dataset = KittiDataset(
            filenames_file=os.path.join(self.test_dir, 'filenames.txt'),
            mode='train',
            args=self.args,
        )
        self.assertEqual(len(dataset), len(self.file_names))

    def test_getitem(self):
        # Test __getitem__
        dataset = KittiDataset(
            filenames_file=os.path.join(self.test_dir, 'filenames.txt'),
            mode='train',
            args=self.args,
            transform=ToTensor('train')  # Assuming this is a transform you want to apply
        )
        sample = dataset[0]
        self.assertIsInstance(sample['image'], torch.FloatTensor)
        self.assertIsInstance(sample['depth'], torch.FloatTensor)
        self.assertEqual(sample['image'].size(), (3, 480, 640))  # Assuming RGB images
        self.assertEqual(sample['depth'].size(), (1, 480, 640))  # Assuming single channel for depth

if __name__ == '__main__':
    unittest.main()
