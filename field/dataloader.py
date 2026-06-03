import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MirrorReflectionsDataset(Dataset):
    def __init__(self, root_dir, image_size=256, is_train=True):
        self.root_dir = root_dir
        self.image_size = image_size
        self.is_train = is_train
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        search_pattern = os.path.join(root_dir, '*_input.png')
        self.input_files = sorted(glob.glob(search_pattern))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        base_name = os.path.basename(input_path)
        file_id = base_name.rsplit('_input', 1)[0]
        gt_path = os.path.join(self.root_dir, f"{file_id}_gt.png")

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found for: {file_id}")

        dist_img = Image.open(input_path).convert('RGB')
        unwarped_img = Image.open(gt_path).convert('RGB')
        
        dist_tensor = self.transform(dist_img)
        unwarped_tensor = self.transform(unwarped_img)
        
        h, w = self.image_size, self.image_size
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h), 
            torch.linspace(-1, 1, w), 
            indexing='ij'
        )
        grid_coords = torch.stack([grid_x, grid_y], dim=0) # [2, H, W]
        
        model_input = torch.cat([dist_tensor, grid_coords], dim=0)
        
        return model_input, dist_tensor, unwarped_tensor