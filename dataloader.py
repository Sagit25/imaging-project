from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os


# 이 데이터셋은 그냥 대강 이미지가 어떻게 들어오겠다라고 굉장히 대강 만들었기에, 추후 실제 이미지셋에 맞게 수정필요!!!
class MirrorReflectionsDataset(Dataset):
    def __init__(self, root_dir, image_size=128):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        self.distorted_dir = os.path.join(root_dir, 'distorted')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.unwarped_dir = os.path.join(root_dir, 'unwarped')
        
        if os.path.exists(self.distorted_dir):
            self.image_files = sorted(os.listdir(self.distorted_dir))
        else:
            self.image_files = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        dist_img = Image.open(os.path.join(self.distorted_dir, filename)).convert('RGB')
        mask_img = Image.open(os.path.join(self.masks_dir, filename)).convert('L')
        unwarped_img = Image.open(os.path.join(self.unwarped_dir, filename)).convert('RGB')
        
        dist_tensor = self.transform(dist_img)
        unwarped_tensor = self.transform(unwarped_img)
        mask_tensor = self.mask_transform(mask_img)
        
        condition = torch.cat([dist_tensor, mask_tensor], dim=0)
        return condition, unwarped_tensor