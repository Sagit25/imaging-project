import torch
from PIL import Image

# Seed your Work
def seed_everything(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

# Save sample images
def save_images(tensor, prefix):
    # tensor: [N, C, H, W]
    tensor = tensor.cpu() / 2. + 0.5  # Normalize from [-1,1] -> [0,1]
    tensor = tensor.clamp(0,1)
    for idx, img in enumerate(tensor):
        img = img.permute(1, 2, 0).detach().cpu().numpy()  # [H,W,C]
        img = (img * 255).astype('uint8')
        img_pil = Image.fromarray(img)
        img_pil.save(f'outputs/{prefix}_{idx}.png')