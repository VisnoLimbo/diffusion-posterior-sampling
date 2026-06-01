import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from data.dataloader import register_dataset

@register_dataset(name='small_test')
class SmallTestDataset(Dataset):
    def __init__(self, root, transforms=True):
        self.root = root
        self.image_files = [f for f in os.listdir(root) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Basic image transforms
        if transforms:
            self.transform = T.Compose([
                T.Resize(256),  # Resize to model input size
                T.CenterCrop(256),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image