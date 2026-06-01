import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing import Optional, Callable, List, Tuple
import os
import pickle

from data.dataloader import register_dataset
from aoa_amp_data import generate_training_data


@register_dataset(name='aoa_amp')
class AoAAmpDataset(VisionDataset):
    """
    Dataset for AoA and Amplitude maps generated from base station positions.
    Each sample contains concatenated AoA and amplitude maps as a 2-channel image.
    """
    
    def __init__(self, 
                 root: str,
                 transforms: Optional[Callable] = None,
                 grid_resolution: float = 1.0,
                 num_bs: int = 1,
                 bs_range: Tuple[float, float] = (-40, 40),
                 wavelength: float = 0.1,
                 num_samples: int = 1000,
                 cache_data: bool = True):
        """
        Args:
            root: Root directory for caching data
            transforms: Transform pipeline for data augmentation
            grid_resolution: Grid resolution for spatial sampling
            num_bs: Number of base stations per sample
            bs_range: Range for random BS placement
            wavelength: Wavelength for path loss calculation
            num_samples: Number of samples in dataset
            cache_data: Whether to cache generated data
        """
        super().__init__(root, transforms)
        
        self.grid_resolution = grid_resolution
        self.num_bs = num_bs
        self.bs_range = bs_range
        self.wavelength = wavelength
        self.num_samples = num_samples
        self.cache_data = cache_data
        
        # Calculate grid dimensions
        self.x_lim = [-50, 50]
        self.y_lim = [-50, 50]
        self.num_samples_x = int((self.x_lim[1] - self.x_lim[0]) / grid_resolution)
        self.num_samples_y = int((self.y_lim[1] - self.y_lim[0]) / grid_resolution)
        
        # Create cache directory
        if self.cache_data:
            os.makedirs(root, exist_ok=True)
            self.cache_file = os.path.join(root, f'aoa_amp_cache_{num_samples}_{num_bs}_{grid_resolution}.pkl')
            
        # Generate or load cached data
        self._prepare_data()
    
    def _prepare_data(self):
        """Generate or load cached dataset"""
        if self.cache_data and os.path.exists(self.cache_file):
            print(f"Loading cached data from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.data = pickle.load(f)
        else:
            print(f"Generating {self.num_samples} AoA/Amplitude samples...")
            self.data = []
            
            for i in range(self.num_samples):
                # Generate random BS positions
                bs_positions = []
                for _ in range(self.num_bs):
                    bs_x = np.random.uniform(self.bs_range[0], self.bs_range[1])
                    bs_y = np.random.uniform(self.bs_range[0], self.bs_range[1])
                    bs_positions.append((bs_x, bs_y))
                
                # Generate AoA/amplitude maps
                sample_data = generate_training_data(
                    self.grid_resolution, 
                    bs_positions, 
                    self.wavelength
                )[0]
                
                # Convert to tensor format
                tensor_data = self._convert_to_tensor(sample_data)
                self.data.append(tensor_data)
                
                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1}/{self.num_samples} samples")
            
            # Cache the data
            if self.cache_data:
                print(f"Caching data to {self.cache_file}")
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.data, f)
    
    def _convert_to_tensor(self, sample_data):
        """Convert AoA/amplitude maps to tensor format"""
        aoa_maps = []
        amp_maps = []
        
        for bs_idx in range(self.num_bs):
            aoa_map = sample_data['aoa_map'][f'bs_{bs_idx}']
            amp_map = sample_data['amplitude_map'][f'bs_{bs_idx}']
            
            # Reshape to 2D grid
            aoa_2d = aoa_map.reshape(self.num_samples_x, self.num_samples_y)
            amp_2d = amp_map.reshape(self.num_samples_x, self.num_samples_y)
            
            aoa_maps.append(aoa_2d)
            amp_maps.append(amp_2d)
        
        # Stack all BS maps: shape = (2*num_bs, H, W)
        all_maps = np.stack(aoa_maps + amp_maps, axis=0)
        
        # Normalize data to [-1, 1] range for diffusion training
        # AoA is already in [-pi, pi], normalize to [-1, 1]
        all_maps[:self.num_bs] = all_maps[:self.num_bs] / np.pi
        
        # Amplitude: normalize using percentile-based scaling
        amp_data = all_maps[self.num_bs:]
        amp_min, amp_max = np.percentile(amp_data, [5, 95])
        all_maps[self.num_bs:] = 2 * (amp_data - amp_min) / (amp_max - amp_min) - 1
        all_maps[self.num_bs:] = np.clip(all_maps[self.num_bs:], -1, 1)
        
        return torch.FloatTensor(all_maps)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor: Shape (2*num_bs, H, W) where first num_bs channels are AoA
                         and last num_bs channels are amplitude maps
        """
        sample = self.data[idx]
        
        if self.transforms is not None:
            sample = self.transforms(sample)
            
        return sample


# Example usage and testing
if __name__ == "__main__":
    # Create dataset
    dataset = AoAAmpDataset(
        root="./data/aoa_amp_cache",
        grid_resolution=0.50,
        num_bs=20,
        num_samples=100
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0].shape}")
    print(f"Sample range: [{dataset[0].min():.3f}, {dataset[0].max():.3f}]")