"""
Dataset for AoA and Amplitude maps generated using ray tracing with buildings.
This dataset uses the strongest 3 paths approach with randomized building configurations.
GPU-optimized version for faster data processing.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from typing import Optional, Callable, List, Tuple
import os
import pickle
import sys
import random
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import time

from data.dataloader import register_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# from aoa_amp_building_data import generate_building_training_data, load_training_data


@register_dataset(name='aoa_amp_building')
class AoAAmpBuildingDataset(VisionDataset):
    """
    GPU-optimized dataset for AoA and Amplitude maps generated using ray tracing with buildings.
    Raw samples contain 6 channels: 3 AoA maps + 3 amplitude maps for the strongest paths.
    A configurable subset of channels can be returned for ablation experiments.
    Buildings are randomized across different configurations.
    """
    
    def __init__(self, 
                 root: str,
                 transforms: Optional[Callable] = None,
                 map_size: Tuple[int, int] = (100, 100),
                 grid_spacing: float = 1.0,
                 bs_grid_spacing: float = 5.0,
                 num_building_sets: int = 120,
                 building_distribution: Tuple[int, int, int] = (40, 40, 40),
                 use_existing_data: bool = True,
                 normalize_data: bool = True,
                 building_size_range: Tuple[Tuple[int, int], Tuple[int, int]] = ((15, 35), (10, 25)),
                 min_building_distance: float = 5.0,
                 seed: int = 42,
                 device: str = 'auto',
                 use_gpu_processing: bool = True,
                 batch_size: int = 32,
                 num_workers: int = None,
                 pin_memory: bool = True,
                 prefetch_factor: int = 2,
                 return_index: bool = False,
                 selected_channels: Optional[List[int]] = None):
        """
        Args:
            root: Root directory for caching data
            transforms: Transform pipeline for data augmentation
            map_size: Size of the map in meters (width, height)
            grid_spacing: Grid spacing for UE positions in meters
            bs_grid_spacing: Grid spacing for BS positions in meters
            num_building_sets: Total number of building configuration sets
            building_distribution: (num_1_building, num_2_buildings, num_3_buildings)
            use_existing_data: Whether to use existing data or generate new
            normalize_data: Whether to normalize data to [-1, 1] range
            building_size_range: ((min_width, max_width), (min_height, max_height))
            min_building_distance: Minimum distance between buildings
            seed: Random seed for reproducible building configurations
            device: Device to use ('cuda', 'cpu', or 'auto')
            use_gpu_processing: Whether to use GPU for data processing
            batch_size: Batch size for GPU processing
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            prefetch_factor: Number of samples loaded in advance by each worker
            return_index: Whether to return the sample index along with the data
            selected_channels: Optional channel indices to keep from the raw 6-channel tensor.
                               Channel order is [AoA1, AoA2, AoA3, Amp1, Amp2, Amp3].
                               Use None to keep all channels.
        """
        super().__init__(root, transforms)
        
        self.map_size = map_size
        self.grid_spacing = grid_spacing
        self.bs_grid_spacing = bs_grid_spacing
        self.use_existing_data = use_existing_data
        self.normalize_data = normalize_data
        self.num_building_sets = num_building_sets
        self.building_distribution = building_distribution
        self.building_size_range = building_size_range
        self.min_building_distance = min_building_distance
        self.seed = seed
        self.use_gpu_processing = use_gpu_processing
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.return_index = return_index
        self.selected_channels = self._validate_selected_channels(selected_channels)
        self.channel_cache_tag = self._build_channel_cache_tag()
        # Keep full dynamic range after affine normalization.
        self.amplitude_post_clip = False
        
        # Setup device
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Setup number of workers
        if num_workers is None:
            self.num_workers = min(8, mp.cpu_count())
        else:
            self.num_workers = num_workers
        
        # MPS does not support concurrent GPU access from multiple threads
        if self.device.type == 'mps':
            self.num_workers = 0
        
        print(f"Using device: {self.device}")
        print(f"GPU processing enabled: {self.use_gpu_processing}")
        print(f"Number of workers: {self.num_workers}")
        if self.selected_channels is None:
            print("Selected channels: all [0, 1, 2, 3, 4, 5]")
        else:
            print(f"Selected channels: {self.selected_channels}")
        
        # Validate building distribution
        if sum(building_distribution) != num_building_sets:
            raise ValueError(f"Building distribution {building_distribution} doesn't sum to {num_building_sets}")
        
        # Generate randomized building configurations
        self.building_config_sets = self._generate_building_configurations()
        
        # Create cache directory
        os.makedirs(root, exist_ok=True)
        
        # Prepare data
        self._prepare_data()

    def _validate_selected_channels(self, selected_channels: Optional[List[int]]) -> Optional[List[int]]:
        """
        Validate and normalize selected channel indices.

        The full 12-channel layout is:
            0..2  -> AoA path 1, 2, 3
            3..5  -> Amplitude path 1, 2, 3
            6..8  -> sin(phase) path 1, 2, 3
            9..11 -> cos(phase) path 1, 2, 3
        """
        if selected_channels is None:
            return None

        if not isinstance(selected_channels, (list, tuple)):
            raise ValueError("selected_channels must be a list or tuple of integers in [0, 11]")

        if len(selected_channels) == 0:
            raise ValueError("selected_channels cannot be empty")

        normalized_channels = [int(ch) for ch in selected_channels]
        for ch in normalized_channels:
            if ch < 0 or ch > 11:
                raise ValueError(f"Invalid channel index {ch}. Valid range is [0, 11]")

        if len(set(normalized_channels)) != len(normalized_channels):
            raise ValueError(f"selected_channels contains duplicates: {normalized_channels}")

        return normalized_channels

    def _build_channel_cache_tag(self) -> str:
        """Build a cache suffix for channel-selection-specific tensor caches."""
        if self.selected_channels is None:
            return "all"
        return "ch_" + "-".join(str(ch) for ch in self.selected_channels)

    def _apply_channel_selection(self, data):
        """Apply configured channel selection to a tensor/array with channel-first layout."""
        if self.selected_channels is None:
            return data
        return data[self.selected_channels]
    
    def _generate_building_configurations(self) -> List[List[dict]]:
        """Generate randomized building configurations using parallel processing"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Create tasks for parallel processing
        tasks = []
        for num_buildings, count in enumerate(self.building_distribution, 1):
            for i in range(count):
                # Use different seeds for each configuration to ensure uniqueness
                config_seed = self.seed + len(tasks)
                tasks.append((num_buildings, config_seed))
        
        # Generate configurations in parallel or sequentially
        print(f"Generating {len(tasks)} building configurations using {self.num_workers} workers...")
        
        if self.num_workers > 0:
            # Use parallel processing
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                configurations = list(executor.map(self._generate_single_configuration_worker, tasks))
        else:
            # Sequential processing (required for MPS backend)
            configurations = [self._generate_single_configuration_worker(task) for task in tasks]
        
        print(f"Generated {len(configurations)} building configuration sets:")
        for i in range(len(self.building_distribution)):
            print(f"  - {self.building_distribution[i]} sets with {i + 1} building(s)")

        return configurations
    
    def _generate_single_configuration_worker(self, task):
        """Worker function for parallel configuration generation"""
        num_buildings, config_seed = task
        
        # Set seed for this worker
        random.seed(config_seed)
        np.random.seed(config_seed)
        
        return self._generate_single_configuration(num_buildings)
    
    def _generate_single_configuration(self, num_buildings: int) -> List[dict]:
        """Generate a single building configuration with specified number of buildings"""
        buildings = []
        max_attempts = 1000
        
        for building_idx in range(num_buildings):
            attempts = 0
            while attempts < max_attempts:
                # Random building size
                width = random.randint(*self.building_size_range[0])
                height = random.randint(*self.building_size_range[1])
                
                # Random position (ensure building fits in map)
                margin = 5  # Keep buildings away from map edges
                x = random.uniform(margin, self.map_size[0] - width - margin)
                y = random.uniform(margin, self.map_size[1] - height - margin)
                
                new_building = {'x': x, 'y': y, 'width': width, 'height': height}
                
                # Check if this building overlaps with existing ones
                if self._is_valid_building_placement(new_building, buildings):
                    buildings.append(new_building)
                    break
                    
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"Warning: Could not place building {building_idx + 1} after {max_attempts} attempts")
        
        return buildings
    
    def _is_valid_building_placement(self, new_building: dict, existing_buildings: List[dict]) -> bool:
        """Check if a new building placement is valid (doesn't overlap with existing ones)"""
        if not existing_buildings:
            return True
            
        for existing in existing_buildings:
            # Check if buildings overlap (with minimum distance buffer)
            if self._buildings_overlap(new_building, existing, self.min_building_distance):
                return False
        return True
    
    def _buildings_overlap(self, building1: dict, building2: dict, min_distance: float) -> bool:
        """Check if two buildings overlap considering minimum distance"""
        # Expand buildings by min_distance/2 on each side
        b1_left = building1['x'] - min_distance/2
        b1_right = building1['x'] + building1['width'] + min_distance/2
        b1_bottom = building1['y'] - min_distance/2
        b1_top = building1['y'] + building1['height'] + min_distance/2
        
        b2_left = building2['x'] - min_distance/2
        b2_right = building2['x'] + building2['width'] + min_distance/2
        b2_bottom = building2['y'] - min_distance/2
        b2_top = building2['y'] + building2['height'] + min_distance/2
        
        # Check for overlap
        return not (b1_right <= b2_left or b2_right <= b1_left or 
                   b1_top <= b2_bottom or b2_top <= b1_bottom)
    
    def _prepare_data(self):
        """Generate or load cached dataset using the RAM-bounded streaming
        pipeline.

        self.data is NOT held in RAM at any point -- it stays None throughout.
        Raw samples live on disk in the HDF5 cache; tensors are streamed in
        and held in self.tensor_data (this is the only large in-RAM object).
        Metadata is built from raw HDF5 attrs directly. This is what makes
        54,000-sample runs possible on a single Colab machine.
        """
        cache_file = os.path.join(self.root, f'all_building_configs_data_{self.seed}.h5')
        if self.channel_cache_tag == 'all':
            tensor_cache_file = os.path.join(self.root, f'tensor_data_{self.seed}.h5')
        else:
            tensor_cache_file = os.path.join(self.root, f'tensor_data_{self.seed}_{self.channel_cache_tag}.h5')

        old_pickle_cache = os.path.join(self.root, f'all_building_configs_data_{self.seed}.pkl')

        self.data = None  # streaming flow keeps raw samples on disk only
        self.metadata = []

        try:
            if self.use_existing_data and os.path.exists(cache_file):
                print(f"Found existing raw HDF5: {cache_file}")
                if (os.path.exists(tensor_cache_file)
                        and self._is_tensor_cache_compatible(tensor_cache_file)):
                    print(f"Loading pre-converted tensor data: {tensor_cache_file}")
                    self.tensor_data = self._load_tensor_cache(tensor_cache_file)
                else:
                    if os.path.exists(tensor_cache_file):
                        print("⚠️  Tensor cache incompatible; rebuilding (streaming)...")
                    else:
                        print("Tensor cache missing; building from raw HDF5 (streaming)...")
                    self._convert_to_tensors_streaming(cache_file)
                    self._save_tensor_cache(tensor_cache_file)
                self._build_metadata_from_raw_hdf5(cache_file)

            elif self.use_existing_data and os.path.exists(old_pickle_cache):
                # Legacy pickle migration -- needs self.data briefly.
                print(f"Migrating data from pickle to HDF5 format...")
                with open(old_pickle_cache, 'rb') as f:
                    self.data = pickle.load(f)
                self._save_raw_data_hdf5(cache_file)
                os.remove(old_pickle_cache)
                print(f"Migration complete. Old pickle file removed.")
                self._convert_to_tensors_gpu()
                self._save_tensor_cache(tensor_cache_file)
                self._create_metadata_list()
                self.data = None

            else:
                raise FileNotFoundError("Generating new data as requested")

        except (FileNotFoundError, ImportError) as e:
            if isinstance(e, ImportError):
                print("⚠️  h5py not available, falling back to legacy pickle path")
                print("Generating new training data (non-streaming, legacy)...")
                self._generate_all_data()
                with open(old_pickle_cache, 'wb') as f:
                    pickle.dump(self.data, f)
                self._convert_to_tensors_gpu()
                self._save_tensor_cache(tensor_cache_file)
                self._create_metadata_list()
            else:
                print("Generating new building training data (streaming)...")
                self._generate_all_data_streaming(cache_file)
                self._convert_to_tensors_streaming(cache_file)
                self._save_tensor_cache(tensor_cache_file)
                self._build_metadata_from_raw_hdf5(cache_file)

        print(
            f"Dataset prepared with {len(self.tensor_data)} samples "
            f"across {len(self.building_config_sets)} building configurations"
        )
        self._print_dataset_statistics()

    def _save_raw_data_hdf5(self, cache_file):
        """Save raw dataset to HDF5 format"""
        try:
            import h5py
            import json
            
            print(f"💾 Saving raw data to HDF5 format: {cache_file}...")
            
            with h5py.File(cache_file, 'w') as f:
                # Save dataset metadata
                f.attrs['num_samples'] = len(self.data)
                f.attrs['map_size'] = self.map_size
                f.attrs['grid_spacing'] = self.grid_spacing
                f.attrs['bs_grid_spacing'] = self.bs_grid_spacing
                f.attrs['seed'] = self.seed
                f.attrs['num_building_sets'] = self.num_building_sets
                f.attrs['building_distribution'] = self.building_distribution
                
                # Save building configurations as JSON
                f.attrs['building_configs'] = json.dumps(self.building_config_sets)
                
                # Create groups for different data types
                samples_group = f.create_group('samples')
                
                # Save each sample's data
                for i, sample in enumerate(self.data):
                    sample_group = samples_group.create_group(f'sample_{i:06d}')
                    
                    # Save arrays with compression
                    sample_group.create_dataset('aoa_maps',
                                              data=np.stack(sample['aoa_maps']),
                                              compression='gzip', compression_opts=6)
                    sample_group.create_dataset('amplitude_maps',
                                              data=np.stack(sample['amplitude_maps']),
                                              compression='gzip', compression_opts=6)
                    if 'phase_maps' in sample and sample['phase_maps'] is not None:
                        sample_group.create_dataset('phase_maps',
                                                  data=np.stack(sample['phase_maps']),
                                                  compression='gzip', compression_opts=6)
                    sample_group.create_dataset('los_map',
                                              data=sample['los_map'],
                                              compression='gzip', compression_opts=6)
                    
                    # Save scalar/small data as attributes
                    sample_group.attrs['bs_pos'] = sample['bs_pos']
                    sample_group.attrs['map_size'] = sample['map_size']
                    sample_group.attrs['grid_spacing'] = sample['grid_spacing']
                    sample_group.attrs['config_idx'] = sample.get('config_idx', -1)
                    sample_group.attrs['num_buildings'] = sample.get('num_buildings', 0)
                    
                    # Save buildings as JSON (more flexible than trying to store as HDF5)
                    sample_group.attrs['buildings'] = json.dumps(sample['buildings'])
            
            print(f"✅ Saved {len(self.data)} samples to HDF5 format")
            
        except ImportError:
            print("⚠️  h5py not available, cannot save in HDF5 format")
            raise

    # ----------------------------------------------------------------- #
    #  Streaming pipeline (RAM-bounded; required for 54,000+ samples)   #
    # ----------------------------------------------------------------- #

    def _generate_all_data_streaming(self, cache_file):
        """Generate building configs sequentially and stream-write each one's
        samples to HDF5 immediately, then free the per-config dicts.

        Peak RAM stays around the size of a single config (~1 GB) instead of
        accumulating tens of GB in self.data. Required to scale to 54,000+
        samples on Colab without OOM. After completion self.data is None;
        downstream consumers must read from the raw HDF5 file directly.
        """
        try:
            import h5py
            import json
            import gc
        except ImportError:
            print("⚠️  h5py not available, cannot stream-save in HDF5 format")
            raise

        print(f"\n🏗️  Streaming generation -> {cache_file}")
        print(f"⚙️  Device: {self.device}  |  {len(self.building_config_sets)} configs")

        progress = tqdm(
            desc="Building configs",
            total=len(self.building_config_sets),
            unit="config",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        )

        sample_idx = 0
        with h5py.File(cache_file, 'w') as f:
            # Dataset-level metadata
            f.attrs['map_size'] = self.map_size
            f.attrs['grid_spacing'] = self.grid_spacing
            f.attrs['bs_grid_spacing'] = self.bs_grid_spacing
            f.attrs['seed'] = self.seed
            f.attrs['num_building_sets'] = self.num_building_sets
            f.attrs['building_distribution'] = self.building_distribution
            f.attrs['building_configs'] = json.dumps(self.building_config_sets)

            samples_group = f.create_group('samples')

            for config_idx, building_config in enumerate(self.building_config_sets):
                config_data = self._generate_config_data_worker(
                    (config_idx, building_config)
                )

                for sample in config_data:
                    sg = samples_group.create_group(f'sample_{sample_idx:06d}')
                    sg.create_dataset('aoa_maps',
                                      data=np.stack(sample['aoa_maps']),
                                      compression='gzip', compression_opts=6)
                    sg.create_dataset('amplitude_maps',
                                      data=np.stack(sample['amplitude_maps']),
                                      compression='gzip', compression_opts=6)
                    if sample.get('phase_maps') is not None:
                        sg.create_dataset('phase_maps',
                                          data=np.stack(sample['phase_maps']),
                                          compression='gzip', compression_opts=6)
                    sg.create_dataset('los_map',
                                      data=sample['los_map'],
                                      compression='gzip', compression_opts=6)
                    sg.attrs['bs_pos'] = sample['bs_pos']
                    sg.attrs['map_size'] = sample['map_size']
                    sg.attrs['grid_spacing'] = sample['grid_spacing']
                    sg.attrs['config_idx'] = sample.get('config_idx', config_idx)
                    sg.attrs['num_buildings'] = sample.get(
                        'num_buildings', len(building_config))
                    sg.attrs['buildings'] = json.dumps(sample['buildings'])
                    sample_idx += 1

                # Drop the per-config dicts and force collection so RAM stays flat.
                del config_data
                gc.collect()

                progress.update(1)
                progress.set_postfix_str(f"{sample_idx} samples")

            f.attrs['num_samples'] = sample_idx

        progress.close()
        print(f"\n✅ Streamed {sample_idx} samples to {cache_file}")
        self.data = None  # not held in RAM

    def _convert_to_tensors_streaming(self, raw_cache_file):
        """Build self.tensor_data by reading samples one at a time from the raw
        HDF5 file, instead of relying on self.data being in RAM.

        Avoids the 35-40 GB self.data peak that the non-streaming path needs
        before tensor conversion starts. Self.tensor_data still ends up fully
        in RAM (~42 GB at 54k) -- that's needed for training -- but we no
        longer require self.data on top.
        """
        try:
            import h5py
            import json
            import gc
        except ImportError:
            print("⚠️  h5py not available, falling back to non-streaming conversion")
            self._convert_to_tensors_gpu()
            return

        print(f"\n🔄 Converting tensors (streaming from raw HDF5)...")
        self.tensor_data = []

        with h5py.File(raw_cache_file, 'r') as f:
            samples_group = f['samples']
            n_total = int(f.attrs.get('num_samples', len(samples_group.keys())))

            with tqdm(total=n_total, desc="Tensor conversion",
                      unit="sample", ncols=100) as pbar:
                for i in range(n_total):
                    key = f'sample_{i:06d}'
                    if key not in samples_group:
                        pbar.update(1)
                        continue
                    sg = samples_group[key]

                    aoa_stack = sg['aoa_maps'][:]
                    amp_stack = sg['amplitude_maps'][:]
                    los_map = sg['los_map'][:]
                    if 'phase_maps' in sg:
                        phase_stack = sg['phase_maps'][:]
                        phase_list = [phase_stack[j]
                                      for j in range(phase_stack.shape[0])]
                    else:
                        phase_list = [np.zeros_like(aoa_stack[j], dtype=np.float32)
                                      for j in range(aoa_stack.shape[0])]

                    sample = {
                        'aoa_maps':       [aoa_stack[j] for j in range(aoa_stack.shape[0])],
                        'amplitude_maps': [amp_stack[j] for j in range(amp_stack.shape[0])],
                        'phase_maps':     phase_list,
                        'los_map':        los_map,
                    }

                    tensor = self._convert_to_tensor_gpu(sample)
                    self.tensor_data.append(tensor)
                    pbar.update(1)

                    if (i + 1) % 1000 == 0:
                        gc.collect()

        print(f"✅ Converted {len(self.tensor_data)} samples to tensors")

    def _build_metadata_from_raw_hdf5(self, raw_cache_file):
        """Build self.metadata directly from raw HDF5 attrs (no self.data).

        Stored per-sample is small (~0.5 KB), so 54,000 entries is ~30 MB --
        comfortably in RAM. Used by sample_condition_building.py and friends
        for plotting building overlays etc.
        """
        try:
            import h5py
            import json
        except ImportError:
            raise

        self.metadata = []
        with h5py.File(raw_cache_file, 'r') as f:
            samples_group = f['samples']
            n_total = int(f.attrs.get('num_samples', len(samples_group.keys())))
            for i in range(n_total):
                key = f'sample_{i:06d}'
                if key not in samples_group:
                    continue
                sg = samples_group[key]
                self.metadata.append({
                    'bs_pos':       np.array(sg.attrs['bs_pos']),
                    'buildings':    json.loads(sg.attrs['buildings']),
                    'map_size':     tuple(sg.attrs['map_size']),
                    'grid_spacing': float(sg.attrs['grid_spacing']),
                    'config_idx':   int(sg.attrs.get('config_idx', 0)),
                    'num_buildings': int(sg.attrs.get('num_buildings', 0)),
                })
        print(f"✅ Built metadata for {len(self.metadata)} samples from raw HDF5")

    def _load_raw_data_hdf5(self, cache_file):
        """Load raw dataset from HDF5 format"""
        try:
            import h5py
            import json
            
            print(f"📖 Loading raw data from HDF5: {cache_file}")
            
            data = []
            with h5py.File(cache_file, 'r') as f:
                samples_group = f['samples']
                
                # Get number of samples
                num_samples = len(samples_group.keys())
                
                # Load samples with progress bar
                with tqdm(total=num_samples, desc="Loading samples", unit="sample", ncols=100) as pbar:
                    for i in range(num_samples):
                        sample_key = f'sample_{i:06d}'
                        if sample_key not in samples_group:
                            continue
                            
                        sample_group = samples_group[sample_key]
                        
                        # Load arrays
                        aoa_maps_stack = sample_group['aoa_maps'][:]
                        amplitude_maps_stack = sample_group['amplitude_maps'][:]
                        los_map = sample_group['los_map'][:]

                        # Convert stacked arrays back to lists
                        aoa_maps = [aoa_maps_stack[j] for j in range(aoa_maps_stack.shape[0])]
                        amplitude_maps = [amplitude_maps_stack[j] for j in range(amplitude_maps_stack.shape[0])]

                        # Phase maps (added in v2). Fall back to zero phase for legacy caches.
                        if 'phase_maps' in sample_group:
                            phase_maps_stack = sample_group['phase_maps'][:]
                            phase_maps = [phase_maps_stack[j] for j in range(phase_maps_stack.shape[0])]
                        else:
                            phase_maps = [np.zeros_like(m, dtype=np.float32) for m in aoa_maps]

                        # Load metadata
                        buildings = json.loads(sample_group.attrs['buildings'])

                        sample = {
                            'aoa_maps': aoa_maps,
                            'amplitude_maps': amplitude_maps,
                            'phase_maps': phase_maps,
                            'los_map': los_map,
                            'bs_pos': sample_group.attrs['bs_pos'],
                            'map_size': sample_group.attrs['map_size'],
                            'grid_spacing': sample_group.attrs['grid_spacing'],
                            'buildings': buildings,
                            'config_idx': sample_group.attrs.get('config_idx', -1),
                            'num_buildings': sample_group.attrs.get('num_buildings', 0)
                        }
                        
                        data.append(sample)
                        pbar.update(1)
            
            print(f"✅ Loaded {len(data)} samples from HDF5")
            return data
            
        except ImportError:
            print("⚠️  h5py not available, cannot load HDF5 format")
            raise
        except Exception as e:
            print(f"❌ Error loading HDF5 raw data: {e}")
            raise
    
    def _generate_all_data(self):
        """Generate data for all building configurations using parallel processing with progress bar"""
        self.data = []
        
        # Create tasks for parallel data generation
        tasks = [(i, config) for i, config in enumerate(self.building_config_sets)]
        
        # GPU works best with sequential configs (avoid thread contention on one device)
        # and larger inner batches; multi-threading only helps for CPU generation.
        if self.device.type in ('cuda', 'mps'):
            gen_workers = 1
        else:
            gen_workers = self.num_workers

        print(f"\n🏗️  Generating training data for {len(tasks)} building configurations...")
        print(f"⚙️  Using {gen_workers} workers on {self.device}")
        
        # Create a progress bar for configurations
        config_progress = tqdm(
            desc="Building configs", 
            total=len(tasks), 
            unit="config",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Track progress with a callback function
        completed_configs = 0
        config_lock = threading.Lock()
        
        def update_progress(future):
            nonlocal completed_configs
            with config_lock:
                completed_configs += 1
                config_progress.update(1)
            config_progress.set_postfix_str(f"Config {completed_configs}/{len(tasks)}")
        
        # Use ThreadPoolExecutor for I/O bound operations or sequential processing
        if gen_workers > 0:
            with ThreadPoolExecutor(max_workers=gen_workers) as executor:
                # Submit all tasks and add progress callback
                futures = []
                for task in tasks:
                    future = executor.submit(self._generate_config_data_worker, task)
                    future.add_done_callback(update_progress)
                    futures.append(future)
                
                # Collect results as they complete
                results = []
                for future in futures:
                    result = future.result()
                    results.append(result)
        else:
            # Sequential processing for MPS backend
            results = []
            for task in tasks:
                result = self._generate_config_data_worker(task)
                results.append(result)
                update_progress(None)  # Update progress manually
        
        config_progress.close()
        
        # Flatten results with progress bar
        print("\n📊 Consolidating dataset...")
        total_samples = sum(len(config_data) for config_data in results)
        
        sample_progress = tqdm(
            desc="Samples", 
            total=total_samples, 
            unit="sample",
            ncols=100
        )
        
        for config_data in results:
            self.data.extend(config_data)
            sample_progress.update(len(config_data))
        
        sample_progress.close()
        print(f"✅ Generated {len(self.data)} total samples")
    
    def _generate_config_data_worker(self, task):
        """Worker function for parallel data generation"""
        config_idx, building_config = task
        
        # Reduce verbose output - only show for first few configs or periodically
        show_progress = (config_idx < 3) or (config_idx % 10 == 0)
        
        # Try to use GPU-accelerated version if available and requested
        if self.use_gpu_processing and self.device.type in ('cuda', 'mps'):
            try:
                from aoa_amp_building_data_gpu import generate_building_training_data_gpu_batch
                # if show_progress:
                #     print(f"🔥 Using GPU acceleration for config {config_idx + 1}")
                
                config_data = generate_building_training_data_gpu_batch(
                    map_size=self.map_size,
                    grid_spacing=self.grid_spacing,
                    bs_grid_spacing=self.bs_grid_spacing,
                    building_configs=building_config,
                    save_dir=None,  # Don't save individual configs to disk
                    device=str(self.device),
                    batch_size=64,  # Larger batches to saturate GPU
                    num_workers=1,  # Single worker per config to avoid nested parallelism
                    progress_callback=None  # Disable inner progress for cleaner output
                )
            except ImportError:
                if show_progress:
                    print(f"⚠️  GPU acceleration not available, falling back to optimized CPU")
                # Fallback to optimized CPU version
                try:
                    from aoa_amp_building_data_optimized import generate_building_training_data_optimized
                    config_data = generate_building_training_data_optimized(
                        map_size=self.map_size,
                        grid_spacing=self.grid_spacing,
                        bs_grid_spacing=self.bs_grid_spacing,
                        building_configs=building_config,
                        save_dir=None,
                        num_workers=1,
                        use_multiprocessing=False
                    )
                except ImportError:
                    # Final fallback to original version
                    from aoa_amp_building_data import generate_building_training_data
                    config_data = generate_building_training_data(
                        map_size=self.map_size,
                        grid_spacing=self.grid_spacing,
                        bs_grid_spacing=self.bs_grid_spacing,
                        building_configs=building_config,
                        save_dir=None
                    )
        else:
            # Use CPU-optimized version
            try:
                from aoa_amp_building_data_optimized import generate_building_training_data_optimized
                if show_progress:
                    print(f"💻 Using optimized CPU for config {config_idx + 1}")
                config_data = generate_building_training_data_optimized(
                    map_size=self.map_size,
                    grid_spacing=self.grid_spacing,
                    bs_grid_spacing=self.bs_grid_spacing,
                    building_configs=building_config,
                    save_dir=None,
                    num_workers=1,
                    use_multiprocessing=False
                )
            except ImportError:
                # Fallback to original version
                if show_progress:
                    print(f" Using original CPU version for config {config_idx + 1}")
                from aoa_amp_building_data import generate_building_training_data
                config_data = generate_building_training_data(
                    map_size=self.map_size,
                    grid_spacing=self.grid_spacing,
                    bs_grid_spacing=self.bs_grid_spacing,
                    building_configs=building_config,
                    save_dir=None
                )
        
        # Add configuration index to each sample
        for sample in config_data:
            sample['config_idx'] = config_idx
            sample['num_buildings'] = len(building_config)
        
        return config_data
    
    def _convert_to_tensors_gpu(self):
        """Convert data to tensors using GPU acceleration"""
        print("\n🔄 Converting data to tensor format...")
        
        if self.use_gpu_processing and self.device.type in ('cuda', 'mps'):
            self._convert_with_gpu_batching()
        else:
            self._convert_with_cpu_batching()
    
    def _convert_with_gpu_batching(self):
        """Convert data using GPU batch processing"""
        self.tensor_data = []
        num_samples = len(self.data)
        
        print(f"🔥 Converting {num_samples} samples using GPU batching...")
        
        # Use tqdm for progress tracking
        with tqdm(total=num_samples, desc="GPU Conversion", unit="sample", ncols=100) as pbar:
            for i in range(0, num_samples, self.batch_size):
                batch_end = min(i + self.batch_size, num_samples)
                batch_samples = self.data[i:batch_end]
                
                # Process batch on GPU
                batch_tensors = []
                for sample in batch_samples:
                    tensor = self._convert_to_tensor_gpu(sample)
                    batch_tensors.append(tensor)
                
                # Stack tensors and keep on GPU
                if batch_tensors:
                    batch_tensor = torch.stack(batch_tensors)
                    
                    # Split back to individual tensors (still on GPU)
                    for j in range(batch_tensor.shape[0]):
                        self.tensor_data.append(batch_tensor[j])
                
                pbar.update(batch_end - i)
    
    def _convert_with_cpu_batching(self):
        """Convert data using CPU with parallel processing"""
        print(f"💻 Converting {len(self.data)} samples using CPU parallel processing...")
        
        # Use ThreadPoolExecutor for CPU-bound tensor conversion or sequential processing
        if self.num_workers > 0:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Use tqdm to track progress
                self.tensor_data = list(tqdm(
                    executor.map(self._convert_to_tensor, self.data),
                    total=len(self.data),
                    desc="CPU Conversion",
                    unit="sample",
                    ncols=100
                ))
        else:
            # Sequential processing for MPS backend
            self.tensor_data = []
            for data_item in tqdm(self.data, total=len(self.data), 
                                desc="CPU Conversion", unit="sample", ncols=100):
                tensor = self._convert_to_tensor(data_item)
                self.tensor_data.append(tensor)
        
        # Pin memory if requested
        if self.pin_memory and self.device.type == 'cuda':
            print("📌 Pinning memory for faster GPU transfer...")
            with tqdm(total=len(self.tensor_data), desc="Pinning memory", unit="sample", ncols=100) as pbar:
                pinned_data = []
                for tensor in self.tensor_data:
                    pinned_data.append(tensor.pin_memory())
                    pbar.update(1)
                self.tensor_data = pinned_data
    
    def _convert_to_tensor_gpu(self, sample_data):
        """
        Convert a single raw sample to a 12-channel tensor.

        Channel layout (channel-first, [-1, 1] when normalized):
            0..2  -> AoA path 1, 2, 3
            3..5  -> Amplitude path 1, 2, 3
            6..8  -> sin(phase) path 1, 2, 3
            9..11 -> cos(phase) path 1, 2, 3
        """
        aoa_maps = sample_data['aoa_maps']            # List of 3 2D arrays (degrees)
        amplitude_maps = sample_data['amplitude_maps']  # List of 3 2D arrays (dB)

        # Phase maps may be missing in legacy caches; fall back to zero phase.
        phase_maps = sample_data.get('phase_maps')
        if phase_maps is None:
            phase_maps = [np.zeros_like(m, dtype=np.float32) for m in aoa_maps]

        aoa_arr = np.stack(aoa_maps, axis=0).astype(np.float32)               # (3, H, W)
        amp_arr = np.stack(amplitude_maps, axis=0).astype(np.float32)         # (3, H, W)
        phase_arr = np.stack(phase_maps, axis=0).astype(np.float32)           # (3, H, W)

        # Phase -> sin/cos decomposition (already naturally bounded in [-1, 1]).
        sin_phase = np.sin(phase_arr).astype(np.float32)                      # (3, H, W)
        cos_phase = np.cos(phase_arr).astype(np.float32)                      # (3, H, W)

        all_maps = np.concatenate([aoa_arr, amp_arr, sin_phase, cos_phase], axis=0)  # (12, H, W)

        # Convert to tensor and keep on CPU for DataLoader compatibility.
        tensor = torch.from_numpy(all_maps).float().cpu()

        if self.normalize_data:
            # AoA channels (0..2): [-180, 180] deg -> [-1, 1]
            tensor[:3] = tensor[:3] / 180.0

            # Amplitude channels (3..5): typical [-90, -40] dB -> [-1, 1]
            amp_data = tensor[3:6]
            tensor[3:6] = 2 * (amp_data - (-90)) / ((-40) - (-90)) - 1

            # Phase channels (6..11) are already in [-1, 1] (sin/cos), no rescaling.

        tensor = self._apply_channel_selection(tensor)
        return tensor
        # """Convert single sample to tensor using GPU operations"""
        # aoa_maps = sample_data['aoa_maps']  # List of 3 2D arrays
        # amplitude_maps = sample_data['amplitude_maps']  # List of 3 2D arrays
        
        # # First, normalize all maps to 2D and get expected shape
        # processed_aoa_maps = []
        # for i, aoa_map in enumerate(aoa_maps):
        #     if aoa_map.ndim > 2:
        #         aoa_map = np.squeeze(aoa_map)
        #     processed_aoa_maps.append(aoa_map)
        
        # processed_amp_maps = []
        # for i, amp_map in enumerate(amplitude_maps):
        #     if amp_map.ndim > 2:
        #         amp_map = np.squeeze(amp_map)
        #     processed_amp_maps.append(amp_map)
        
        # # Use the first processed map to determine expected shape
        # expected_shape = processed_aoa_maps[0].shape
        # all_maps_list = processed_aoa_maps + processed_amp_maps
        
        # # Check for shape mismatches and fix them
        # fixed_maps = []
        # for i, map_data in enumerate(all_maps_list):
        #     if map_data.shape != expected_shape:
        #         try:
        #             from scipy.ndimage import zoom
        #             zoom_factors = [expected_shape[j] / map_data.shape[j] for j in range(len(map_data.shape))]
        #             if len(zoom_factors) != len(expected_shape):
        #                 if map_data.ndim > len(expected_shape):
        #                     map_data = np.squeeze(map_data)
        #                     zoom_factors = [expected_shape[j] / map_data.shape[j] for j in range(len(expected_shape))]
                    
        #             map_data = zoom(map_data, zoom_factors, order=1)
        #         except ImportError:
        #             if len(expected_shape) == 2 and map_data.ndim == 2:
        #                 old_h, old_w = map_data.shape
        #                 new_h, new_w = expected_shape
        #                 y_coords = np.linspace(0, old_h-1, new_h).astype(int)
        #                 x_coords = np.linspace(0, old_w-1, new_w).astype(int)
        #                 map_data = map_data[np.ix_(y_coords, x_coords)]
        #             else:
        #                 map_data = np.squeeze(map_data)
        #                 if map_data.ndim == 2 and map_data.shape != expected_shape:
        #                     old_h, old_w = map_data.shape
        #                     new_h, new_w = expected_shape
        #                     y_coords = np.linspace(0, old_h-1, new_h).astype(int)
        #                     x_coords = np.linspace(0, old_w-1, new_w).astype(int)
        #                     map_data = map_data[np.ix_(y_coords, x_coords)]
        #         except Exception as e:
        #             map_data = np.squeeze(map_data)
                        
        #     fixed_maps.append(map_data)
        
        # # Final verification that all shapes match
        # final_shapes = [map_data.shape for map_data in fixed_maps]
        # if len(set(final_shapes)) > 1:
        #     from collections import Counter
        #     shape_counts = Counter(final_shapes)
        #     most_common_shape = shape_counts.most_common(1)[0][0]
            
        #     final_fixed_maps = []
        #     for i, map_data in enumerate(fixed_maps):
        #         if map_data.shape != most_common_shape:
        #             try:
        #                 from scipy.ndimage import zoom
        #                 zoom_factors = [most_common_shape[j] / map_data.shape[j] for j in range(len(most_common_shape))]
        #                 map_data = zoom(map_data, zoom_factors, order=1)
        #             except:
        #                 if len(most_common_shape) == 2 and map_data.ndim == 2:
        #                     old_h, old_w = map_data.shape
        #                     new_h, new_w = most_common_shape
        #                     y_coords = np.linspace(0, old_h-1, new_h).astype(int)
        #                     x_coords = np.linspace(0, old_w-1, new_w).astype(int)
        #                     map_data = map_data[np.ix_(y_coords, x_coords)]
        #         final_fixed_maps.append(map_data)
        #     fixed_maps = final_fixed_maps
            
        #     final_shapes_check = [map_data.shape for map_data in fixed_maps]
        #     if len(set(final_shapes_check)) > 1:
        #         raise ValueError(f"Could not resolve shape mismatch. Final shapes: {final_shapes_check}")
        
        # # Stack all maps: shape = (6, H, W) - 3 AoA + 3 Amplitude
        # all_maps = np.stack(fixed_maps, axis=0)
        
        # # Convert to tensor but keep on CPU for DataLoader compatibility
        # tensor = torch.from_numpy(all_maps).float()
        
        # if self.normalize_data:
        #     # Normalize AoA maps (first 3 channels) from [-180, 180] to [-1, 1]
        #     tensor[:3] = tensor[:3] / 180.0
            
        #     # Normalize amplitude maps (last 3 channels) 
        #     # Amplitude is in dB, typically ranging from -90 to -40 dB
        #     amp_data = tensor[3:]
        #     tensor[3:] = 2 * (amp_data - (-90)) / ((-40) - (-90)) - 1
        #     tensor[3:] = torch.clamp(tensor[3:], -1, 1)
        
        # return tensor  # Return on CPU for DataLoader compatibility
    
    def _convert_to_tensor(self, sample_data):
        """
        CPU version of _convert_to_tensor_gpu — same 12-channel layout:
            0..2 AoA, 3..5 Amplitude, 6..8 sin(phase), 9..11 cos(phase).
        """
        aoa_maps = sample_data['aoa_maps']
        amplitude_maps = sample_data['amplitude_maps']

        phase_maps = sample_data.get('phase_maps')
        if phase_maps is None:
            phase_maps = [np.zeros_like(m, dtype=np.float32) for m in aoa_maps]

        aoa_arr = np.stack(aoa_maps, axis=0).astype(np.float32)
        amp_arr = np.stack(amplitude_maps, axis=0).astype(np.float32)
        phase_arr = np.stack(phase_maps, axis=0).astype(np.float32)

        sin_phase = np.sin(phase_arr).astype(np.float32)
        cos_phase = np.cos(phase_arr).astype(np.float32)

        all_maps = np.concatenate([aoa_arr, amp_arr, sin_phase, cos_phase], axis=0)

        if self.normalize_data:
            all_maps[:3] = all_maps[:3] / 180.0
            amp_data = all_maps[3:6]
            all_maps[3:6] = 2 * (amp_data - (-90)) / ((-40) - (-90)) - 1
            # Phase channels (6..11) already in [-1, 1] from sin/cos.

        all_maps = self._apply_channel_selection(all_maps)
        return torch.FloatTensor(all_maps)
    
    def _save_tensor_cache(self, cache_file):
        """Save tensor data to cache using HDF5 for better performance"""
        print(f"Saving tensor cache to {cache_file}...")
        
        if self.tensor_data:
            # Determine file format based on extension
            file_ext = os.path.splitext(cache_file)[1].lower()
            
            if file_ext == '.h5' or file_ext == '.hdf5':
                self._save_to_hdf5(cache_file)
            elif file_ext == '.npz':
                self._save_to_npz(cache_file)
            else:
                # Fallback to PyTorch format
                self._save_to_pytorch(cache_file)
    
    def _save_to_hdf5(self, cache_file):
        """Save tensor data to HDF5 format (most efficient for large datasets)"""
        try:
            import h5py
            
            print("💾 Saving to HDF5 format...")
            with h5py.File(cache_file, 'w') as f:
                # Save tensor data
                if self.tensor_data:
                    cpu_tensors = [tensor.cpu().numpy() for tensor in self.tensor_data]
                    stacked_tensors = np.stack(cpu_tensors, axis=0)
                    
                    # Use compression for smaller file sizes
                    f.create_dataset('tensor_data', data=stacked_tensors, 
                                   compression='gzip', compression_opts=6)
                
                # Save metadata
                f.attrs['num_samples'] = len(self.tensor_data)
                f.attrs['map_size'] = self.map_size
                f.attrs['grid_spacing'] = self.grid_spacing
                f.attrs['normalized'] = self.normalize_data
                f.attrs['amplitude_post_clip'] = self.amplitude_post_clip
                f.attrs['seed'] = self.seed
                f.attrs['num_channels'] = (
                    int(self.tensor_data[0].shape[0]) if self.tensor_data else 0
                )
                f.attrs['selected_channels'] = (
                    'all' if self.selected_channels is None
                    else ','.join(str(ch) for ch in self.selected_channels)
                )
                
                # Save building configurations (as JSON string for HDF5 compatibility)
                import json
                f.attrs['building_configs'] = json.dumps(self.building_config_sets)
                
            print(f"✅ Saved {len(self.tensor_data)} samples to HDF5 format")
            
        except ImportError:
            print("⚠️  h5py not available, falling back to PyTorch format")
            self._save_to_pytorch(cache_file.replace('.h5', '.pt').replace('.hdf5', '.pt'))
    
    def _save_to_npz(self, cache_file):
        """Save tensor data to compressed NumPy format"""
        print("💾 Saving to compressed NPZ format...")
        
        if self.tensor_data:
            cpu_tensors = [tensor.cpu().numpy() for tensor in self.tensor_data]
            stacked_tensors = np.stack(cpu_tensors, axis=0)
            
            # Save with compression
            np.savez_compressed(
                cache_file,
                tensor_data=stacked_tensors,
                map_size=self.map_size,
                grid_spacing=self.grid_spacing,
                normalized=self.normalize_data,
                amplitude_post_clip=np.int8(self.amplitude_post_clip),
                seed=self.seed,
                num_channels=np.int16(stacked_tensors.shape[1]),
                selected_channels=np.array(
                    self.selected_channels if self.selected_channels is not None else [-1],
                    dtype=np.int16,
                ),
                num_samples=len(self.tensor_data)
            )
            
        print(f"✅ Saved {len(self.tensor_data)} samples to NPZ format")
    
    def _save_to_pytorch(self, cache_file):
        """Save tensor data to PyTorch format (fallback)"""
        print("💾 Saving to PyTorch format...")
        
        # Move all tensors to CPU for saving
        cpu_tensors = [tensor.cpu() for tensor in self.tensor_data]
        stacked_tensors = torch.stack(cpu_tensors)
        
        # Save with metadata
        save_dict = {
            'tensor_data': stacked_tensors,
            'metadata': {
                'map_size': self.map_size,
                'grid_spacing': self.grid_spacing,
                'normalized': self.normalize_data,
                'amplitude_post_clip': self.amplitude_post_clip,
                'seed': self.seed,
                'selected_channels': (
                    self.selected_channels if self.selected_channels is not None else 'all'
                ),
                'num_samples': len(self.tensor_data)
            }
        }
        
        torch.save(save_dict, cache_file)
        print(f"✅ Saved {len(self.tensor_data)} samples to PyTorch format")
    
    def _load_tensor_cache(self, cache_file):
        """Load tensor data from the specified format"""
        file_ext = os.path.splitext(cache_file)[1].lower()
        
        if file_ext == '.h5' or file_ext == '.hdf5':
            return self._load_from_hdf5(cache_file)
        elif file_ext == '.npz':
            return self._load_from_npz(cache_file)
        else:  # PyTorch format
            return self._load_from_pytorch(cache_file)

    @staticmethod
    def _as_bool(value):
        """Best-effort cast of metadata values to bool."""
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return bool(int(value))
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='ignore')
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'1', 'true', 'yes', 'y', 'on'}:
                return True
            if normalized in {'0', 'false', 'no', 'n', 'off'}:
                return False
        return None

    def _read_tensor_cache_metadata(self, cache_file):
        """Read metadata fields from a tensor cache without loading full tensors."""
        file_ext = os.path.splitext(cache_file)[1].lower()
        metadata = {}

        try:
            if file_ext == '.h5' or file_ext == '.hdf5':
                import h5py
                with h5py.File(cache_file, 'r') as f:
                    metadata['normalized'] = f.attrs.get('normalized', None)
                    metadata['amplitude_post_clip'] = f.attrs.get('amplitude_post_clip', None)
                return metadata

            if file_ext == '.npz':
                npz_data = np.load(cache_file)
                metadata['normalized'] = npz_data['normalized'].item() if 'normalized' in npz_data else None
                metadata['amplitude_post_clip'] = (
                    npz_data['amplitude_post_clip'].item() if 'amplitude_post_clip' in npz_data else None
                )
                return metadata

            saved_data = torch.load(cache_file, map_location='cpu')
            if isinstance(saved_data, dict):
                cached_meta = saved_data.get('metadata', {})
                metadata['normalized'] = cached_meta.get('normalized', None)
                metadata['amplitude_post_clip'] = cached_meta.get('amplitude_post_clip', None)
            return metadata
        except Exception as exc:
            print(f"⚠️  Failed to read tensor cache metadata from {cache_file}: {exc}")
            return None

    def _is_tensor_cache_compatible(self, cache_file):
        """Check whether cached tensors match current normalization semantics."""
        metadata = self._read_tensor_cache_metadata(cache_file)
        if metadata is None:
            return False

        cached_normalized = self._as_bool(metadata.get('normalized'))
        expected_normalized = bool(self.normalize_data)
        if cached_normalized is not None and cached_normalized != expected_normalized:
            print(
                f"⚠️  Tensor cache normalized={cached_normalized} but expected "
                f"normalized={expected_normalized}"
            )
            return False

        if expected_normalized:
            cached_amp_clip = self._as_bool(metadata.get('amplitude_post_clip'))
            # Legacy caches did not persist this field and were created with clipping enabled.
            if cached_amp_clip is None:
                cached_amp_clip = True

            if cached_amp_clip != bool(self.amplitude_post_clip):
                print(
                    f"⚠️  Tensor cache amplitude_post_clip={cached_amp_clip} but expected "
                    f"amplitude_post_clip={self.amplitude_post_clip}"
                )
                return False

        # Channel-count check: the dataset now produces 12-channel tensors
        # (3 AoA + 3 Amplitude + 3 sin(phase) + 3 cos(phase)). Any cache built
        # with the older 6-channel layout must be rebuilt.
        expected_channels = (
            len(self.selected_channels) if self.selected_channels is not None else 12
        )
        cached_channels = metadata.get('num_channels')
        if cached_channels is not None and int(cached_channels) != int(expected_channels):
            print(
                f"⚠️  Tensor cache num_channels={cached_channels} but expected "
                f"num_channels={expected_channels} (phase added in v2)."
            )
            return False

        return True
    
    def _load_from_hdf5(self, cache_file):
        """Load tensor data from HDF5 format"""
        try:
            import h5py
            
            print(f"📖 Loading from HDF5 format: {cache_file}")
            with h5py.File(cache_file, 'r') as f:
                # Load tensor data
                tensor_array = f['tensor_data'][:]
                
                # Convert to list of tensors on target device
                tensor_list = []
                for i in range(tensor_array.shape[0]):
                    tensor = torch.from_numpy(tensor_array[i]).to(self.device)
                    tensor_list.append(tensor)
                
                print(f"✅ Loaded {len(tensor_list)} samples from HDF5")
                return tensor_list
                
        except ImportError:
            print("⚠️  h5py not available, cannot load HDF5 format")
            return None
        except Exception as e:
            print(f"❌ Error loading HDF5 cache: {e}")
            return None
    
    def _load_from_npz(self, cache_file):
        """Load tensor data from NPZ format"""
        try:
            print(f"📖 Loading from NPZ format: {cache_file}")
            data = np.load(cache_file)
            tensor_array = data['tensor_data']
            
            # Convert to list of tensors on target device
            tensor_list = []
            for i in range(tensor_array.shape[0]):
                tensor = torch.from_numpy(tensor_array[i]).to(self.device)
                tensor_list.append(tensor)
            
            print(f"✅ Loaded {len(tensor_list)} samples from NPZ")
            return tensor_list
            
        except Exception as e:
            print(f"❌ Error loading NPZ cache: {e}")
            return None
    
    def _load_from_pytorch(self, cache_file):
        """Load tensor data from PyTorch format"""
        try:
            print(f"📖 Loading from PyTorch format: {cache_file}")
            # Load directly to the target device
            saved_data = torch.load(cache_file, map_location=self.device)
            
            if isinstance(saved_data, dict) and 'tensor_data' in saved_data:
                tensor_data = saved_data['tensor_data']
            else:
                tensor_data = saved_data
            
            # Convert to list of tensors on device
            if isinstance(tensor_data, torch.Tensor):
                tensor_list = [tensor_data[i].to(self.device) for i in range(tensor_data.shape[0])]
            else:
                tensor_list = [tensor.to(self.device) for tensor in tensor_data]
            
            print(f"✅ Loaded {len(tensor_list)} samples from PyTorch format")
            return tensor_list
            
        except Exception as e:
            print(f"❌ Error loading PyTorch cache: {e}")
            return None
    
    def _create_metadata_list(self):
        """Create a parallel metadata list for quick access without loading full tensors"""
        self.metadata = []
        for sample in self.data:
            metadata = {
                'bs_pos': sample['bs_pos'],  # Base station position (x, y)
                'buildings': sample['buildings'],  # List of building configurations
                'map_size': sample['map_size'],  # Map dimensions
                'grid_spacing': sample['grid_spacing'],  # Grid spacing
                'config_idx': sample.get('config_idx', 0),  # Building configuration index
                'num_buildings': sample.get('num_buildings', len(sample['buildings']))  # Number of buildings
            }
            self.metadata.append(metadata)
        
        print(f"✅ Created metadata list with {len(self.metadata)} entries")
    
    def _print_dataset_statistics(self):
        """Print dataset statistics from whatever source is available.

        In the streaming flow self.data is None, so we read from self.metadata
        (built from raw HDF5 attrs). In the legacy non-streaming flow self.data
        is populated and we use that.
        """
        config_counts = {}
        source = self.metadata if (
            hasattr(self, 'metadata') and self.metadata
        ) else (self.data or [])

        for item in source:
            nb = item.get('num_buildings',
                          len(item.get('buildings', [])))
            config_counts[nb] = config_counts.get(nb, 0) + 1

        print("Dataset statistics:")
        for nb in sorted(config_counts.keys()):
            print(f"  - {config_counts[nb]} samples with {nb} building(s)")

        if self.tensor_data is not None and len(self.tensor_data) > 0:
            sample_device = (self.tensor_data[0].device
                             if hasattr(self.tensor_data[0], 'device') else 'CPU')
            print(f"Tensor data device: {sample_device}")
    
    def __len__(self):
        return len(self.tensor_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor or tuple: If return_index=False, returns shape (C, H, W)
                                  where C is len(selected_channels) or 6 when using all channels.
                                  If return_index=True, returns (tensor, idx)
                                  Channel order follows selected_channels.
        """
        sample = self.tensor_data[idx]
        
        # Tensor is on CPU, DataLoader will handle GPU transfer with pin_memory
        if self.transforms is not None:
            sample = self.transforms(sample)
        
        if self.return_index:
            return sample, idx
        else:
            return sample
    
    def get_sample_with_metadata(self, idx):
        """
        Get sample tensor along with its metadata.
        Useful for visualization and evaluation.
        
        Returns:
            tuple: (tensor, metadata) where
                - tensor: Shape (C, H, W) - the data tensor
                - metadata: dict with keys:
                    - 'bs_pos': Base station position (x, y)
                    - 'buildings': List of building configurations
                    - 'map_size': Map dimensions
                    - 'grid_spacing': Grid spacing
                    - 'config_idx': Building configuration index
                    - 'num_buildings': Number of buildings
        """
        tensor = self[idx]
        metadata = self.metadata[idx].copy()  # Return a copy to avoid modifications
        return tensor, metadata
    
    def get_metadata(self, idx):
        """
        Get metadata for a specific sample index.
        
        Args:
            idx: Sample index
            
        Returns:
            dict: Metadata dictionary with BS position, buildings, etc.
        """
        return self.metadata[idx].copy()
    
    def get_building_configurations(self):
        """Get all building configurations used in the dataset"""
        return self.building_config_sets
    
    def get_dataloader(self, batch_size=None, shuffle=True, **kwargs):
        """Get optimized DataLoader for this dataset"""
        if batch_size is None:
            batch_size = self.batch_size
        
        # # Check if tensors are on GPU
        # tensors_on_gpu = False
        # if self.tensor_data and len(self.tensor_data) > 0:
        #     sample_tensor = self.tensor_data[0]
        #     if hasattr(sample_tensor, 'device') and sample_tensor.device.type == 'cuda':
        #         tensors_on_gpu = True
        
        # if tensors_on_gpu:
        #     # Tensors are on GPU - must use single-threaded DataLoader
        #     dataloader_kwargs = {
        #         'batch_size': batch_size,
        #         'shuffle': shuffle,
        #         'num_workers': 0,  # Must be 0 when tensors are on GPU
        #         'pin_memory': False,  # Not needed when data is already on GPU
        #         'persistent_workers': False,
        #         # Don't set prefetch_factor when num_workers=0
        #     }
        #     print(f"DataLoader config: workers=0 (GPU tensors), pin_memory=False, device={self.device}")
        # else:
        #     # Tensors are on CPU - can use multiprocessing
        #     dataloader_kwargs = {
        #         'batch_size': batch_size,
        #         'shuffle': shuffle,
        #         'num_workers': min(self.num_workers or 4, 4),
        #         'pin_memory': True,  # Pin memory for fast GPU transfer
        #         'prefetch_factor': 2,
        #         'persistent_workers': True,
        #     }
        #     print(f"DataLoader config: workers={dataloader_kwargs['num_workers']}, pin_memory=True, device={self.device}")
        # Use consistent DataLoader configuration
        dataloader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': min(self.num_workers or 4, 4),  # Moderate parallelism
            'pin_memory': False, #self.device.type == 'cuda',  # Pin memory only for CUDA
            'prefetch_factor': 2,
            'persistent_workers': True,
        }
        # Allow overrides from kwargs
        dataloader_kwargs.update(kwargs)
        
        # Ensure prefetch_factor is not set when num_workers=0
        if dataloader_kwargs.get('num_workers', 0) == 0:
            dataloader_kwargs.pop('prefetch_factor', None)
        
        return torch.utils.data.DataLoader(self, **dataloader_kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test the GPU-optimized dataset
    dataset = AoAAmpBuildingDataset(
        root="./data/building_training_gpu",
        map_size=(64, 64),
        grid_spacing=1.0,
        bs_grid_spacing=15.0,  # Larger spacing for testing
        num_building_sets=1,  # Smaller number for testing
        building_distribution=(0, 1, 0),  # 4 configs each
        use_existing_data=False,  # Generate new data for testing
        device='auto',
        use_gpu_processing=True,
        batch_size=16,
        num_workers=4,
        seed=42
    )
    
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Sample device: {sample.device}")
        print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
        
        # Test optimized dataloader
        dataloader = dataset.get_dataloader(batch_size=8, shuffle=True)
        print(f"Created optimized DataLoader with {len(dataloader)} batches")
        
        # Test a batch
        batch = next(iter(dataloader))
        print(f"Batch shape: {batch.shape}")
        print(f"Batch device: {batch.device}")
