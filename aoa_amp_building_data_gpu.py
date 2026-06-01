"""
GPU-accelerated data generation for building training dataset.
This module integrates the GPU ray tracing with the dataset generation pipeline.
"""

import torch
import numpy as np
import os
import json
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("Warning: h5py not available, falling back to pickle")
    import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import List, Dict, Tuple
import time

from aoa_amp_building_gpu import RayTracingAoAMapGPU

# Check CUDA availability
def get_best_device():
    """Get the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def generate_sample_data_gpu(args):
    """Generate a single training sample using GPU acceleration."""
    bs_pos, map_size, grid_spacing, building_configs, device = args
    
    # Create ray tracer (without verbose output during worker execution)
    rt = RayTracingAoAMapGPU(map_size=map_size, grid_spacing=grid_spacing, device=device, verbose=False)
    
    # Extract BS position coordinates
    bs_x, bs_y = bs_pos
    
    # Set base station position
    rt.set_base_station(bs_x, bs_y)
    
    # Add buildings
    for building in building_configs:
        rt.add_building(building['x'], building['y'], 
                      building['width'], building['height'])
    
    # Generate AoA, amplitude, and phase maps for 3 strongest paths using GPU
    aoa_maps, los_map = rt.generate_aoa_map_gpu(num_paths=3)
    amplitude_maps = rt.generate_amplitude_map_gpu(num_paths=3)
    phase_maps = rt.generate_phase_map_gpu(num_paths=3)

    # Store sample
    sample = {
        'bs_pos': np.array([bs_x, bs_y]),
        'aoa_maps': aoa_maps,
        'amplitude_maps': amplitude_maps,
        'phase_maps': phase_maps,
        'los_map': los_map,
        'buildings': building_configs,
        'map_size': map_size,
        'grid_spacing': grid_spacing
    }

    return sample


def save_dataset_hdf5(dataset, filepath):
    """Save dataset in HDF5 format for better performance with large datasets."""
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 storage. Install with: pip install h5py")
    
    print(f"💾 Saving {len(dataset)} samples to HDF5 format: {filepath}")
    
    with h5py.File(filepath, 'w') as f:
        # Create groups for organized storage
        samples_group = f.create_group('samples')
        metadata_group = f.create_group('metadata')
        
        num_samples = len(dataset)
        
        # Get dimensions from first sample
        if num_samples > 0:
            first_sample = dataset[0]
            map_shape = first_sample['aoa_maps'].shape
            
            # Create datasets for efficient storage
            # Base station positions
            bs_positions = f.create_dataset('bs_positions', 
                                          (num_samples, 2), 
                                          dtype=np.float32,
                                          compression='gzip', compression_opts=6)
            
            # AoA maps (samples, num_paths, height, width)
            aoa_maps = f.create_dataset('aoa_maps', 
                                       (num_samples,) + map_shape,
                                       dtype=np.float32,
                                       compression='gzip', compression_opts=6)
            
            # Amplitude maps (samples, num_paths, height, width)
            amplitude_maps = f.create_dataset('amplitude_maps', 
                                            (num_samples,) + map_shape,
                                            dtype=np.float32,
                                            compression='gzip', compression_opts=6)
            
            # LOS maps (samples, height, width)
            los_shape = first_sample['los_map'].shape
            los_maps = f.create_dataset('los_maps', 
                                       (num_samples,) + los_shape,
                                       dtype=np.float32,
                                       compression='gzip', compression_opts=6)
            
            # Store data efficiently
            for i, sample in enumerate(dataset):
                bs_positions[i] = sample['bs_pos']
                aoa_maps[i] = sample['aoa_maps']
                amplitude_maps[i] = sample['amplitude_maps']
                los_maps[i] = sample['los_map']
            
            # Store metadata
            metadata_group.attrs['num_samples'] = num_samples
            metadata_group.attrs['map_size'] = first_sample['map_size']
            metadata_group.attrs['grid_spacing'] = first_sample['grid_spacing']
            
            # Store building configurations as JSON string
            buildings_json = json.dumps(first_sample['buildings'])
            metadata_group.attrs['buildings'] = buildings_json
            
            # Store format version for future compatibility
            metadata_group.attrs['format_version'] = '1.0'
            metadata_group.attrs['created_by'] = 'aoa_amp_building_data_gpu.py'
            
    print(f"✅ Dataset saved successfully to {filepath}")


def load_dataset_hdf5(filepath):
    """Load dataset from HDF5 format."""
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 loading. Install with: pip install h5py")
    
    print(f"📖 Loading dataset from HDF5: {filepath}")
    
    dataset = []
    
    with h5py.File(filepath, 'r') as f:
        # Load metadata
        metadata = f['metadata']
        num_samples = metadata.attrs['num_samples']
        map_size = tuple(metadata.attrs['map_size'])
        grid_spacing = float(metadata.attrs['grid_spacing'])
        buildings = json.loads(metadata.attrs['buildings'])
        
        # Load data arrays
        bs_positions = f['bs_positions'][:]
        aoa_maps = f['aoa_maps'][:]
        amplitude_maps = f['amplitude_maps'][:]
        los_maps = f['los_maps'][:]
        
        # Reconstruct dataset
        for i in range(num_samples):
            sample = {
                'bs_pos': bs_positions[i],
                'aoa_maps': aoa_maps[i],
                'amplitude_maps': amplitude_maps[i],
                'los_map': los_maps[i],
                'buildings': buildings,
                'map_size': map_size,
                'grid_spacing': grid_spacing
            }
            dataset.append(sample)
    
    print(f"✅ Loaded {len(dataset)} samples from HDF5")
    return dataset


def save_dataset_pickle(dataset, filepath):
    """Save dataset in pickle format (fallback)."""
    print(f"💾 Saving {len(dataset)} samples to pickle format: {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"✅ Dataset saved to {filepath}")


def load_dataset_pickle(filepath):
    """Load dataset from pickle format (fallback)."""
    print(f"📖 Loading dataset from pickle: {filepath}")
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    print(f"✅ Loaded {len(dataset)} samples from pickle")
    return dataset

def generate_batch_data_gpu_optimized(bs_positions_batch, map_size, grid_spacing, building_configs, device):
    """
    Generate samples for a batch of BS positions on the GPU.

    Each BS position is now just a handful of vectorized tensor ops (see
    RayTracingAoAMapGPU._generate_ranked_path_maps_vectorized), so we simply
    create one ray tracer (buildings added once) and loop the positions.

    set_base_station() resets the per-position cache, and the cache key
    includes the BS position, so AoA / amplitude / phase are always computed
    for the correct BS position (this fixes the old stale-cache bug where
    every sample in a batch reused the first position's AoA map).

    Returns:
    --------
    batch_results : list of dict
        Each dict contains 'bs_pos', 'aoa_maps', 'amplitude_maps',
        'phase_maps', 'los_map', 'buildings', 'map_size', 'grid_spacing'.
    """

    if device != 'cuda' and device != 'mps':
        return [generate_sample_data_cpu((pos, map_size, grid_spacing, building_configs, device))
                for pos in bs_positions_batch]

    # Create ONE ray tracer instance; buildings are added once for the batch.
    rt = RayTracingAoAMapGPU(map_size=map_size, grid_spacing=grid_spacing, device=device, verbose=False)
    for building in building_configs:
        rt.add_building(building['x'], building['y'],
                        building['width'], building['height'])

    batch_results = []

    for bs_pos in bs_positions_batch:
        bs_x, bs_y = float(bs_pos[0]), float(bs_pos[1])

        # set_base_station() clears the ranked-map cache for this BS position.
        rt.set_base_station(bs_x, bs_y)

        # First call computes (and caches) all three map types + LOS for this
        # BS position; the next two calls are cheap cache hits.
        aoa_maps, los_map = rt.generate_aoa_map_gpu(num_paths=3)
        amplitude_maps = rt.generate_amplitude_map_gpu(num_paths=3)
        phase_maps = rt.generate_phase_map_gpu(num_paths=3)

        batch_results.append({
            'bs_pos': np.array([bs_x, bs_y]),
            'aoa_maps': aoa_maps,            # list of 3 2D numpy arrays
            'amplitude_maps': amplitude_maps,  # list of 3 2D numpy arrays
            'phase_maps': phase_maps,        # list of 3 2D numpy arrays
            'los_map': los_map,
            'buildings': building_configs,
            'map_size': map_size,
            'grid_spacing': grid_spacing,
        })

    # Clear GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()

    return batch_results

def generate_sample_data_cpu(args):
    """CPU version for fallback - should be separate from GPU version."""
    bs_pos, map_size, grid_spacing, building_configs, device = args
    
    # Use CPU-optimized ray tracer
    from aoa_amp_building_data import RayTracingAoAMap  # CPU version
    
    rt = RayTracingAoAMap(map_size=map_size, grid_spacing=grid_spacing)
    bs_x, bs_y = bs_pos
    rt.set_base_station(bs_x, bs_y)
    
    for building in building_configs:
        rt.add_building(building['x'], building['y'], 
                      building['width'], building['height'])
    
    aoa_maps, los_map = rt.generate_aoa_map(num_paths=3)
    amplitude_maps = rt.generate_amplitude_map(num_paths=3)
    phase_maps = rt.generate_phase_map(num_paths=3)

    return {
        'bs_pos': np.array([bs_x, bs_y]),
        'aoa_maps': aoa_maps,
        'amplitude_maps': amplitude_maps,
        'phase_maps': phase_maps,
        'los_map': los_map,
        'buildings': building_configs,
        'map_size': map_size,
        'grid_spacing': grid_spacing
    }


def generate_building_training_data_gpu_batch(map_size=(128, 128), grid_spacing=1, bs_grid_spacing=5, 
                                            building_configs=None, save_dir='data/building_training',
                                            device='auto', batch_size=8, num_workers=None,
                                            progress_callback=None, verbose=False):
    """
    Generate training dataset using GPU-accelerated ray tracing with batch processing.
    
    Parameters:
    -----------
    map_size : tuple
        Size of the map in meters (width, height)
    grid_spacing : float
        Grid spacing for UE positions in meters
    bs_grid_spacing : float
        Grid spacing for BS positions in meters
    building_configs : list of dict
        List of building configurations [{'x': x, 'y': y, 'width': w, 'height': h}]
    save_dir : str or None
        Directory to save the training data. If None, data won't be saved to disk.
    device : str
        Device to use ('cuda', 'cpu', or 'auto')
    batch_size : int
        Number of base station positions to process in each batch
    num_workers : int or None
        Number of worker processes. If None, uses 1 for GPU or CPU count for CPU.
    progress_callback : callable or None
        Function to call with progress updates
    verbose : bool
        Whether to show detailed progress information
    
    Returns:
    --------
    dataset : list of dict
        Each dict contains 'bs_pos', 'aoa_maps', 'amplitude_maps', 'los_map'
    """
    
    # Setup device
    if device == 'auto':
        device = get_best_device()
    
    # Default building configuration if none provided
    if building_configs is None:
        building_configs = [
            {'x': 20, 'y': 20, 'width': 30, 'height': 15},
            {'x': 75, 'y': 56, 'width': 25, 'height': 19}
        ]
    
    # Create save directory if it doesn't exist and save_dir is not None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Set number of workers
    if num_workers is None:
        if str(device) in ('cuda', 'mps'):
            num_workers = 1  # GPU works best with fewer processes
        else:
            num_workers = min(mp.cpu_count() // 2, 4)
    
    # Generate BS positions on a grid
    # Handle both tuple and single integer map_size
    if isinstance(map_size, (tuple, list)):
        map_x, map_y = map_size[0], map_size[1]
    else:
        map_x, map_y = map_size, map_size
    
    bs_x_positions = np.arange(5, map_x-5, bs_grid_spacing)  # Avoid edges
    bs_y_positions = np.arange(5, map_y-5, bs_grid_spacing)
    
    # Create all BS position combinations
    bs_positions = [(x, y) for x in bs_x_positions for y in bs_y_positions]
    total_samples = len(bs_positions)
    
    # Only show details if verbose mode is enabled
    if verbose:
        print(f"Generating {total_samples} training samples using GPU acceleration...")
        print(f"Map size: {map_size}, Grid spacing: {grid_spacing}")
        print(f"BS grid spacing: {bs_grid_spacing}")
        print(f"Buildings: {len(building_configs)} buildings")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}, Workers: {num_workers}")
    
    dataset = []
    start_time = time.time()
    
    device_str = str(device)
    if device_str in ('cuda', 'mps'):
        
        # Use the optimized batch processing
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_positions = bs_positions[i:batch_end]
            
            # Process batch with single GPU instance
            batch_samples = generate_batch_data_gpu_optimized(
                batch_positions, map_size, grid_spacing, building_configs, device
            )
            dataset.extend(batch_samples)
            
            if verbose:
                print(f"✅ GPU batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size} completed")
            
            if progress_callback:
                progress_callback(batch_end, total_samples)
    
    else:
        # CPU processing - use multiprocessing
        if verbose:
            print("Using CPU with multiprocessing...")
        
        # Prepare arguments for workers
        worker_args = [(pos, map_size, grid_spacing, building_configs, device) for pos in bs_positions]
        
        # Use ThreadPoolExecutor for CPU (since we're using PyTorch which handles threading well)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Process in batches to provide progress updates
            for i in range(0, len(worker_args), batch_size):
                batch_end = min(i + batch_size, len(worker_args))
                batch_args = worker_args[i:batch_end]
                
                # Submit batch
                batch_results = list(executor.map(generate_sample_data_gpu, batch_args))
                dataset.extend(batch_results)
                
                # Only show progress if verbose or if callback is provided
                if progress_callback:
                    progress_callback(batch_end, total_samples)
    
    total_time = time.time() - start_time
    
    # Save dataset only if save_dir is provided
    if save_dir is not None:
        # Use HDF5 format if available, fallback to pickle
        if HDF5_AVAILABLE:
            dataset_file = os.path.join(save_dir, f'training_data_gpu_grid{bs_grid_spacing}.h5')
            print(f"Saving dataset to HDF5 format: {dataset_file}...")
            save_dataset_hdf5(dataset, dataset_file)
        else:
            dataset_file = os.path.join(save_dir, f'training_data_gpu_grid{bs_grid_spacing}.pkl')
            print(f"HDF5 not available, using pickle format: {dataset_file}...")
            save_dataset_pickle(dataset, dataset_file)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average speed: {len(dataset)/total_time:.2f} samples/sec")
    
    # GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleaned up")
    
    return dataset


def load_training_data(data_dir, bs_grid_spacing=5):
    """
    Load training data from either HDF5 or pickle format.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the training data
    bs_grid_spacing : float
        Grid spacing for BS positions (used in filename)
    
    Returns:
    --------
    dataset : list of dict
        Loaded dataset, or None if no file found
    """
    # Try HDF5 first (preferred format)
    hdf5_file = os.path.join(data_dir, f'training_data_gpu_grid{bs_grid_spacing}.h5')
    if os.path.exists(hdf5_file):
        if HDF5_AVAILABLE:
            return load_dataset_hdf5(hdf5_file)
        else:
            print(f"Found HDF5 file {hdf5_file} but h5py not available")
    
    # Fallback to pickle
    pickle_file = os.path.join(data_dir, f'training_data_gpu_grid{bs_grid_spacing}.pkl')
    if os.path.exists(pickle_file):
        return load_dataset_pickle(pickle_file)
    
    print(f"No training data found in {data_dir}")
    return None


def benchmark_gpu_vs_cpu(map_size=(50, 50), grid_spacing=5, bs_grid_spacing=15, 
                       building_configs=None, num_samples=9):
    """
    Benchmark GPU vs CPU performance for ray tracing
    """
    if building_configs is None:
        building_configs = [
            {'x': 10, 'y': 10, 'width': 15, 'height': 10},
            {'x': 25, 'y': 30, 'width': 12, 'height': 8}
        ]
    
    print("=" * 60)
    print("GPU vs CPU Ray Tracing Benchmark")
    print("=" * 60)
    
    # Generate limited BS positions for testing
    bs_x_positions = np.arange(5, map_size[0]-5, bs_grid_spacing)[:3]  # Limit to 3
    bs_y_positions = np.arange(5, map_size[1]-5, bs_grid_spacing)[:3]  # Limit to 3
    bs_positions = [(x, y) for x in bs_x_positions for y in bs_y_positions]
    
    results = {}
    
    # Test CPU
    print("\nTesting CPU performance...")
    start_time = time.time()
    cpu_dataset = []
    
    for pos in bs_positions:
        args = (pos, map_size, grid_spacing, building_configs, 'cpu')
        sample = generate_sample_data_gpu(args)
        cpu_dataset.append(sample)
    
    cpu_time = time.time() - start_time
    results['cpu'] = {'time': cpu_time, 'samples': len(cpu_dataset)}
    
    print(f"CPU: {len(cpu_dataset)} samples in {cpu_time:.3f}s ({len(cpu_dataset)/cpu_time:.2f} samples/s)")
    
    # Test GPU if available
    if torch.cuda.is_available():
        print("\nTesting GPU performance...")
        start_time = time.time()
        gpu_dataset = []
        
        for pos in bs_positions:
            args = (pos, map_size, grid_spacing, building_configs, 'cuda')
            sample = generate_sample_data_gpu(args)
            gpu_dataset.append(sample)
        
        gpu_time = time.time() - start_time
        results['gpu'] = {'time': gpu_time, 'samples': len(gpu_dataset)}
        
        print(f"GPU: {len(gpu_dataset)} samples in {gpu_time:.3f}s ({len(gpu_dataset)/gpu_time:.2f} samples/s)")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"\nGPU Speedup: {speedup:.2f}x")
        
        # Memory usage
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
    else:
        print("\nGPU not available for testing")
    
    print("=" * 60)
    return results


if __name__ == "__main__":
    # Test the GPU-accelerated data generation
    print("Testing GPU-accelerated building training data generation...")
    st = time.time()
    # Small test case
    test_dataset = generate_building_training_data_gpu_batch(
        map_size=(128, 128),
        grid_spacing=1,
        bs_grid_spacing=15,
        building_configs=[
            {'x': 10, 'y': 10, 'width': 15, 'height': 10},
            {'x': 25, 'y': 30, 'width': 12, 'height': 8}
        ],
        save_dir=None,
        device='auto',
        batch_size=4,
        num_workers=4
    )
    et = time.time()
    print(f"Test completed in {et - st:.2f}s")
    
    print(f"\nGenerated {len(test_dataset)} samples")
    if test_dataset:
        sample = test_dataset[0]
        print(f"Sample shape - AoA: {sample['aoa_maps'][0].shape}")
        print(f"Sample shape - Amplitude: {sample['amplitude_maps'][0].shape}")
        print(f"LOS percentage: {np.sum(sample['los_map'])/sample['los_map'].size*100:.1f}%")
    
    # Run benchmark
    print("\n" + "="*60)
    benchmark_results = benchmark_gpu_vs_cpu()