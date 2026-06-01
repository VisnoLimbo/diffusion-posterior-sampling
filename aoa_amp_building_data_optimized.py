"""
Optimized version of generate_building_training_data with performance improvements.
"""

import numpy as np
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
from aoa_amp_building import RayTracingAoAMap


def generate_sample_data(args):
    """Worker function to generate data for a single BS position"""
    bs_pos, map_size, grid_spacing, building_configs = args
    bs_x, bs_y = bs_pos
    
    # Create ray tracing model
    rt = RayTracingAoAMap(map_size=map_size, grid_spacing=grid_spacing)
    
    # Set base station position
    rt.set_base_station(bs_x, bs_y)
    
    # Add buildings
    for building in building_configs:
        rt.add_building(building['x'], building['y'], 
                      building['width'], building['height'])
    
    # Generate AoA, amplitude, and phase maps for 3 strongest paths
    aoa_maps, los_map = rt.generate_aoa_map(num_paths=3)
    amplitude_maps = rt.generate_amplitude_map(num_paths=3)
    phase_maps = rt.generate_phase_map(num_paths=3)

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


def generate_building_training_data_optimized(map_size=(100, 100), grid_spacing=1, bs_grid_spacing=1, 
                                            building_configs=None, save_dir='data/building_training',
                                            num_workers=None, use_multiprocessing=True, 
                                            chunk_size=None, progress_callback=None):
    """
    Optimized version of generate_building_training_data using parallel processing.
    
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
    num_workers : int or None
        Number of worker processes. If None, uses CPU count.
    use_multiprocessing : bool
        Whether to use multiprocessing (True) or threading (False)
    chunk_size : int or None
        Chunk size for parallel processing
    progress_callback : callable or None
        Function to call with progress updates
    
    Returns:
    --------
    dataset : list of dict
        Each dict contains 'bs_pos', 'aoa_maps', 'amplitude_maps', 'los_map'
    """
    
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
        num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    # Generate BS positions on a grid
    bs_x_positions = np.arange(5, map_size[0]-5, bs_grid_spacing)  # Avoid edges
    bs_y_positions = np.arange(5, map_size[1]-5, bs_grid_spacing)
    
    # Create all BS position combinations
    bs_positions = [(x, y) for x in bs_x_positions for y in bs_y_positions]
    total_samples = len(bs_positions)
    
    print(f"Generating {total_samples} training samples with {num_workers} workers...")
    print(f"Map size: {map_size}, Grid spacing: {grid_spacing}")
    print(f"BS grid spacing: {bs_grid_spacing}")
    print(f"Buildings: {len(building_configs)} buildings")
    print(f"Using {'multiprocessing' if use_multiprocessing else 'threading'}")
    
    # Prepare arguments for workers
    worker_args = [(pos, map_size, grid_spacing, building_configs) for pos in bs_positions]
    
    # Set chunk size
    if chunk_size is None:
        chunk_size = max(1, total_samples // (num_workers * 4))
    
    dataset = []
    
    # Use parallel processing
    executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
    
    with executor_class(max_workers=num_workers) as executor:
        # Submit all tasks
        if use_multiprocessing:
            results = executor.map(generate_sample_data, worker_args, chunksize=chunk_size)
        else:
            results = executor.map(generate_sample_data, worker_args)
        
        # Collect results with progress tracking
        for i, sample in enumerate(results):
            dataset.append(sample)
            
            if (i + 1) % 50 == 0 or (i + 1) == total_samples:
                print(f"Completed {i + 1}/{total_samples} samples ({(i + 1)/total_samples*100:.1f}%)")
                
                if progress_callback:
                    progress_callback(i + 1, total_samples)
    
    # Save dataset only if save_dir is provided
    if save_dir is not None:
        dataset_file = os.path.join(save_dir, f'training_data_grid{bs_grid_spacing}.pkl')
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {dataset_file}")
    
    print(f"Total samples: {len(dataset)}")
    return dataset


def generate_building_training_data_batch(map_size=(100, 100), grid_spacing=1, bs_grid_spacing=1, 
                                        building_configs=None, save_dir='data/building_training',
                                        batch_size=10):
    """
    Memory-efficient batch processing version for very large datasets.
    
    Parameters:
    -----------
    batch_size : int
        Number of samples to process in each batch
    """
    
    # Default building configuration if none provided
    if building_configs is None:
        building_configs = [
            {'x': 20, 'y': 20, 'width': 30, 'height': 15},
            {'x': 75, 'y': 56, 'width': 25, 'height': 19}
        ]
    
    # Create save directory if it doesn't exist and save_dir is not None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Generate BS positions on a grid
    bs_x_positions = np.arange(5, map_size[0]-5, bs_grid_spacing)
    bs_y_positions = np.arange(5, map_size[1]-5, bs_grid_spacing)
    bs_positions = [(x, y) for x in bs_x_positions for y in bs_y_positions]
    
    total_samples = len(bs_positions)
    dataset = []
    
    print(f"Generating {total_samples} samples in batches of {batch_size}...")
    
    # Process in batches
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_positions = bs_positions[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
        
        # Generate batch data using optimized function
        batch_args = [(pos, map_size, grid_spacing, building_configs) for pos in batch_positions]
        
        with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(batch_args))) as executor:
            batch_results = list(executor.map(generate_sample_data, batch_args))
        
        dataset.extend(batch_results)
        
        # Optional: Save intermediate results
        if save_dir is not None and len(dataset) % (batch_size * 5) == 0:
            temp_file = os.path.join(save_dir, f'temp_data_{len(dataset)}.pkl')
            with open(temp_file, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"Intermediate save: {len(dataset)} samples")
    
    # Save final dataset
    if save_dir is not None:
        dataset_file = os.path.join(save_dir, f'training_data_grid{bs_grid_spacing}.pkl')
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {dataset_file}")
        
        # Clean up temporary files
        for temp_file in os.listdir(save_dir):
            if temp_file.startswith('temp_data_'):
                os.remove(os.path.join(save_dir, temp_file))
    
    print(f"Total samples: {len(dataset)}")
    return dataset


# Backward compatibility - replace the original function
def generate_building_training_data(map_size=(100, 100), grid_spacing=1, bs_grid_spacing=1, 
                                  building_configs=None, save_dir='data/building_training'):
    """
    Wrapper for backward compatibility - uses optimized version by default.
    """
    return generate_building_training_data_optimized(
        map_size=map_size,
        grid_spacing=grid_spacing,
        bs_grid_spacing=bs_grid_spacing,
        building_configs=building_configs,
        save_dir=save_dir,
        num_workers=None,  # Auto-detect
        use_multiprocessing=True
    )


if __name__ == "__main__":
    # Test the optimized function
    import time
    
    print("Testing optimized data generation...")
    start_time = time.time()
    
    # Small test case
    test_dataset = generate_building_training_data_optimized(
        map_size=(50, 50),
        grid_spacing=5,
        bs_grid_spacing=10,
        building_configs=[
            {'x': 10, 'y': 10, 'width': 15, 'height': 10}
        ],
        save_dir=None,
        num_workers=4
    )
    
    end_time = time.time()
    print(f"Generated {len(test_dataset)} samples in {end_time - start_time:.2f} seconds")
    print(f"Speed: {len(test_dataset)/(end_time - start_time):.2f} samples/second")