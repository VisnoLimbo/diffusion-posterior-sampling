"""
Training script for diffusion model on AoA/Amplitude data with buildings.
This script trains on the ray tracing data with fixed buildings and strongest 3 paths.
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import logging
from tqdm import tqdm
import numpy as np

from guided_diffusion.gaussian_diffusion import create_sampler, extract_and_expand
from guided_diffusion.unet import create_model
from torch.utils.data import DataLoader, Subset
from data.dataloader import get_dataset, get_dataloader
from data.aoa_amp_building_dataset import AoAAmpBuildingDataset  # Import to register the dataset
from util.logger import get_logger
import torch.multiprocessing as mp


def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, step, loss, checkpoint_dir, filename="checkpoint.pt"):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'loss': loss
    }
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, checkpoint_path, map_location=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        loss = checkpoint['loss']
    else:
        # Checkpoint is the state_dict directly
        model.load_state_dict(checkpoint)
        step = 0
        loss = 0.0
    
    print(f"Checkpoint loaded from {checkpoint_path}, step: {step}, loss: {loss:.4f}")
    return step, loss


def compute_loss(model, diffusion, batch, device):
    """Compute diffusion training loss"""
    
    batch = batch.to(device)
    
    # Sample random timesteps
    batch_size = batch.shape[0]
    timesteps = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
    
    # Generate noise
    noise = torch.randn_like(batch)
    
    # Add noise to data using the diffusion process coefficients
    coef1 = extract_and_expand(diffusion.sqrt_alphas_cumprod, timesteps, batch)
    coef2 = extract_and_expand(diffusion.sqrt_one_minus_alphas_cumprod, timesteps, batch)
    noisy_batch = coef1 * batch + coef2 * noise
    
    # Predict noise with the model
    model_output = model(noisy_batch, timesteps)
    
    # Handle learned variance case
    if model_output.shape[1] == batch.shape[1] * 2:  # learn_sigma=True
        # Split the output: first half is noise prediction, second half is variance
        predicted_noise = model_output[:, :batch.shape[1]]
        predicted_variance = model_output[:, batch.shape[1]:]
        target = noise
        
        # Only compute loss on the noise prediction part for now
        loss = F.mse_loss(predicted_noise, target, reduction='mean')
    else:
        # Standard case: model predicts just the noise
        target = noise
        loss = F.mse_loss(model_output, target, reduction='mean')
    
    return loss


def train_step(model, diffusion, optimizer, batch, device):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    loss = compute_loss(model, diffusion, batch, device)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def sample_and_save(model, diffusion, device, save_path, num_samples=4, data_channels=6, image_size=100):
    """Generate and save samples for visualization"""
    model.eval()
    
    with torch.no_grad():
        # Start from random noise
        shape = (num_samples, data_channels, image_size, image_size)
        img = torch.randn(shape, device=device)
        
        # Simple sampling loop (simplified DDPM sampling)
        for i in reversed(range(0, diffusion.num_timesteps, 50)):  # Sample every 50 steps for speed
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            
            # Get model prediction
            with torch.no_grad():
                model_output = model(img, t)
                
                # Handle learned variance case
                if model_output.shape[1] == data_channels * 2:
                    predicted_noise = model_output[:, :data_channels]
                else:
                    predicted_noise = model_output
                
                # Simple denoising step
                if i > 0:
                    # Add some noise for non-final steps
                    noise = torch.randn_like(img)
                    
                    # Use diffusion coefficients for proper denoising
                    alpha_t = extract_and_expand(diffusion.alphas_cumprod, t, img)
                    beta_t = extract_and_expand(diffusion.betas, t, img)
                    
                    # Simplified denoising (approximation of DDPM sampling)
                    img = (img - beta_t * predicted_noise / torch.sqrt(1 - alpha_t)) / torch.sqrt(1 - beta_t)
                    
                    if i > 1:
                        img += torch.sqrt(beta_t) * noise
                else:
                    # Final step
                    img = img - predicted_noise
        
        # Convert to numpy and save
        samples_np = img.cpu().numpy()
        np.save(save_path, samples_np)
        print(f"Samples saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model on AoA/Amplitude data with buildings")
    parser.add_argument('--model_config', type=str, default='configs/aoa_amp_building_config.yaml',
                       help='Path to model configuration file')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml',
                       help='Path to diffusion configuration file')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/aoa_amp_building',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume from, or "latest" to auto-find latest checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use')
    parser.add_argument('--log_dir', type=str, default='./logs/aoa_amp_building',
                       help='Directory for tensorboard logs')
    parser.add_argument('--generate_data', action='store_true', default=False,
                       help='Generate new training data before training')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger()
    
    # Device setup
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    
    # Extract training parameters
    batch_size = model_config.get('batch_size', 8)
    learning_rate = float(model_config.get('learning_rate', 1e-4))
    num_epochs = model_config.get('num_epochs', 500)
    save_interval = model_config.get('save_interval', 12000)
    log_interval = model_config.get('log_interval', 50)
    epoch_save_interval = model_config.get('epoch_save_interval', 1)
    
    logger.info(f"Training parameters: lr={learning_rate}, batch_size={batch_size}, epochs={num_epochs}")
    
    # Get data channels from config
    data_channels = model_config.get('data_channels', 6)  # 6 channels for building data
    image_size = model_config.get('image_size', 100)
    
    # Create model - pass data_channels so it creates correct architecture
    model_params = {
        k: v
        for k, v in model_config.items()
        if k
        not in [
            'batch_size',
            'learning_rate',
            'num_epochs',
            'save_interval',
            'epoch_save_interval',
            'log_interval',
            'dataset',
            'dataloader',
        ]
    }
    
    # Ensure data_channels is passed to create_model
    model_params['data_channels'] = data_channels
    
    model = create_model(**model_params)
    
    model = model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create diffusion sampler
    diffusion = create_sampler(**diffusion_config)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Setup dataset and dataloader
    dataset_config = model_config['dataset']
    
    # If --generate_data is set, force regeneration by disabling cached data
    if args.generate_data:
        logger.info("Forcing regeneration of training data...")
        dataset_config['use_existing_data'] = False
    dataloader_config = model_config.get('dataloader', {})
    
    full_dataset = get_dataset(**dataset_config)
    total_samples = len(full_dataset)
    logger.info(f"Full dataset size: {total_samples}")
    
    # ==========================================================================
    # TRAIN/TEST SPLIT BY BUILDING CONFIGURATION
    # Reads num_building_sets and building_distribution from config so the
    # split adapts automatically (works for both small CPU tests and full runs).
    # Training: first ~90% configs per group
    # Testing:  last ~10% configs per group (at least 1)
    # ==========================================================================

    building_dist = dataset_config.get('building_distribution', [20, 20, 20])
    num_building_configs = sum(building_dist)
    configs_per_group = building_dist[0]  # assumes equal groups
    test_configs_per_group = max(1, configs_per_group // 10)  # ~10%, at least 1
    train_configs_per_group = configs_per_group - test_configs_per_group
    
    # Calculate samples per building configuration
    samples_per_config = total_samples // num_building_configs
    logger.info(f"Samples per building config: {samples_per_config}")
    
    # Build train indices: configs 0-17, 20-37, 40-57
    train_indices = []
    for group in range(3):  # 3 groups: 1-building, 2-building, 3-building
        group_start_config = group * configs_per_group  # 0, 20, 40
        for config_offset in range(train_configs_per_group):  # 0-17
            config_id = group_start_config + config_offset
            sample_start = config_id * samples_per_config
            sample_end = sample_start + samples_per_config
            train_indices.extend(range(sample_start, sample_end))
    
    logger.info(f"Building configs: {num_building_configs} total, {configs_per_group} per group")
    logger.info(f"Train/test split: {train_configs_per_group}/{test_configs_per_group} per group")
    logger.info(f"Training samples: {len(train_indices)} ({train_configs_per_group * 3} configs x {samples_per_config} samples)")
    
    # Create training subset
    train_dataset = Subset(full_dataset, train_indices)
    
    # Create dataloader
    dl_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': dataloader_config.get('num_workers', 0),
        'pin_memory': dataloader_config.get('pin_memory', False),
    }
    train_dataloader = DataLoader(train_dataset, **dl_kwargs)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Number of batches: {len(train_dataloader)}")
    
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        # Handle case where dataset returns (tensor, idx) tuple
        sample_tensor = sample[0] if isinstance(sample, (tuple, list)) else sample
        logger.info(f"Sample shape: {sample_tensor.shape}")
        logger.info(f"Sample range: [{sample_tensor.min():.3f}, {sample_tensor.max():.3f}]")
        logger.info(f"Sample range: [{train_dataset[0].min():.3f}, {train_dataset[0].max():.3f}]")
    
    # Setup tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Resume from checkpoint if specified
    start_step = 0
    start_epoch = 0

    def _epoch_from_ckpt(path):
        """Extract the epoch number from a 'checkpoint_epoch_N.pt' name (-1 if absent)."""
        import re
        m = re.search(r'checkpoint_epoch_(\d+)\.pt$', os.path.basename(path))
        return int(m.group(1)) if m else -1

    if args.resume:
        resume_path = args.resume
        if resume_path == 'latest' or not os.path.isfile(resume_path):
            # Pick the latest checkpoint by EPOCH NUMBER (numeric), not alphabetical
            # order, so checkpoint_epoch_100 correctly beats checkpoint_epoch_85.
            ckpt_files = glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint_epoch_*.pt'))
            if ckpt_files:
                resume_path = max(ckpt_files, key=_epoch_from_ckpt)
                logger.info(f"Found latest checkpoint: {resume_path}")
            else:
                logger.info("No checkpoint found to resume from. Starting from scratch.")
                resume_path = None
        if resume_path:
            start_step, _ = load_checkpoint(model, optimizer, resume_path, map_location=device)
            # Continue the epoch counter from the checkpoint so a resumed run trains
            # epochs (start_epoch+1 .. num_epochs) and saves correctly-named, non-
            # overwriting checkpoints instead of restarting at "Epoch 1".
            ep = _epoch_from_ckpt(resume_path)
            if ep >= 0:
                start_epoch = ep
                logger.info(
                    f"Resuming from epoch {start_epoch} -> will train epochs "
                    f"{start_epoch + 1}..{num_epochs}"
                )
            else:
                logger.info("Could not parse epoch from checkpoint name; epoch count starts at 0.")

    # Training loop
    logger.info("Starting training...")
    step = start_step

    if start_epoch >= num_epochs:
        logger.info(
            f"Checkpoint epoch ({start_epoch}) >= num_epochs ({num_epochs}); "
            f"nothing left to train. Exiting."
        )
        writer.close()
        return

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Training step
            loss = train_step(model, diffusion, optimizer, batch, device)
            
            epoch_loss += loss
            num_batches += 1
            step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            # # Logging
            # if step % log_interval == 0:
            #     writer.add_scalar('Train/Loss', loss, step)
            #     logger.info(f"Step {step}, Loss: {loss:.4f}")
            
            # # Save checkpoint 
            # if step % save_interval == 0:
            #     avg_loss = epoch_loss / num_batches
            #     save_checkpoint(model, optimizer, step, avg_loss, args.checkpoint_dir, 
            #                   f"checkpoint_step_{step}.pt")
                
            #     # Generate samples for visualization
            #     sample_path = os.path.join(args.checkpoint_dir, f"samples_step_{step}.npy")
            #     sample_and_save(model, diffusion, device, sample_path, 
            #                   data_channels=data_channels, image_size=image_size)
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        writer.add_scalar('Train/EpochLoss', avg_epoch_loss, epoch)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save end-of-epoch checkpoint
        if (epoch + 1) % epoch_save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                step,
                avg_epoch_loss,
                args.checkpoint_dir,
                f"checkpoint_epoch_{epoch+1}.pt"
            )

            sample_path = os.path.join(
                args.checkpoint_dir,
                f"samples_epoch_{epoch+1}.npy"
            )
            sample_and_save(
                model,
                diffusion,
                device,
                sample_path,
                data_channels=data_channels,
                image_size=image_size
            )
    
    logger.info("Training completed!")
    writer.close()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
