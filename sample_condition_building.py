"""Conditional sampling script tailored to the AoA/Amplitude building dataset."""

from functools import partial
from collections import defaultdict
import argparse
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import yaml

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from torch.utils.data import DataLoader, Subset
from data.dataloader import get_dataset, get_dataloader
from data.aoa_amp_building_dataset import AoAAmpBuildingDataset  # noqa: F401
from util.img_utils import clear_color, mask_generator, normalize_np
from util.logger import get_logger


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_tensor_channels(
    tensor,
    out_dir: str,
    base_name: str,
    cmap: str = 'viridis',
    normalize: bool = True,
    channel_multipliers: Optional[Sequence[float]] = None,
    channel_value_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    channel_cmaps: Optional[Sequence[str]] = None,
):
    data = tensor.detach().cpu().clone()
    if data.ndim == 4:
        data = data.squeeze(0)
    if data.ndim == 2:
        data = data.unsqueeze(0)

    os.makedirs(out_dir, exist_ok=True)
    for idx in range(data.shape[0]):
        channel = data[idx].numpy()
        if channel_multipliers is not None and idx < len(channel_multipliers):
            channel = channel * channel_multipliers[idx]

        channel_name = os.path.join(out_dir, f"{base_name}_channel{idx + 1}.pdf")
        current_cmap = cmap
        if channel_cmaps is not None and idx < len(channel_cmaps) and channel_cmaps[idx]:
            current_cmap = channel_cmaps[idx]

        if normalize:
            img = normalize_np(channel)
            plt.imsave(channel_name, img, cmap=current_cmap)
        else:
            kwargs = {}
            if channel_value_ranges is not None and idx < len(channel_value_ranges):
                vmin, vmax = channel_value_ranges[idx]
                kwargs['vmin'] = vmin
                kwargs['vmax'] = vmax
            plt.imsave(channel_name, channel, cmap=current_cmap, **kwargs)


def save_tensor_npy(tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, tensor.detach().cpu().numpy())


def save_aoa_radians(tensor, out_dir: str, base_name: str, aoa_channels: int):
    arr = tensor.detach().cpu().numpy().copy()
    if arr.ndim == 4:
        aoa = arr[:, :aoa_channels] * np.pi
    elif arr.ndim == 3:
        aoa = arr[:aoa_channels] * np.pi
        aoa = aoa[None, ...]
    else:
        return

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{base_name}_aoa_rad.npy"), aoa)


def plot_12channel_comparison(input_tensor, label_tensor, recon_tensor, save_path,
                              metadata=None, denormalize=True):
    """
    Plot input, label, and reconstruction side-by-side.

    The model uses a 12-channel internal representation (AoA, Amp, sin(phase),
    cos(phase) for 3 strongest paths) so that the circular phase variable is
    smooth and learnable. For visualization we recover the single phase angle
    per path via atan2(sin, cos) -- much more intuitive than two
    near-identical ring patterns. Final display has 9 columns:

        AoA1, AoA2, AoA3, Amp1, Amp2, Amp3, Phase1, Phase2, Phase3

    Args:
        input_tensor:  (12, H, W) tensor (or batched (1, 12, H, W))
        label_tensor:  (12, H, W) ground truth tensor
        recon_tensor:  (12, H, W) reconstructed tensor
        save_path:     where to save the comparison PDF
        metadata:      optional dict with 'bs_pos', 'buildings', etc.
        denormalize:   if True, undo the [-1, 1] normalization for display
    """
    # Convert tensors to numpy
    input_np = input_tensor.detach().cpu().numpy()
    label_np = label_tensor.detach().cpu().numpy()
    recon_np = recon_tensor.detach().cpu().numpy()

    # Handle batch dimension
    if input_np.ndim == 4:
        input_np = input_np[0]
    if label_np.ndim == 4:
        label_np = label_np[0]
    if recon_np.ndim == 4:
        recon_np = recon_np[0]

    # --- 2D phase unwrap helper -- converts the wrapped phase (rainbow rings)
    # into a smooth monotonic gradient (like the amplitude plot). Falls back to
    # wrapped phase if scikit-image isn't installed or unwrap fails (e.g. on
    # the masked Input row where the data is mostly noise).
    def _try_unwrap(p_wrapped_3xhw):
        try:
            from skimage.restoration import unwrap_phase
        except Exception:
            return p_wrapped_3xhw, False
        out = np.zeros_like(p_wrapped_3xhw)
        for k in range(p_wrapped_3xhw.shape[0]):
            try:
                out[k] = unwrap_phase(p_wrapped_3xhw[k])
            except Exception:
                out[k] = p_wrapped_3xhw[k]
        return out, True

    # --- Denormalize each channel group and recover phase angle from sin/cos ---
    def _split(arr, denorm, unwrap):
        aoa = arr[:3]
        amp = arr[3:6]
        sin_ph = arr[6:9]
        cos_ph = arr[9:12]
        if denorm:
            aoa = aoa * 180.0                          # [-1,1] -> [-180,180] deg
            amp = (amp + 1) * 25.0 - 90.0              # [-1,1] -> [-90,-40] dB
            # sin/cos are already in [-1,1]; no rescaling needed
        phase = np.arctan2(sin_ph, cos_ph)             # shape (3, H, W), [-pi, pi]
        if unwrap:
            phase, _ = _try_unwrap(phase)
        return aoa, amp, phase

    # Input is sparse + noisy after masking -- unwrap would give garbage.
    # Ground Truth + Reconstruction unwrap cleanly to smooth gradients.
    in_aoa, in_amp, in_phase = _split(input_np, denormalize, unwrap=False)
    gt_aoa, gt_amp, gt_phase = _split(label_np, denormalize, unwrap=True)
    rc_aoa, rc_amp, rc_phase = _split(recon_np, denormalize, unwrap=True)

    # 3 rows x 9 columns
    fig = plt.figure(figsize=(27, 9))
    gs = fig.add_gridspec(3, 9, hspace=0.15, wspace=0.30,
                          left=0.02, right=0.98, top=0.92, bottom=0.05)

    aoa_cmap = LinearSegmentedColormap.from_list(
        'aoa', ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'red'])

    bs_pos = metadata.get('bs_pos') if metadata else None
    buildings = metadata.get('buildings', []) if metadata else []
    grid_spacing = metadata.get('grid_spacing', 1.0) if metadata else 1.0

    row_titles = ['Input', 'Ground Truth', 'Reconstruction']
    col_titles = (['AoA 1', 'AoA 2', 'AoA 3']
                  + ['Amp 1', 'Amp 2', 'Amp 3']
                  + ['Phase 1', 'Phase 2', 'Phase 3'])

    # group_data[row_idx] = list of 9 2-D arrays in column order
    group_data = [
        list(in_aoa) + list(in_amp) + list(in_phase),
        list(gt_aoa) + list(gt_amp) + list(gt_phase),
        list(rc_aoa) + list(rc_amp) + list(rc_phase),
    ]
    # Phase colour range: use the GT unwrapped range so GT and Recon share it.
    gt_phase_min = float(np.min([gt_phase[k].min() for k in range(3)]))
    gt_phase_max = float(np.max([gt_phase[k].max() for k in range(3)]))
    # per-column plot parameters: (cmap, vmin, vmax, cbar_label)
    col_params = (
        [(aoa_cmap, -180, 180, 'Angle (deg)')] * 3
        + [('hot', -90, -40, 'Power (dB)')] * 3
        + [('viridis', gt_phase_min, gt_phase_max, 'Unwrapped phase (rad)')] * 3
    )

    def _overlay(ax):
        """Draw buildings + BS star on a given axes."""
        if not buildings:
            return
        for building in buildings:
            x, y = building['x'], building['y']
            w, h = building['width'], building['height']
            rect = Rectangle((x / grid_spacing, y / grid_spacing),
                              w / grid_spacing, h / grid_spacing,
                              linewidth=1.5, edgecolor='white',
                              facecolor='gray', alpha=0.3)
            ax.add_patch(rect)
        if bs_pos is not None:
            ax.plot(bs_pos[0] / grid_spacing, bs_pos[1] / grid_spacing,
                    'w*', markersize=10, markeredgecolor='black', markeredgewidth=0.8)

    for row_idx in range(3):
        for col_idx in range(9):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            d = group_data[row_idx][col_idx]
            cmap, vmin, vmax, cbar_label = col_params[col_idx]

            im = ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                           extent=[0, d.shape[1], 0, d.shape[0]])

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(row_titles[row_idx], fontsize=11, fontweight='bold')

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', length=0)

            if row_idx == 1:
                _overlay(ax)

            if row_idx == 2:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=6)
                # Label one column per group (middle column of each group)
                if col_idx in (1, 4, 7):
                    cbar.set_label(cbar_label, fontsize=8)

            ax.grid(False)

    title = 'AoA / Amplitude / Phase Reconstruction Comparison'
    if metadata and bs_pos is not None:
        title += f'  |  BS: ({bs_pos[0]:.0f}, {bs_pos[1]:.0f}), {len(buildings)} building(s)'
    fig.suptitle(title, fontsize=12, fontweight='bold')

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    print(f"Comparison plot saved to {save_path}")


def plot_12channel_single(tensor, save_path, title_prefix='Sample',
                          metadata=None, denormalize=True):
    """
    Plot a single 12-channel tensor (for input, label, or reconstruction alone).
    Layout: 4 rows (AoA, Amplitude, sin(phase), cos(phase)) x 3 columns (3 paths)

    Args:
        tensor: Tensor of shape (12, H, W) or (1, 12, H, W)
        save_path: Path to save the figure
        title_prefix: Prefix for the title (e.g., 'Input', 'Label', 'Reconstruction')
        metadata: Optional dict with 'bs_pos', 'buildings', etc.
        denormalize: Whether to denormalize from [-1, 1] range
    """
    data = tensor.detach().cpu().numpy()
    if data.ndim == 4:
        data = data[0]

    if denormalize:
        aoa_maps = data[:3] * 180.0
        amp_maps = (data[3:6] + 1) * 25.0 - 90.0
    else:
        aoa_maps = data[:3]
        amp_maps = data[3:6]
    sin_maps = data[6:9]
    cos_maps = data[9:12]

    fig, axes = plt.subplots(4, 3, figsize=(16, 20))

    aoa_cmap = LinearSegmentedColormap.from_list(
        'aoa', ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'red'])

    bs_pos = metadata.get('bs_pos') if metadata else None
    buildings = metadata.get('buildings', []) if metadata else []
    map_size = metadata.get('map_size') if metadata else None
    grid_spacing = metadata.get('grid_spacing', 1.0) if metadata else 1.0

    row_groups = [
        (aoa_maps,  aoa_cmap, -180, 180, 'Angle (deg)',   'AoA Path'),
        (amp_maps,  'hot',    -90,  -40, 'Power (dB)',    'Amp Path'),
        (sin_maps,  'coolwarm', -1,   1, 'sin(phase)',    'sin Path'),
        (cos_maps,  'coolwarm', -1,   1, 'cos(phase)',    'cos Path'),
    ]

    def _overlay(ax):
        if buildings:
            for building in buildings:
                x, y = building['x'], building['y']
                w, h = building['width'], building['height']
                rect = Rectangle((x / grid_spacing, y / grid_spacing),
                                  w / grid_spacing, h / grid_spacing,
                                  linewidth=2, edgecolor='white', facecolor='gray', alpha=0.3)
                ax.add_patch(rect)
        if bs_pos is not None:
            ax.plot(bs_pos[0] / grid_spacing, bs_pos[1] / grid_spacing,
                    'w*', markersize=15, markeredgecolor='black', markeredgewidth=1.5)

    for row_idx, (maps, cmap, vmin, vmax, cbar_lbl, title_base) in enumerate(row_groups):
        for i in range(3):
            ax = axes[row_idx, i]
            im = ax.imshow(maps[i], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                           extent=[0, maps[i].shape[1], 0, maps[i].shape[0]])
            ax.set_title(f'{title_base} {i+1}', fontsize=11, fontweight='bold')
            if i == 0:
                ax.set_ylabel(cbar_lbl, fontsize=10)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', length=0)

            _overlay(ax)

            plt.colorbar(im, ax=ax, label=cbar_lbl)
            stats_text = f'[{maps[i].min():.2f}, {maps[i].max():.2f}]'
            ax.text(0.02, 0.98, stats_text,
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)

    title = f'{title_prefix} (12 Channels)'
    if metadata and bs_pos is not None:
        title += f'\nBS: ({bs_pos[0]:.1f}, {bs_pos[1]:.1f}) | '
        title += f'{len(buildings)} Building(s)'
        if map_size is not None:
            title += f' | Map: {map_size[0]}x{map_size[1]}m'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Single-view plot saved to {save_path}")


def compute_nmse(reference: torch.Tensor, estimate: torch.Tensor):
    ref = reference.detach()
    est = estimate.detach()

    mse = torch.mean((est - ref) ** 2)
    denom = torch.mean(ref ** 2)
    total_nmse = (mse / (denom + 1e-15)).item()

    per_channel = []
    if ref.dim() == 4:
        for channel in range(ref.shape[1]):
            ref_c = ref[:, channel]
            est_c = est[:, channel]
            mse_c = torch.mean((est_c - ref_c) ** 2)
            denom_c = torch.mean(ref_c ** 2)
            per_channel.append((mse_c / (denom_c + 1e-15)).item())

    return total_nmse, per_channel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='configs/aoa_amp_building_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/aoa_amp_building_inpainting.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'train', 'all'])
    parser.add_argument('--mask_prob', type=float, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    logger = get_logger()
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    logger.info(f"Device set to {device_str}.")

    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    extra_keys = {
        'batch_size', 'learning_rate', 'num_epochs', 'save_interval',
        'epoch_save_interval', 'log_interval', 'dataset', 'dataloader',
        'model_path'
    }
    model_params = {k: v for k, v in model_config.items() if k not in extra_keys}

    data_channels = int(model_config.get('data_channels', 6))
    
    # Ensure data_channels is passed to create_model
    model_params['data_channels'] = data_channels

    model = create_model(**model_params)

    model = model.to(device)

    model_path = model_config.get('model_path')
    if model_path:
        logger.info(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)

    model.eval()

    measure_cfg = task_config['measurement']
    if args.mask_prob is not None and 'mask_opt' in measure_cfg:
        mp = float(args.mask_prob)
        mo = dict(measure_cfg['mask_opt'])
        mo['mask_prob_range'] = (mp, mp)
        measure_cfg['mask_opt'] = mo
    operator = get_operator(device=device, **measure_cfg['operator'])
    noiser = get_noise(**measure_cfg['noise'])
    logger.info(f"Operation: {measure_cfg['operator']['name']} / Noise: {measure_cfg['noise']['name']}")

    cond_cfg = task_config['conditioning']
    cond_method = get_conditioning_method(cond_cfg['method'], operator, noiser, **cond_cfg['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method: {cond_cfg['method']}")

    diffusion = create_sampler(**diffusion_config)
    sample_fn = partial(
        diffusion.p_sample_loop,
        model=model,
        measurement_cond_fn=measurement_cond_fn,
        record=False,
    )

    out_path = os.path.join(args.save_dir, measure_cfg['operator']['name'])
    for sub in ['input', 'recon', 'label']:
        os.makedirs(os.path.join(out_path, sub), exist_ok=True)

    data_cfg = task_config['data']
    data_kwargs = data_cfg.copy()
    dataset_name = data_kwargs.pop('name')
    root = data_kwargs.pop('root')
    data_kwargs.pop('num_samples', None)

    if dataset_name != 'aoa_amp_building':
        raise ValueError('sample_condition_building.py is intended for aoa_amp_building dataset only.')

    # Enable return_index to get actual dataset indices (important for metadata lookup)
    data_kwargs['return_index'] = True
    full_dataset = get_dataset(dataset_name, root, **data_kwargs)
    total_samples = len(full_dataset)
    logger.info(f"Full dataset size: {total_samples}")
    
    # Derive split from dataset config (adapts to both small test and full run)
    building_dist = data_cfg.get('building_distribution',
                                 data_kwargs.get('building_distribution', [20, 20, 20]))
    num_building_configs = sum(building_dist)
    configs_per_group = building_dist[0]
    test_configs_per_group = max(1, configs_per_group // 10)
    train_configs_per_group = configs_per_group - test_configs_per_group
    samples_per_config = total_samples // num_building_configs

    if args.split == 'test':
        indices_by_group = {1: [], 2: [], 3: []}
        for group in range(3):
            group_start_config = group * configs_per_group
            for config_offset in range(train_configs_per_group, configs_per_group):
                config_id = group_start_config + config_offset
                sample_start = config_id * samples_per_config
                sample_end = sample_start + samples_per_config
                indices_by_group[group + 1].extend(range(sample_start, sample_end))
        import random
        random.seed()
        for nb in indices_by_group:
            random.shuffle(indices_by_group[nb])
        interleaved = []
        max_len = max(len(v) for v in indices_by_group.values())
        for i in range(max_len):
            for nb in [1, 2, 3]:
                if i < len(indices_by_group[nb]):
                    interleaved.append(indices_by_group[nb][i])
        loader = DataLoader(Subset(full_dataset, interleaved), batch_size=1, shuffle=False, num_workers=0)
        logger.info(f"Using test split with {len(interleaved)} samples")
    elif args.split == 'train':
        indices = []
        for group in range(3):
            group_start_config = group * configs_per_group
            for config_offset in range(0, train_configs_per_group):
                config_id = group_start_config + config_offset
                sample_start = config_id * samples_per_config
                sample_end = sample_start + samples_per_config
                indices.extend(range(sample_start, sample_end))
        loader = DataLoader(Subset(full_dataset, indices), batch_size=1, shuffle=False, num_workers=0)
        logger.info(f"Using train split with {len(indices)} samples")
    else:
        loader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=0)
        logger.info(f"Using all samples: {total_samples}")

    if measure_cfg['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(**measure_cfg['mask_opt'])

    nmse_totals = []
    channel_nmse_records = defaultdict(list)
    nmse_per_sample = []
    max_samples = args.num_samples if args.num_samples is not None else None

    aoa_channels = 3  # first three channels are AoA
    # 12 channels: AoA(0-2), Amp(3-5), sin(6-8), cos(9-11)
    channel_ranges = ([(-np.pi, np.pi)] * 3
                      + [(-1.0, 1.0)] * 3
                      + [(-1.0, 1.0)] * 3
                      + [(-1.0, 1.0)] * 3)
    channel_scales = [np.pi] * 3 + [1.0] * 9
    channel_cmaps = ['hsv'] * 3 + ['hot'] * 3 + ['coolwarm'] * 3 + ['coolwarm'] * 3

    for enum_idx, (ref_img, dataset_idx) in enumerate(loader):
        # dataset_idx is the original index in the full dataset (from return_index=True)
        # test_indices[enum_idx] gives us the same original index
        original_idx = dataset_idx.item()
        logger.info(f"Inference for test sample {enum_idx} (original dataset index: {original_idx})")
        fname_base = str(enum_idx).zfill(5)
        ref_img = ref_img.to(device)
        
        # Fetch metadata using the original dataset index
        metadata = None
        # Access the underlying full dataset (Subset wraps the original dataset)
        underlying_dataset = loader.dataset.dataset if hasattr(loader.dataset, 'dataset') else loader.dataset
        if hasattr(underlying_dataset, 'get_metadata'):
            try:
                metadata = underlying_dataset.get_metadata(original_idx)
                logger.info(f"Loaded metadata: BS @ {metadata.get('bs_pos')}, {len(metadata.get('buildings', []))} building(s)")
            except Exception as e:
                logger.warning(f"Could not load metadata for sample {enum_idx}: {e}")

        if measure_cfg['operator']['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        else:
            y = operator.forward(ref_img)
            y_n = noiser(y)

        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

        total_nmse, per_channel_nmse = compute_nmse(ref_img, sample)
        nmse_totals.append(total_nmse)
        for c_idx, value in enumerate(per_channel_nmse):
            channel_nmse_records[c_idx].append(value)
        nmse_per_sample.append((enum_idx, total_nmse, per_channel_nmse))

        ch_labels = (['AoA1', 'AoA2', 'AoA3']
                     + ['Amp1', 'Amp2', 'Amp3']
                     + ['sin1', 'sin2', 'sin3']
                     + ['cos1', 'cos2', 'cos3'])
        nmse_msg = ", ".join(
            f"{ch_labels[c_idx]}: {value:.4e}" for c_idx, value in enumerate(per_channel_nmse)
        ) if per_channel_nmse else ""
        logger.info(f"NMSE (sample {enum_idx}): total {total_nmse:.4e} {nmse_msg}")

        # Save comprehensive comparison plot with metadata in comparison folder
        comparison_path = os.path.join(out_path, 'comparison', f'{fname_base}_comparison.pdf')
        plot_12channel_comparison(y_n, ref_img, sample, comparison_path,
                                 metadata=metadata, denormalize=True)
        
        # Save combined 6-channel plots in their respective directories
        # plot_6channel_single(y_n, os.path.join(out_path, 'input', f'{fname_base}_combined.pdf'),
        #                     title_prefix='Input (Degraded)', metadata=metadata, denormalize=True)
        # plot_6channel_single(ref_img, os.path.join(out_path, 'label', f'{fname_base}_combined.pdf'),
        #                     title_prefix='Ground Truth', metadata=metadata, denormalize=True)
        # plot_6channel_single(sample, os.path.join(out_path, 'recon', f'{fname_base}_combined.pdf'),
        #                     title_prefix='Reconstruction', metadata=metadata, denormalize=True)

        # # Legacy channel-by-channel saves (kept for compatibility)
        # save_tensor_channels(
        #     y_n,
        #     os.path.join(out_path, 'input'),
        #     fname_base,
        #     cmap='viridis',
        #     normalize=False,
        #     channel_multipliers=channel_scales,
        #     channel_value_ranges=channel_ranges,
        #     channel_cmaps=channel_cmaps,
        # )
        # save_tensor_channels(
        #     ref_img,
        #     os.path.join(out_path, 'label'),
        #     fname_base,
        #     cmap='viridis',
        #     normalize=False,
        #     channel_multipliers=channel_scales,
        #     channel_value_ranges=channel_ranges,
        #     channel_cmaps=channel_cmaps,
        # )
        # save_tensor_channels(
        #     sample,
        #     os.path.join(out_path, 'recon'),
        #     fname_base,
        #     cmap='viridis',
        #     normalize=False,
        #     channel_multipliers=channel_scales,
        #     channel_value_ranges=channel_ranges,
        #     channel_cmaps=channel_cmaps,
        # )

        # save_tensor_npy(y_n, os.path.join(out_path, 'input', f'{fname_base}.npy'))
        # save_tensor_npy(ref_img, os.path.join(out_path, 'label', f'{fname_base}.npy'))
        # save_tensor_npy(sample, os.path.join(out_path, 'recon', f'{fname_base}.npy'))

        # save_aoa_radians(y_n, os.path.join(out_path, 'input'), fname_base, aoa_channels)
        # save_aoa_radians(ref_img, os.path.join(out_path, 'label'), fname_base, aoa_channels)
        # save_aoa_radians(sample, os.path.join(out_path, 'recon'), fname_base, aoa_channels)

        # Limit number of processed samples if requested
        if max_samples is not None and (enum_idx + 1) >= max_samples:
            logger.info(f"Reached requested num_samples = {max_samples}, stopping.")
            break

    if nmse_totals:
        avg_total_nmse = sum(nmse_totals) / len(nmse_totals)
        logger.info(f"Average NMSE over {len(nmse_totals)} samples: {avg_total_nmse:.4e}")
        avg_channels = []
        for c_idx, values in channel_nmse_records.items():
            avg_channel_nmse = sum(values) / len(values)
            avg_channels.append(avg_channel_nmse)
            logger.info(f"Average NMSE channel {c_idx + 1}: {avg_channel_nmse:.4e}")

        # Persist NMSE metrics for downstream aggregation
        try:
            import csv, json
            metrics_dir = os.path.join(out_path, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)

            # Derive mask_prob for filename
            mask_prob_val = None
            mpr = measure_cfg.get('mask_opt', {}).get('mask_prob_range')
            try:
                if isinstance(mpr, (int, float)):
                    mask_prob_val = float(mpr)
                elif isinstance(mpr, (list, tuple)) and len(mpr) >= 1:
                    mask_prob_val = float(mpr[0])
            except Exception:
                mask_prob_val = None

            suffix = f"_mask_{mask_prob_val:.2f}" if mask_prob_val is not None else ""

            # Per-sample CSV
            csv_path = os.path.join(metrics_dir, f"nmse_samples{suffix}.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['sample_idx', 'total_nmse'] + [f'ch{i+1}_nmse' for i in range(data_channels)]
                writer.writerow(header)
                for idx, total, ch_list in nmse_per_sample:
                    row = [idx, total] + list(ch_list)
                    writer.writerow(row)

            # Summary JSON
            summary_path = os.path.join(metrics_dir, f"nmse_summary{suffix}.json")
            with open(summary_path, 'w') as f:
                json.dump({
                    'samples': len(nmse_totals),
                    'avg_total_nmse': avg_total_nmse,
                    'avg_channel_nmse': avg_channels,
                    'mask_prob': mask_prob_val
                }, f)
            logger.info(f"Saved NMSE metrics to {csv_path} and {summary_path}")
        except Exception as e:
            logger.warning(f"Failed to save NMSE metrics: {e}")


if __name__ == '__main__':
    main()
