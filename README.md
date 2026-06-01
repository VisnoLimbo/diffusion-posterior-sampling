# Diffusion-Based Wireless Channel Map Reconstruction with Reflection Phase Modeling


**Can we rebuild a complete wireless "radio map" of a city block from just 10% of the measurements — including the hard-to-capture signal *phase*?** This project does exactly that: it (1) generates a physically accurate synthetic dataset of radio maps and (2) trains a diffusion model to reconstruct the full map from sparse, scattered samples.

## The idea
- A **ray tracer** simulates how a 2.4 GHz signal travels from a base station to every point on a 128×128 grid via three paths: **line-of-sight, reflection off buildings, and diffraction around corners**.
- For each path it stores **angle of arrival, amplitude, and phase** — with phase from real physics (**Fresnel** reflection coefficients for concrete via ITU-R P.2040-4, **UTD** for 90° corners), not guessed.
- Phase is encoded as **(sin, cos)** to avoid the ±π "wrap-around" problem, producing a **12-channel** radio map.
- A **diffusion model (DDPM, ~21M-parameter U-Net)** learns what realistic radio maps look like; at inference, **Diffusion Posterior Sampling (DPS)** fills in a full map from only 10% of observed pixels.

## Why it matters
Earlier work reconstructed only angle + amplitude (6 channels) and treated phase as a constant. This project adds **physically grounded phase** as a first-class quantity — and shows the learned model recovers phase exactly where classical interpolation fails.

## Key results
*(epoch 100, 16 unseen test scenes, 90% of pixels missing)*
- Overall reconstruction error **NMSE = 0.162** across all 12 channels.
- Direct-path angle and phase recovered almost perfectly (NMSE 0.002–0.012).
- **Up to ~400× more accurate than nearest-neighbour interpolation on the phase channels**, where interpolation is no better than a constant guess.
- Reconstructed (sin, cos) pairs stay within ~5% of the unit circle — the model learned the circular nature of phase on its own.

## How to run
Python 3.9–3.11, GPU recommended.
```bash
pip install -r requirements.txt

# 1) generate the dataset (54,000 samples, deterministic seed 42)
python train_aoa_amp_building.py --model_config configs/aoa_amp_building_config.yaml --generate_data

# 2) train the diffusion model (100 epochs)
python train_aoa_amp_building.py --model_config configs/aoa_amp_building_config.yaml

# 3) reconstruct + evaluate under 90% masking (with classical baseline)
python sample_condition_building.py --model_config configs/aoa_amp_building_config.yaml \
    --task_config configs/aoa_amp_building_inpainting.yaml --mask_prob 0.9 --baseline
```
Or open `colab_train_building.ipynb` in Google Colab (GPU runtime) and run top to bottom.

## What's in the repo
| File / folder | Role |
|---|---|
| `aoa_amp_building.py`, `aoa_amp_building_gpu.py` | ray tracer (LOS / reflection / diffraction + Fresnel & UTD phase) |
| `aoa_amp_building_data_gpu.py`, `data/aoa_amp_building_dataset.py` | dataset generation + streaming HDF5 storage |
| `train_aoa_amp_building.py` | DDPM training |
| `sample_condition_building.py` | DPS reconstruction + interpolation baseline + NMSE |
| `guided_diffusion/` | U-Net (~21M params) and the diffusion process |
| `configs/` | model, diffusion, and inpainting settings |
| `colab_train_building.ipynb` | end-to-end notebook used for the results |

## Data & model weights
The 54,000-sample dataset (~22 GB HDF5) and trained checkpoints are not stored here. The dataset **regenerates exactly** from the command above (fixed seed 42); checkpoints are available on request.

## Built on
This work extends **Diffusion Posterior Sampling (DPS)** (Chung et al., ICLR 2023) and OpenAI's **guided-diffusion**.

## Citation
> V. K. Limbu, *Dataset Generation for Diffusion-Based Wireless Channel Map Reconstruction Including Reflection Phase Modeling*, MSc thesis, Aarhus University, 2026.

## License
MIT © 2026 Vishnu Kumar Limbu. Components derived from DPS and guided-diffusion remain under their upstream licenses.
