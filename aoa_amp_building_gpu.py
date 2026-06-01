"""
GPU-accelerated RayTracingAoAMap class using PyTorch for CUDA acceleration.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import math
from typing import List, Tuple, Dict, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', module='numba')

# --- Material & UTD constants for phase computation (ITU-R P.2040-4) ---
_FREQ_HZ = 2.4e9
_WAVELENGTH = 3e8 / _FREQ_HZ
_WAVENUMBER = 2.0 * np.pi / _WAVELENGTH
_CONCRETE_EPS = complex(5.24, -17.98 * 0.0462 * (2.4 ** 0.7822) / 2.4)
_PHASE_COARSEN = 64    # lambda_eff = 64*lambda = 8 m  ->  ~16 cycles / 128 m map
_WEDGE_N = 1.5
_CORNER_WEDGE = [
    (0.0,           False),
    (np.pi,         True),
    (3*np.pi/2,     True),
    (0.0,           True),
]

# Check CUDA availability
def get_best_device():
    """Get the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Update the initialization
MPS_AVAILABLE = torch.backends.mps.is_available()
CUDA_AVAILABLE = torch.cuda.is_available()

if MPS_AVAILABLE:
    print(f"MPS (Apple Silicon GPU) available")
elif CUDA_AVAILABLE:
    print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
else:
    print("No GPU acceleration available, using CPU")


def calculate_aoa_gpu(ue_positions: torch.Tensor, bs_pos: torch.Tensor) -> torch.Tensor:
    """Calculate AoA for all UE positions using GPU"""
    vec = ue_positions - bs_pos.unsqueeze(0).unsqueeze(0)  # Broadcasting
    aoa_rad = torch.atan2(vec[..., 1], vec[..., 0])
    aoa_deg = aoa_rad * 180.0 / math.pi
    return aoa_deg


def calculate_distance_gpu(ue_positions: torch.Tensor, bs_pos: torch.Tensor) -> torch.Tensor:
    """Calculate distances for all UE positions using GPU"""
    vec = ue_positions - bs_pos.unsqueeze(0).unsqueeze(0)  # Broadcasting
    distances = torch.norm(vec, dim=-1)
    return distances


def calculate_path_loss_gpu(distances: torch.Tensor, los_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Calculate path loss using GPU"""
    frequency = 2.4e9  # 2.4 GHz
    wavelength = 3e8 / frequency
    
    # Avoid division by zero
    distances_safe = torch.clamp(distances, min=1e-6)
    
    # Calculate free space path loss
    pl = 20 * torch.log10(4 * math.pi * distances_safe / wavelength)
    
    # Add penetration loss for NLOS
    nlos_penalty = torch.tensor(15.0, device=device)
    pl = torch.where(los_mask, pl, pl + nlos_penalty)
    
    return pl


def line_segment_intersection_gpu(p1: torch.Tensor, p2: torch.Tensor, 
                                p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
    """Check line segment intersection using GPU vectorized operations"""
    # Vectorized line intersection check
    # p1, p2: line 1 endpoints, p3, p4: line 2 endpoints
    
    d = p2 - p1  # Direction vector of line 1
    e = p4 - p3  # Direction vector of line 2
    
    # Calculate denominator
    denom = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
    
    # Check for parallel lines
    parallel_mask = torch.abs(denom) < 1e-10
    
    # Calculate parameters
    f = p3 - p1
    t = (f[..., 0] * e[..., 1] - f[..., 1] * e[..., 0]) / torch.clamp(denom, min=1e-10)
    u = (f[..., 0] * d[..., 1] - f[..., 1] * d[..., 0]) / torch.clamp(denom, min=1e-10)
    
    # Check if intersection is within both line segments
    intersection = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1) & ~parallel_mask
    
    return intersection


def check_los_gpu(ue_positions: torch.Tensor, bs_pos: torch.Tensor, 
                 building_edges: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Check line-of-sight using GPU vectorized operations"""
    if building_edges.numel() == 0:
        return torch.ones(ue_positions.shape[:-1], device=device, dtype=torch.bool)
    
    n_y, n_x = ue_positions.shape[0], ue_positions.shape[1]
    los_map = torch.ones((n_y, n_x), device=device, dtype=torch.bool)
    
    # Process in batches to manage memory
    batch_size = 1000
    total_positions = n_y * n_x
    
    # Flatten UE positions for easier processing
    ue_flat = ue_positions.reshape(-1, 2)
    los_flat = torch.ones(total_positions, device=device, dtype=torch.bool)
    
    for i in range(0, total_positions, batch_size):
        end_idx = min(i + batch_size, total_positions)
        ue_batch = ue_flat[i:end_idx]  # Shape: (batch_size, 2)
        
        # Check intersection with all building edges
        for edge_idx in range(building_edges.shape[0]):
            edge = building_edges[edge_idx]  # Shape: (2, 2) - two endpoints
            p3, p4 = edge[0], edge[1]
            
            # Broadcast for vectorized intersection check
            bs_batch = bs_pos.unsqueeze(0).expand(ue_batch.shape[0], -1)  # (batch_size, 2)
            p3_batch = p3.unsqueeze(0).expand(ue_batch.shape[0], -1)      # (batch_size, 2)
            p4_batch = p4.unsqueeze(0).expand(ue_batch.shape[0], -1)      # (batch_size, 2)
            
            # Check intersection
            intersects = line_segment_intersection_gpu(bs_batch, ue_batch, p3_batch, p4_batch)
            
            # Update LOS status
            los_flat[i:end_idx] = los_flat[i:end_idx] & ~intersects
    
    return los_flat.reshape(n_y, n_x)


class RayTracingAoAMapGPU:
    def __init__(self, map_size, grid_spacing=1, device=None, verbose=False):
        """
        Initialize GPU-accelerated ray tracing for AoA map generation.
        
        Args:
            map_size: Size of the map (x_max, y_max) or single integer for square map
            grid_spacing: Grid spacing for UE positions
            device: Device to use ('cuda', 'cpu', or None for auto)
            verbose: Whether to print initialization details
        """
        # Set device
        if device is None:
            self.device = get_best_device()
        elif device == 'auto':
            self.device = get_best_device()
        else:
            self.device = torch.device(device)
        
        if verbose:
            print(f"RayTracing using device: {self.device}")
        
        # Store verbose setting for use in other methods
        self.verbose = verbose
        
        # Handle both tuple and single integer map_size
        if isinstance(map_size, (tuple, list)):
            map_x, map_y = map_size[0], map_size[1]
        else:
            map_x, map_y = map_size, map_size
        
        # Create grid points
        self.x_grid = torch.arange(0, map_x, grid_spacing, dtype=torch.float32, device=self.device)
        self.y_grid = torch.arange(0, map_y, grid_spacing, dtype=torch.float32, device=self.device)
        
        # Create meshgrid
        Y, X = torch.meshgrid(self.y_grid, self.x_grid, indexing='ij')
        self.X = X
        self.Y = Y
        
        # Pre-compute UE positions for vectorized operations
        self.ue_positions = torch.stack([self.X, self.Y], dim=-1)  # Shape: (n_y, n_x, 2)
        
        # Store grid dimensions for batch processing
        self.num_y, self.num_x = self.X.shape
        
        # Initialize structures
        self.bs_pos = None
        self.buildings = []
        self.building_edges = None
        self._ranked_maps_cache = None
        
        # Flag for batch processing optimization
        self.building_edges_computed = False
        
        if self.verbose:
            print(f"Grid shape: {self.ue_positions.shape[:-1]}")
            print(f"Total UE positions: {self.ue_positions.shape[0] * self.ue_positions.shape[1]}")
        
    def set_base_station(self, x, y):
        """Set base station position"""
        self.bs_pos = torch.tensor([x, y], dtype=torch.float32, device=self.device)
        self._ranked_maps_cache = None
        
    def add_building(self, x, y, width, height):
        """
        Add a rectangular building
        
        Parameters:
        -----------
        x, y : float
            Bottom-left corner coordinates
        width, height : float
            Building dimensions
        """
        building = {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'corners': [
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height]
            ]
        }
        self.buildings.append(building)
        self._ranked_maps_cache = None
        
        # Update building edges for vectorized LOS calculation
        self._update_building_edges()
    
    def _update_building_edges(self):
        """Update building edges tensor for GPU operations"""
        all_edges = []
        
        for building in self.buildings:
            x_min, y_min = building['x'], building['y']
            x_max = x_min + building['width']
            y_max = y_min + building['height']
            
            # Rectangle edges
            edges = [
                [[x_min, y_min], [x_max, y_min]],  # bottom
                [[x_max, y_min], [x_max, y_max]],  # right
                [[x_max, y_max], [x_min, y_max]],  # top
                [[x_min, y_max], [x_min, y_min]]   # left
            ]
            
            for edge in edges:
                all_edges.append(edge)
        
        if all_edges:
            self.building_edges = torch.tensor(all_edges, dtype=torch.float32, device=self.device)
        else:
            self.building_edges = torch.empty((0, 2, 2), dtype=torch.float32, device=self.device)
        
        # Mark building edges as computed for batch processing optimization
        self.building_edges_computed = True

    def _line_segments_intersect_np(self, p1, p2, p3, p4):
        """Check if two line segments intersect (NumPy scalar version)."""
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        p3 = np.asarray(p3, dtype=np.float64)
        p4 = np.asarray(p4, dtype=np.float64)

        d = p2 - p1
        e = p4 - p3
        denom = d[0] * e[1] - d[1] * e[0]
        if abs(denom) < 1e-10:
            return False

        t = ((p3[0] - p1[0]) * e[1] - (p3[1] - p1[1]) * e[0]) / denom
        u = ((p3[0] - p1[0]) * d[1] - (p3[1] - p1[1]) * d[0]) / denom
        return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0

    def _line_intersects_rectangle_np(self, p1, p2, rect):
        """Check if a line segment intersects a rectangle."""
        x_min, y_min = rect['x'], rect['y']
        x_max = x_min + rect['width']
        y_max = y_min + rect['height']

        edges = [
            (np.array([x_min, y_min]), np.array([x_max, y_min])),
            (np.array([x_max, y_min]), np.array([x_max, y_max])),
            (np.array([x_max, y_max]), np.array([x_min, y_max])),
            (np.array([x_min, y_max]), np.array([x_min, y_min])),
        ]

        for edge_start, edge_end in edges:
            if self._line_segments_intersect_np(p1, p2, edge_start, edge_end):
                return True
        return False

    def _is_los_np(self, ue_pos, bs_pos):
        """Check LoS between BS and UE using rectangle intersections."""
        for building in self.buildings:
            if self._line_intersects_rectangle_np(bs_pos, ue_pos, building):
                return False
        return True

    def _point_to_segment_distance_np(self, point, seg_start, seg_end):
        """Distance from point to segment using projection."""
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq < 1e-12:
            return np.linalg.norm(point - seg_start)

        t = np.dot(point - seg_start, seg_vec) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        projection = seg_start + t * seg_vec
        return np.linalg.norm(point - projection)

    def _get_building_walls_np(self, building):
        """Return rectangle wall segments as (name, start, end)."""
        x_min, y_min = building['x'], building['y']
        x_max = x_min + building['width']
        y_max = y_min + building['height']

        return [
            ('bottom', np.array([x_min, y_min]), np.array([x_max, y_min])),
            ('right', np.array([x_max, y_min]), np.array([x_max, y_max])),
            ('top', np.array([x_max, y_max]), np.array([x_min, y_max])),
            ('left', np.array([x_min, y_max]), np.array([x_min, y_min])),
        ]

    def _get_two_closest_walls_np(self, ue_pos, building):
        """Pick the two wall facets closest to UE for reflection search."""
        walls = self._get_building_walls_np(building)
        distances = []
        for idx, (_, wall_start, wall_end) in enumerate(walls):
            dist = self._point_to_segment_distance_np(ue_pos, wall_start, wall_end)
            distances.append((dist, idx))

        distances.sort(key=lambda item: item[0])
        return [walls[idx] for _, idx in distances[:2]]

    def _reflect_point_across_wall_np(self, point, wall_start, wall_end):
        """Reflect a point across an axis-aligned wall segment."""
        if abs(wall_start[0] - wall_end[0]) < 1e-8:  # vertical wall x = c
            wall_x = wall_start[0]
            return np.array([2.0 * wall_x - point[0], point[1]])

        if abs(wall_start[1] - wall_end[1]) < 1e-8:  # horizontal wall y = c
            wall_y = wall_start[1]
            return np.array([point[0], 2.0 * wall_y - point[1]])

        return None

    def _line_wall_intersection_np(self, line_start, line_end, wall_start, wall_end):
        """Return line-wall intersection if it lies on both finite segments."""
        d = line_end - line_start
        e = wall_end - wall_start
        denom = d[0] * e[1] - d[1] * e[0]

        if abs(denom) < 1e-10:
            return None

        delta = wall_start - line_start
        t = (delta[0] * e[1] - delta[1] * e[0]) / denom
        u = (delta[0] * d[1] - delta[1] * d[0]) / denom

        if not (0.0 < t < 1.0 and 0.0 <= u <= 1.0):
            return None

        return line_start + t * d

    def _reflection_candidates_for_ue_np(self, ue_pos, bs_pos):
        """Generate image-theory reflection candidates from all buildings."""
        candidates = []

        for building in self.buildings:
            two_closest_walls = self._get_two_closest_walls_np(ue_pos, building)
            for _, wall_start, wall_end in two_closest_walls:
                ue_image = self._reflect_point_across_wall_np(ue_pos, wall_start, wall_end)
                if ue_image is None:
                    continue

                reflection_point = self._line_wall_intersection_np(
                    bs_pos, ue_image, wall_start, wall_end
                )
                if reflection_point is None:
                    continue

                dist_bs_to_refl = np.linalg.norm(reflection_point - bs_pos)
                dist_refl_to_ue = np.linalg.norm(ue_pos - reflection_point)
                total_dist = dist_bs_to_refl + dist_refl_to_ue
                if total_dist < 1e-6:
                    continue

                path_loss = self.calculate_path_loss_single(total_dist, True) + 6.0
                amplitude_db = -path_loss
                vec = reflection_point - bs_pos
                aoa = np.degrees(np.arctan2(vec[1], vec[0]))
                incoming_vec = reflection_point - bs_pos
                cos_theta = self._wall_incidence_cos_np(incoming_vec, wall_start, wall_end)
                refl_extra = self._fresnel_te_phase_np(cos_theta)
                phase = self.calculate_phase_single(total_dist, extra_phase=refl_extra)

                candidates.append({
                    'aoa': aoa,
                    'amplitude_db': amplitude_db,
                    'phase': phase,
                    'distance': total_dist,
                })

        candidates.sort(key=lambda item: item['amplitude_db'], reverse=True)
        return candidates

    def _diffraction_candidates_for_ue_np(self, ue_pos, bs_pos):
        """Generate diffraction candidates from all building corners."""
        candidates = []
        for building in self.buildings:
            for corner_idx, corner_raw in enumerate(building['corners']):
                corner = np.array(corner_raw, dtype=np.float64)
                dist_bs_to_corner = np.linalg.norm(corner - bs_pos)
                dist_corner_to_ue = np.linalg.norm(ue_pos - corner)
                total_dist = dist_bs_to_corner + dist_corner_to_ue

                path_loss = self.calculate_path_loss_single(total_dist, True) + 30.0
                amplitude_db = -path_loss
                vec = corner - bs_pos
                aoa = np.degrees(np.arctan2(vec[1], vec[0]))
                diff_extra = self._utd_diffraction_phase_te_np(
                    corner, corner_idx, bs_pos, ue_pos)
                phase = self.calculate_phase_single(total_dist, extra_phase=diff_extra)

                candidates.append({
                    'aoa': aoa,
                    'amplitude_db': amplitude_db,
                    'phase': phase,
                    'distance': total_dist,
                })

        candidates.sort(key=lambda item: item['distance'])
        return candidates

    def _build_ranked_paths_for_ue_np(self, ue_pos, bs_pos, los, num_paths):
        """Build ranked paths: direct, reflections, then diffraction fallback."""
        direct_dist = np.linalg.norm(ue_pos - bs_pos)
        direct_dist = max(direct_dist, 1e-6)
        direct_amp = -self.calculate_path_loss_single(direct_dist, los)
        direct_vec = ue_pos - bs_pos
        direct_aoa = np.degrees(np.arctan2(direct_vec[1], direct_vec[0]))
        direct_phase = self.calculate_phase_single(direct_dist, extra_phase=0.0)

        ranked_paths = [
            {
                'aoa': direct_aoa,
                'amplitude_db': direct_amp,
                'phase': direct_phase,
                'distance': direct_dist,
            }
        ]

        for candidate in self._reflection_candidates_for_ue_np(ue_pos, bs_pos):
            if len(ranked_paths) >= num_paths:
                break
            ranked_paths.append(candidate)

        if len(ranked_paths) < num_paths:
            for candidate in self._diffraction_candidates_for_ue_np(ue_pos, bs_pos):
                if len(ranked_paths) >= num_paths:
                    break
                ranked_paths.append(candidate)

        while len(ranked_paths) < num_paths:
            ranked_paths.append(
                {
                    'aoa': direct_aoa,
                    'amplitude_db': -120.0,
                    'phase': 0.0,
                    'distance': np.inf,
                }
            )

        return ranked_paths

    def _generate_ranked_path_maps_gpu(self, num_paths=3):
        """Generate AoA/amplitude/phase maps with one shared path assignment.

        Thin wrapper: returns a cached result when the BS position and
        num_paths are unchanged, otherwise dispatches to the fully vectorized
        implementation. The cache key now includes the BS position, so a
        stale cache can never leak across base-station positions.
        """
        if self.bs_pos is None:
            raise ValueError("Base station position not set")

        cache = self._ranked_maps_cache
        if (
            cache is not None
            and cache['num_paths'] == num_paths
            and cache['bs_pos'].shape == self.bs_pos.shape
            and torch.equal(cache['bs_pos'], self.bs_pos)
        ):
            # Restore the excess-phase side channel from cache too.
            self._last_extra_maps = [m.copy() for m in cache.get('extra_maps', [])]
            return (
                [m.copy() for m in cache['aoa_maps']],
                [m.copy() for m in cache['amplitude_maps']],
                [m.copy() for m in cache['phase_maps']],
                cache['los_map'].copy(),
            )

        aoa_maps, amplitude_maps, phase_maps, los_map = \
            self._generate_ranked_path_maps_vectorized(num_paths=num_paths)
        # _generate_ranked_path_maps_vectorized populates self._last_extra_maps
        # as a side effect.
        extra_maps = getattr(self, '_last_extra_maps', None) or []

        self._ranked_maps_cache = {
            'num_paths': num_paths,
            'bs_pos': self.bs_pos.detach().clone(),
            'aoa_maps': [m.copy() for m in aoa_maps],
            'amplitude_maps': [m.copy() for m in amplitude_maps],
            'phase_maps': [m.copy() for m in phase_maps],
            'los_map': los_map.copy(),
            'extra_maps': [m.copy() for m in extra_maps],
        }

        return aoa_maps, amplitude_maps, phase_maps, los_map

    def generate_excess_phase_map_gpu(self, num_paths=3):
        """Return per-path excess (interaction-only) phase maps.

        The total propagation phase has two components:

            phase_total = -2*pi*d / (N*lambda)   +   interaction_phase

        The first term is just smooth radial propagation -- the source of the
        N=64-cycle "rainbow rings" visualisation. The second term is the
        physically interesting "phase shift" the thesis is about: the Fresnel
        TE coefficient angle for each reflected path, and the UTD coefficient
        angle for each diffracted path. The direct path's interaction phase
        is exactly 0.

        Visualised with cmap='hsv' over [-pi, pi], the strongest path becomes
        a uniform colour (interaction = 0) and reflection/diffraction paths
        decompose into discrete regions whose colour identifies the wall or
        corner that dominates at each pixel.
        """
        # Triggers compute + cache; populates self._last_extra_maps.
        self._generate_ranked_path_maps_gpu(num_paths=num_paths)
        return [m.copy() for m in self._last_extra_maps]
    
    def _generate_ranked_path_maps_vectorized(self, num_paths=3):
        """Fully vectorized ranked-path map computation (production fast path).

        Computes AoA / amplitude / phase maps for ``num_paths`` ranked paths
        over the ENTIRE UE grid using batched tensor ops on ``self.device``.
        This replaces the original per-UE Python double loop and is typically
        100-1000x faster while reproducing the same physics -- see
        ``_generate_ranked_path_maps_scalar`` and ``validate_vectorized.py``.

        Path assignment (identical to the scalar reference):
          path 0        -> direct path
          paths 1..N-1  -> strongest reflections (image theory, 2 closest
                           walls per building), then diffraction (UTD, by
                           shortest detour) as fallback, then null paths.
        """
        if self.bs_pos is None:
            raise ValueError("Base station position not set")

        device = self.device
        n_y, n_x = self.X.shape
        P = n_y * n_x
        PI = math.pi
        DEG = 180.0 / PI
        wavelength = float(_WAVELENGTH)
        NEG_INF = float('-inf')

        ue = self.ue_positions.reshape(-1, 2).to(torch.float32)
        ue_x, ue_y = ue[:, 0], ue[:, 1]
        bs = self.bs_pos.to(torch.float32).reshape(2)
        bs_x = float(bs[0]); bs_y = float(bs[1])

        # --- line-of-sight (already a GPU kernel) ---
        if self.building_edges is None or self.building_edges.numel() == 0:
            los_bool = torch.ones((n_y, n_x), device=device, dtype=torch.bool)
        else:
            los_bool = check_los_gpu(self.ue_positions, self.bs_pos,
                                     self.building_edges, device)
        los_flat = los_bool.reshape(-1)

        zero = torch.zeros((), device=device)
        nlos_pen = torch.full((), 15.0, device=device)

        def _path_loss(dist, los_mask):
            d = torch.clamp(dist, min=1e-6)
            pl = 20.0 * torch.log10(4.0 * PI * d / wavelength)
            if los_mask is not None:
                pl = pl + torch.where(los_mask, zero, nlos_pen)
            return pl

        def _coarse_phase(dist, extra):
            cp = -2.0 * PI * dist / (_PHASE_COARSEN * wavelength)
            return torch.remainder(cp + extra + PI, 2.0 * PI) - PI

        # ============ direct path ============
        dx = ue_x - bs_x
        dy = ue_y - bs_y
        direct_dist = torch.clamp(torch.sqrt(dx * dx + dy * dy), min=1e-6)
        direct_aoa = torch.atan2(dy, dx) * DEG
        direct_amp = -_path_loss(direct_dist, los_flat)
        direct_phase = _coarse_phase(direct_dist, torch.zeros_like(direct_dist))

        # ============ reflection & diffraction candidates ============
        eps_c = torch.tensor(_CONCRETE_EPS, dtype=torch.complex64, device=device)
        refl_amp_list, refl_aoa_list, refl_phase_list, refl_extra_list = [], [], [], []
        diff_amp_list, diff_aoa_list, diff_phase_list, diff_dist_list, diff_extra_list = [], [], [], [], []

        def _cot(x):
            s = torch.sin(x)
            safe = torch.abs(s) > 1e-6
            s_safe = torch.where(safe, s, torch.ones_like(s))
            return torch.where(safe, torch.cos(x) / s_safe, torch.zeros_like(s))

        for building in self.buildings:
            x_min = float(building['x']); y_min = float(building['y'])
            x_max = x_min + float(building['width'])
            y_max = y_min + float(building['height'])

            # walls: (start_x, start_y, end_x, end_y, is_vertical, const_coord)
            walls = [
                (x_min, y_min, x_max, y_min, False, y_min),  # bottom
                (x_max, y_min, x_max, y_max, True,  x_max),  # right
                (x_max, y_max, x_min, y_max, False, y_max),  # top
                (x_min, y_max, x_min, y_min, True,  x_min),  # left
            ]

            # point-to-segment distance UE->wall, for the "2 closest walls" rule
            wall_dists = []
            for (wsx, wsy, wex, wey, is_vert, c) in walls:
                seg_x, seg_y = wex - wsx, wey - wsy
                seg_len_sq = seg_x * seg_x + seg_y * seg_y
                t = ((ue_x - wsx) * seg_x + (ue_y - wsy) * seg_y) / seg_len_sq
                t = torch.clamp(t, 0.0, 1.0)
                proj_x = wsx + t * seg_x
                proj_y = wsy + t * seg_y
                wall_dists.append(torch.sqrt((ue_x - proj_x) ** 2
                                             + (ue_y - proj_y) ** 2))
            wall_dists = torch.stack(wall_dists, dim=1)              # (P, 4)
            closest_idx = torch.topk(wall_dists, k=2, dim=1,
                                     largest=False).indices
            closest_mask = torch.zeros((P, 4), dtype=torch.bool, device=device)
            closest_mask.scatter_(1, closest_idx,
                                  torch.ones_like(closest_idx, dtype=torch.bool))

            # --- reflection candidate per wall (image theory) ---
            for w_idx, (wsx, wsy, wex, wey, is_vert, c) in enumerate(walls):
                if is_vert:                       # vertical wall x = c
                    img_x = 2.0 * c - ue_x
                    img_y = ue_y
                else:                             # horizontal wall y = c
                    img_x = ue_x
                    img_y = 2.0 * c - ue_y

                d_x = img_x - bs_x
                d_y = img_y - bs_y
                e_x = wex - wsx
                e_y = wey - wsy
                denom = d_x * e_y - d_y * e_x
                ok = torch.abs(denom) > 1e-10
                denom_safe = torch.where(ok, denom, torch.ones_like(denom))
                delta_x = wsx - bs_x
                delta_y = wsy - bs_y
                t = (delta_x * e_y - delta_y * e_x) / denom_safe
                u = (delta_x * d_y - delta_y * d_x) / denom_safe

                rp_x = bs_x + t * d_x
                rp_y = bs_y + t * d_y
                d_bs_rp = torch.sqrt((rp_x - bs_x) ** 2 + (rp_y - bs_y) ** 2)
                d_rp_ue = torch.sqrt((ue_x - rp_x) ** 2 + (ue_y - rp_y) ** 2)
                total_dist = d_bs_rp + d_rp_ue

                valid = (ok & (t > 0.0) & (t < 1.0) & (u >= 0.0) & (u <= 1.0)
                         & (total_dist > 1e-6) & closest_mask[:, w_idx])

                amp = -(_path_loss(total_dist, None) + 6.0)
                aoa = torch.atan2(rp_y - bs_y, rp_x - bs_x) * DEG

                inc_x = rp_x - bs_x
                inc_y = rp_y - bs_y
                inc_norm = torch.clamp(torch.sqrt(inc_x ** 2 + inc_y ** 2),
                                       min=1e-10)
                inc_comp = torch.abs(inc_x) if is_vert else torch.abs(inc_y)
                cos_theta = torch.clamp(inc_comp / inc_norm, 0.0, 1.0)
                sin2 = (1.0 - cos_theta ** 2).to(torch.complex64)
                sqrt_term = torch.sqrt(eps_c - sin2)
                cos_c = cos_theta.to(torch.complex64)
                r_te = (cos_c - sqrt_term) / (cos_c + sqrt_term)
                fresnel = torch.angle(r_te).to(torch.float32)
                phase = _coarse_phase(total_dist, fresnel)

                refl_amp_list.append(torch.where(valid, amp,
                                                 torch.full_like(amp, NEG_INF)))
                refl_aoa_list.append(aoa)
                refl_phase_list.append(phase)
                # Track the pure interaction phase (Fresnel TE angle) for the
                # excess-phase visualisation. Zero for invalid intersections.
                refl_extra_list.append(torch.where(valid, fresnel,
                                                   torch.zeros_like(fresnel)))

            # --- diffraction candidate per corner (UTD) ---
            corners = [(x_min, y_min), (x_max, y_min),
                       (x_max, y_max), (x_min, y_max)]
            for c_idx, (cx, cy) in enumerate(corners):
                d_bs_c = math.hypot(cx - bs_x, cy - bs_y)
                d_c_ue = torch.sqrt((ue_x - cx) ** 2 + (ue_y - cy) ** 2)
                total_dist = d_bs_c + d_c_ue
                amp = -(_path_loss(total_dist, None) + 30.0)
                aoa_val = math.degrees(math.atan2(cy - bs_y, cx - bs_x))

                face0, ext_ccw = _CORNER_WEDGE[c_idx]
                n_pi = _WEDGE_N * PI
                two_n = 2.0 * _WEDGE_N
                g_bs = math.atan2(bs_y - cy, bs_x - cx)
                if ext_ccw:
                    phi_p = (g_bs - face0) % (2.0 * PI)
                else:
                    phi_p = (face0 - g_bs) % (2.0 * PI)
                phi_p = min(max(phi_p, 0.01), n_pi - 0.01)

                g_ue = torch.atan2(ue_y - cy, ue_x - cx)
                if ext_ccw:
                    phi = torch.remainder(g_ue - face0, 2.0 * PI)
                else:
                    phi = torch.remainder(face0 - g_ue, 2.0 * PI)
                phi = torch.clamp(phi, 0.01, n_pi - 0.01)

                bm = phi - phi_p
                bp = phi + phi_p
                cot_sum = (_cot((PI + bm) / two_n) + _cot((PI - bm) / two_n)
                           - _cot((PI + bp) / two_n) - _cot((PI - bp) / two_n))
                base = (3.0 * PI / 4.0
                        + torch.where(cot_sum < 0.0,
                                      torch.full_like(cot_sum, PI),
                                      torch.zeros_like(cot_sum)))
                base = torch.remainder(base + PI, 2.0 * PI) - PI
                utd = torch.where(torch.abs(cot_sum) < 1e-10,
                                  torch.full_like(cot_sum, -PI / 4.0), base)
                phase = _coarse_phase(total_dist, utd)

                diff_amp_list.append(amp)
                diff_aoa_list.append(torch.full((P,), aoa_val, device=device))
                diff_phase_list.append(phase)
                diff_dist_list.append(total_dist)
                # Track the pure UTD interaction phase for the excess-phase
                # visualisation.
                diff_extra_list.append(utd)

        # ============ assemble ranked paths ============
        aoa_maps = [direct_aoa]
        amp_maps = [direct_amp]
        phase_maps = [direct_phase]
        # Excess (interaction-only) phase for path 0 is always 0: direct path
        # has no Fresnel/UTD term.
        extra_maps = [torch.zeros_like(direct_phase)]
        n_extra = num_paths - 1

        if n_extra > 0 and len(refl_amp_list) == 0:
            # no buildings -> only the direct path exists
            for _ in range(n_extra):
                aoa_maps.append(direct_aoa.clone())
                amp_maps.append(torch.full_like(direct_amp, -120.0))
                phase_maps.append(torch.zeros_like(direct_phase))
                extra_maps.append(torch.zeros_like(direct_phase))
        elif n_extra > 0:
            R_amp = torch.stack(refl_amp_list, dim=1)            # (P, 4B)
            R_aoa = torch.stack(refl_aoa_list, dim=1)
            R_phase = torch.stack(refl_phase_list, dim=1)
            R_extra = torch.stack(refl_extra_list, dim=1)
            r_amp_s, r_order = torch.sort(R_amp, dim=1, descending=True,
                                          stable=True)
            r_aoa_s = torch.gather(R_aoa, 1, r_order)
            r_phase_s = torch.gather(R_phase, 1, r_order)
            r_extra_s = torch.gather(R_extra, 1, r_order)
            n_valid = (R_amp > NEG_INF).sum(dim=1)               # (P,)

            D_amp = torch.stack(diff_amp_list, dim=1)            # (P, 4B)
            D_aoa = torch.stack(diff_aoa_list, dim=1)
            D_phase = torch.stack(diff_phase_list, dim=1)
            D_dist = torch.stack(diff_dist_list, dim=1)
            D_extra = torch.stack(diff_extra_list, dim=1)
            _, d_order = torch.sort(D_dist, dim=1, stable=True)
            d_amp_s = torch.gather(D_amp, 1, d_order)
            d_aoa_s = torch.gather(D_aoa, 1, d_order)
            d_phase_s = torch.gather(D_phase, 1, d_order)
            d_extra_s = torch.gather(D_extra, 1, d_order)
            n_diff = D_amp.shape[1]
            n_refl = R_amp.shape[1]

            for k in range(n_extra):
                use_refl = k < n_valid                           # (P,) bool
                kr = min(k, n_refl - 1)
                dk = torch.clamp(k - n_valid, min=0,
                                 max=n_diff - 1).unsqueeze(1)
                aoa_maps.append(torch.where(
                    use_refl, r_aoa_s[:, kr],
                    torch.gather(d_aoa_s, 1, dk).squeeze(1)))
                amp_maps.append(torch.where(
                    use_refl, r_amp_s[:, kr],
                    torch.gather(d_amp_s, 1, dk).squeeze(1)))
                phase_maps.append(torch.where(
                    use_refl, r_phase_s[:, kr],
                    torch.gather(d_phase_s, 1, dk).squeeze(1)))
                extra_maps.append(torch.where(
                    use_refl, r_extra_s[:, kr],
                    torch.gather(d_extra_s, 1, dk).squeeze(1)))

        # ============ inside-building amplitude penalty ============
        inside = torch.zeros(P, dtype=torch.bool, device=device)
        for building in self.buildings:
            x_min = float(building['x']); y_min = float(building['y'])
            x_max = x_min + float(building['width'])
            y_max = y_min + float(building['height'])
            inside = inside | ((ue_x >= x_min) & (ue_x <= x_max)
                               & (ue_y >= y_min) & (ue_y <= y_max))
        for k in range(len(amp_maps)):
            amp_maps[k] = torch.where(inside, amp_maps[k] - 30.0, amp_maps[k])

        # ============ reshape -> list of (n_y, n_x) numpy arrays ============
        aoa_out = [m.reshape(n_y, n_x).detach().cpu().numpy().astype(np.float32)
                   for m in aoa_maps]
        amp_out = [m.reshape(n_y, n_x).detach().cpu().numpy().astype(np.float32)
                   for m in amp_maps]
        phase_out = [m.reshape(n_y, n_x).detach().cpu().numpy().astype(np.float32)
                     for m in phase_maps]
        los_out = los_bool.detach().cpu().numpy().astype(bool)
        # Excess (interaction-only) phase per path, wrapped to [-pi, pi].
        # Stashed on self so generate_excess_phase_map_gpu() can read it; we
        # don't change the return signature here for backward compatibility.
        extra_out = [
            (((m + PI) % (2.0 * PI)) - PI)
            .reshape(n_y, n_x).detach().cpu().numpy().astype(np.float32)
            for m in extra_maps
        ]
        self._last_extra_maps = extra_out

        return aoa_out, amp_out, phase_out, los_out

    def _generate_ranked_path_maps_scalar(self, num_paths=3):
        """Reference per-UE implementation (validation only, NOT production).

        This is the original Python double-loop version. It is kept so that
        ``_generate_ranked_path_maps_vectorized`` can be numerically validated
        against it (see validate_vectorized.py).
        """
        if self.bs_pos is None:
            raise ValueError("Base station position not set")

        x_np = self.X.detach().cpu().numpy()
        y_np = self.Y.detach().cpu().numpy()
        n_y, n_x = x_np.shape
        bs_pos = self.bs_pos.detach().cpu().numpy().astype(np.float64)

        if self.building_edges is None:
            los_map = np.ones((n_y, n_x), dtype=bool)
        else:
            los_map = check_los_gpu(
                self.ue_positions, self.bs_pos, self.building_edges, self.device
            ).detach().cpu().numpy().astype(bool)

        aoa_maps = [np.zeros((n_y, n_x), dtype=np.float32) for _ in range(num_paths)]
        amplitude_maps = [np.full((n_y, n_x), -120.0, dtype=np.float32) for _ in range(num_paths)]
        phase_maps = [np.zeros((n_y, n_x), dtype=np.float32) for _ in range(num_paths)]

        for i in range(n_y):
            for j in range(n_x):
                ue_pos = np.array([x_np[i, j], y_np[i, j]], dtype=np.float64)
                los = bool(los_map[i, j])
                ranked_paths = self._build_ranked_paths_for_ue_np(ue_pos, bs_pos, los, num_paths)
                for k, path in enumerate(ranked_paths[:num_paths]):
                    aoa_maps[k][i, j] = path['aoa']
                    amplitude_maps[k][i, j] = path['amplitude_db']
                    phase_maps[k][i, j] = path['phase']

        inside_building_mask = np.zeros((n_y, n_x), dtype=bool)
        for building in self.buildings:
            x_min, y_min = building['x'], building['y']
            x_max = x_min + building['width']
            y_max = y_min + building['height']
            inside = (x_np >= x_min) & (x_np <= x_max) & (y_np >= y_min) & (y_np <= y_max)
            inside_building_mask |= inside

        for k in range(num_paths):
            amplitude_maps[k][inside_building_mask] -= 30.0

        return aoa_maps, amplitude_maps, phase_maps, los_map

    def generate_aoa_map_gpu(self, num_paths=3):
        """
        Generate AoA maps for ranked paths.

        Parameters:
        -----------
        num_paths : int
            Number of paths to calculate (ranked by amplitude strength)
            
        Returns:
        --------
        aoa_maps : list of 2D arrays
            AoA values for each path at each grid point (strongest to weakest)
        los_map : 2D array
            Boolean map indicating LoS condition
        """
        aoa_maps, _, _, los_map = self._generate_ranked_path_maps_gpu(num_paths=num_paths)

        if self.verbose:
            print(f"Generated {len(aoa_maps)} AoA maps using shared ranked path assignment")

        return aoa_maps, los_map
    
    def _calculate_reflection_points_vectorized(self, ue_positions_flat, building):
        """Calculate reflection points for ALL UE positions at once"""
        total_positions = ue_positions_flat.shape[0]
        
        x_min, y_min = building['x'], building['y']
        x_max = x_min + building['width']
        y_max = y_min + building['height']
        
        # Broadcast BS position for all UE positions
        bs_pos_broadcast = self.bs_pos.unsqueeze(0).expand(total_positions, -1)  # (total_positions, 2)
        
        # Calculate midpoints for all UE positions
        mid_points = (bs_pos_broadcast + ue_positions_flat) / 2  # (total_positions, 2)
        
        # Calculate closest points on each edge for ALL positions simultaneously
        edge_points = []
        
        # Bottom edge
        bottom_points = torch.stack([
            torch.clamp(mid_points[:, 0], x_min, x_max),
            torch.full((total_positions,), y_min, device=self.device)
        ], dim=1)
        edge_points.append(bottom_points)
        
        # Right edge  
        right_points = torch.stack([
            torch.full((total_positions,), x_max, device=self.device),
            torch.clamp(mid_points[:, 1], y_min, y_max)
        ], dim=1)
        edge_points.append(right_points)
        
        # Top edge
        top_points = torch.stack([
            torch.clamp(mid_points[:, 0], x_min, x_max),
            torch.full((total_positions,), y_max, device=self.device)
        ], dim=1)
        edge_points.append(top_points)
        
        # Left edge
        left_points = torch.stack([
            torch.full((total_positions,), x_min, device=self.device),
            torch.clamp(mid_points[:, 1], y_min, y_max)
        ], dim=1)
        edge_points.append(left_points)
        
        # Calculate distances to each edge for all positions
        edge_stack = torch.stack(edge_points, dim=1)  # (total_positions, 4, 2)
        distances = torch.norm(edge_stack - mid_points.unsqueeze(1), dim=2)  # (total_positions, 4)
        
        # Find closest edge for each position
        closest_indices = torch.argmin(distances, dim=1)  # (total_positions,)
        
        # Gather closest points
        closest_points = torch.gather(
            edge_stack, 
            1, 
            closest_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, 2)
        ).squeeze(1)  # (total_positions, 2)
        
        return closest_points
    
    def calculate_reflection_point_gpu(self, ue_pos, building):
        """Calculate reflection point on building wall (GPU version)"""
        x_min, y_min = building['x'], building['y']
        x_max = x_min + building['width']
        y_max = y_min + building['height']
        
        # Find closest point on building perimeter to midpoint of BS-UE line
        mid_point = (self.bs_pos + ue_pos) / 2
        
        # Check distance to each edge and return closest point
        edges = [
            torch.tensor([torch.clamp(mid_point[0], x_min, x_max), y_min], device=self.device),  # bottom
            torch.tensor([x_max, torch.clamp(mid_point[1], y_min, y_max)], device=self.device),  # right
            torch.tensor([torch.clamp(mid_point[0], x_min, x_max), y_max], device=self.device),  # top
            torch.tensor([x_min, torch.clamp(mid_point[1], y_min, y_max)], device=self.device),  # left
        ]
        
        distances = [torch.norm(edge - mid_point) for edge in edges]
        closest_idx = torch.argmin(torch.stack(distances))
        return edges[closest_idx]
    
    def calculate_diffraction_point_gpu(self, ue_pos, building):
        """Calculate diffraction point (closest corner by total path length)"""
        corners = [torch.tensor(corner, dtype=torch.float32, device=self.device) for corner in building['corners']]
        
        # Calculate total path length for each corner
        total_distances = []
        for corner in corners:
            dist_bs_to_corner = torch.norm(corner - self.bs_pos)
            dist_corner_to_ue = torch.norm(ue_pos - corner)
            total_distances.append(dist_bs_to_corner + dist_corner_to_ue)
        
        # Return corner with minimum total path length
        closest_idx = torch.argmin(torch.stack(total_distances))
        return corners[closest_idx]
    
    def generate_aoa_maps_batch_gpu(self, bs_positions_tensor, num_paths=3):
        """Generate AoA maps for multiple BS positions simultaneously using GPU vectorization."""
        batch_size = bs_positions_tensor.shape[0]
        n_y, n_x = self.X.shape
        
        if self.verbose:
            print(f"Generating AoA maps for {batch_size} BS positions in batch mode")
        
        # Initialize results for each path
        all_aoa_maps = []
        
        # Process each BS position in the batch
        for bs_idx, bs_pos in enumerate(bs_positions_tensor):
            # Set BS position for this iteration
            self.bs_pos = bs_pos
            
            # Generate AoA map for this BS position
            aoa_maps, _ = self.generate_aoa_map_gpu(num_paths=num_paths)
            
            # Convert numpy arrays to tensors
            aoa_tensors = [torch.from_numpy(aoa_map).to(self.device) for aoa_map in aoa_maps]
            
            # Stack tensors for this BS position
            bs_aoa_stack = torch.stack(aoa_tensors, dim=0)  # [num_paths, n_y, n_x]
            all_aoa_maps.append(bs_aoa_stack)
        
        # Stack results across batch dimension
        all_aoa_maps = torch.stack(all_aoa_maps, dim=0)  # [batch_size, num_paths, n_y, n_x]
        
        # Generate LOS maps for all positions
        all_los_maps = self._compute_los_batch_vectorized(bs_positions_tensor)
        
        return all_aoa_maps, all_los_maps
        
    def generate_amplitude_maps_batch_gpu(self, bs_positions_tensor, num_paths=3):
        """Generate amplitude maps for multiple BS positions simultaneously using GPU vectorization."""
        batch_size = bs_positions_tensor.shape[0]
        n_y, n_x = self.X.shape
        
        if self.verbose:
            print(f"Generating amplitude maps for {batch_size} BS positions in batch mode")
        
        # Initialize results for each path
        all_amplitude_maps = []
        
        # Process each BS position in the batch
        for bs_idx, bs_pos in enumerate(bs_positions_tensor):
            # Set BS position for this iteration
            self.bs_pos = bs_pos
            
            # Generate amplitude map for this BS position
            amplitude_maps = self.generate_amplitude_map_gpu(num_paths=num_paths)
            
            # Convert numpy arrays to tensors
            amplitude_tensors = [torch.from_numpy(amp_map).to(self.device) for amp_map in amplitude_maps]
            
            # Stack tensors for this BS position
            bs_amplitude_stack = torch.stack(amplitude_tensors, dim=0)  # [num_paths, n_y, n_x]
            all_amplitude_maps.append(bs_amplitude_stack)
        
        # Stack results across batch dimension
        all_amplitude_maps = torch.stack(all_amplitude_maps, dim=0)  # [batch_size, num_paths, n_y, n_x]
        
        return all_amplitude_maps
        
        # Initialize results for each path
        all_amplitude_maps = []
        
        # Process each path
        for path_idx in range(num_paths):
            # Create batch result tensor
            amplitude_batch = torch.zeros(batch_size, n_y, n_x, device=self.device)
            
            # Process each BS position in the batch
            for bs_idx, bs_pos in enumerate(bs_positions_tensor):
                # Set BS position for this iteration
                self.bs_pos = bs_pos
                
                # Generate amplitude map for this BS position
                amplitude_maps = self.generate_amplitude_map_gpu(num_paths=num_paths)
                
                # Store the result for this path
                if path_idx < len(amplitude_maps):
                    amplitude_batch[bs_idx] = torch.from_numpy(amplitude_maps[path_idx]).to(self.device)
            
            all_amplitude_maps.append(amplitude_batch)
        
        return all_amplitude_maps
    
    def _compute_los_batch_vectorized(self, bs_positions_tensor):
        """Compute LOS maps for all BS positions in the batch."""
        batch_size = bs_positions_tensor.shape[0]
        n_y, n_x = self.X.shape
        
        # Create batch result tensor
        los_batch = torch.zeros(batch_size, n_y, n_x, device=self.device, dtype=torch.bool)
        
        # Process each BS position
        for bs_idx, bs_pos in enumerate(bs_positions_tensor):
            # Set BS position
            self.bs_pos = bs_pos
            
            # Compute LOS for this BS position
            los_map = check_los_gpu(self.ue_positions, self.bs_pos, self.building_edges, self.device)
            los_batch[bs_idx] = los_map
        
        return los_batch

    def calculate_path_loss_single(self, distance, los):
        """Calculate path loss for a single path (CPU version for individual calculations)"""
        frequency = 2.4e9  # 2.4 GHz
        wavelength = 3e8 / frequency

        # Avoid division by zero
        distance = max(distance, 1e-6)

        pl = 20 * np.log10(4 * np.pi * distance / wavelength)

        # Add penetration loss for NLOS
        if not los:
            pl += 15  # dB

        return pl

    @staticmethod
    def _wrap_phase_np(phase):
        """Wrap a phase value (radians) into [-pi, pi]."""
        return ((phase + np.pi) % (2.0 * np.pi)) - np.pi

    def calculate_phase_single(self, distance, extra_phase=0.0):
        """
        Return the *coarse-grained* phase of a propagation path.

        The full phase  psi = -k*d + interaction  oscillates ~8 times per
        metre at 2.4 GHz (lambda=0.125 m) and is spatially aliased on a
        1-m grid (16x undersampled).

        We coarsen the propagation term by N=256, giving an effective
        wavelength lambda_eff = 32 m and ~4 smooth cycles across a 128-m
        map (32 samples/cycle on the 1-m grid — well above Nyquist).

          phase = -2*pi*d / (N*lambda) + interaction_phase

          * LOS path        -> smooth radial gradient from BS
          * Reflected path  -> Fresnel phase + smooth distance modulation
          * Diffracted path -> UTD phase   + smooth distance modulation

        Returns a value wrapped to [-pi, pi].
        """
        coarse_propagation = -2.0 * np.pi * distance / (_PHASE_COARSEN * _WAVELENGTH)
        return self._wrap_phase_np(coarse_propagation + extra_phase)

    @staticmethod
    def _fresnel_te_phase_np(cos_theta_i):
        """Phase of Fresnel TE reflection coefficient for concrete at 2.4 GHz."""
        sin2 = 1.0 - cos_theta_i ** 2
        sqrt_term = np.sqrt(_CONCRETE_EPS - sin2)
        R_TE = (cos_theta_i - sqrt_term) / (cos_theta_i + sqrt_term)
        return float(np.angle(R_TE))

    @staticmethod
    def _wall_incidence_cos_np(incoming_vec, wall_start, wall_end):
        """Cosine of incidence angle (from wall normal) for an axis-aligned wall."""
        d = np.linalg.norm(incoming_vec)
        if d < 1e-10:
            return 1.0
        if abs(wall_start[0] - wall_end[0]) < 1e-8:
            return abs(incoming_vec[0]) / d
        if abs(wall_start[1] - wall_end[1]) < 1e-8:
            return abs(incoming_vec[1]) / d
        return 1.0

    @staticmethod
    def _global_to_wedge_angle_np(global_angle, face0_angle, exterior_ccw):
        """Map a global direction angle into the UTD wedge coordinate."""
        if exterior_ccw:
            return (global_angle - face0_angle) % (2.0 * np.pi)
        return (face0_angle - global_angle) % (2.0 * np.pi)

    def _utd_diffraction_phase_te_np(self, corner_pos, corner_idx, bs_pos, ue_pos):
        """Phase of UTD diffraction coefficient (TE, PEC 90-deg wedge, F~1)."""
        face0_angle, ext_ccw = _CORNER_WEDGE[corner_idx]
        n = _WEDGE_N

        phi_prime = self._global_to_wedge_angle_np(
            np.arctan2(*(bs_pos - corner_pos)[::-1]), face0_angle, ext_ccw)
        phi = self._global_to_wedge_angle_np(
            np.arctan2(*(ue_pos - corner_pos)[::-1]), face0_angle, ext_ccw)

        n_pi = n * np.pi
        phi = np.clip(phi, 0.01, n_pi - 0.01)
        phi_prime = np.clip(phi_prime, 0.01, n_pi - 0.01)

        two_n = 2.0 * n
        bm, bp = phi - phi_prime, phi + phi_prime

        def _cot(x):
            s = np.sin(x)
            return np.cos(x) / s if abs(s) > 1e-6 else 0.0

        cot_sum = (_cot((np.pi + bm) / two_n) + _cot((np.pi - bm) / two_n)
                   - _cot((np.pi + bp) / two_n) - _cot((np.pi - bp) / two_n))

        if abs(cot_sum) < 1e-10:
            return -np.pi / 4.0
        phase = 3.0 * np.pi / 4.0 + (np.pi if cot_sum < 0 else 0.0)
        return self._wrap_phase_np(phase)

    def generate_amplitude_map_gpu(self, num_paths=3):
        """
        Generate amplitude maps for ranked paths.
        
        Returns:
        --------
        amplitude_maps : list of 2D arrays
            Amplitude values for each path in dB
        """
        _, amplitude_maps, _, _ = self._generate_ranked_path_maps_gpu(num_paths=num_paths)

        if self.verbose:
            print(f"Generated {len(amplitude_maps)} amplitude maps")
        return amplitude_maps

    def generate_phase_map_gpu(self, num_paths=3):
        """
        Generate phase shift maps for ranked paths.

        Returns
        -------
        phase_maps : list of 2D arrays
            Phase shift in radians, wrapped to [-pi, pi], one map per ranked path.
        """
        _, _, phase_maps, _ = self._generate_ranked_path_maps_gpu(num_paths=num_paths)
        if self.verbose:
            print(f"Generated {len(phase_maps)} phase maps")
        return phase_maps

    def generate_phase_map(self, num_paths=3):
        """Backward compatibility wrapper for generate_phase_map_gpu."""
        return self.generate_phase_map_gpu(num_paths)

    def plot_phase_map(self, phase_maps, path_names=None):
        """
        Plot phase-shift maps for all ranked paths using a circular colormap.
        """
        if isinstance(phase_maps, torch.Tensor):
            phase_maps = [m.cpu().numpy() for m in phase_maps]
        elif isinstance(phase_maps, list):
            phase_maps = [
                m.cpu().numpy() if isinstance(m, torch.Tensor) else m
                for m in phase_maps
            ]

        num_paths = len(phase_maps)
        if path_names is None:
            path_names = [f'Path {i+1}' for i in range(num_paths)]

        # ===== Excess phase visualisation =====
        # The stored phase is total_phase = wrap(-2*pi*d/(N*lambda) + interaction).
        # By subtracting the smooth propagation baseline -2*pi*d_direct/(N*lambda)
        # we recover the interaction component (Fresnel reflection / UTD diffraction
        # phase), which is what actually carries the building physics. For the
        # direct/LOS path this is zero everywhere (uniform colour); for reflected
        # and diffracted paths it forms distinct "regions" per active wall/corner.
        X_cpu = self.X.cpu().numpy()
        Y_cpu = self.Y.cpu().numpy()
        bs_pos_cpu = self.bs_pos.cpu().numpy()
        d_direct = np.sqrt((X_cpu - bs_pos_cpu[0]) ** 2
                           + (Y_cpu - bs_pos_cpu[1]) ** 2)
        propagation_baseline = -2.0 * np.pi * d_direct / (_PHASE_COARSEN * _WAVELENGTH)

        def _excess(p_2d):
            ex = p_2d - propagation_baseline
            return (ex + np.pi) % (2.0 * np.pi) - np.pi   # wrap to [-pi, pi]

        excess_maps = [_excess(np.squeeze(m)) for m in phase_maps]

        fig, axes = plt.subplots(1, num_paths, figsize=(5 * num_paths, 4))
        if num_paths == 1:
            axes = [axes]

        for idx, (excess_2d, name) in enumerate(zip(excess_maps, path_names)):
            ax = axes[idx]

            im = ax.contourf(X_cpu, Y_cpu, excess_2d, levels=40,
                             cmap='twilight', vmin=-np.pi, vmax=np.pi)
            im.set_clim(-np.pi, np.pi)

            for building in self.buildings:
                rect = Rectangle(
                    (building['x'], building['y']),
                    building['width'], building['height'],
                    linewidth=2, edgecolor='black',
                    facecolor='gray', alpha=0.7,
                )
                ax.add_patch(rect)

            ax.plot(bs_pos_cpu[0], bs_pos_cpu[1], 'r*', markersize=20,
                    label='Base Station', markeredgecolor='black',
                    markeredgewidth=1)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{name} - Excess Phase')
            ax.set_aspect('equal')
            ax.legend()

            cbar = plt.colorbar(im, ax=ax, ticks=[-np.pi, 0, np.pi])
            cbar.ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
            cbar.set_label('Excess phase (rad)')

        plt.tight_layout()
        plt.show()
    
    # Backward compatibility methods
    def generate_aoa_map(self, num_paths=3):
        """Backward compatibility wrapper"""
        return self.generate_aoa_map_gpu(num_paths)
    
    def generate_amplitude_map(self, num_paths=3):
        """Backward compatibility wrapper"""
        return self.generate_amplitude_map_gpu(num_paths)
    
    def plot_aoa_map(self, aoa_maps, los_map, path_names=None):
        """
        Plot AoA maps for all paths, reflecting the strongest paths.

        Parameters:
        -----------
        aoa_maps : list of 2D arrays or 4D tensor
            AoA values for each path
        los_map : 2D array
            LoS condition map
        path_names : list of str
            Names for each path (e.g., ['Strongest Path', 'Second Strongest', 'Third Strongest'])
        """
        # Handle both list of arrays and tensor input
        if isinstance(aoa_maps, torch.Tensor):
            # Convert tensor to CPU and handle dimensions
            aoa_tensor = aoa_maps.cpu()
            if aoa_tensor.dim() == 4:
                # 4D tensor: (num_paths, channels, height, width)
                num_paths = aoa_tensor.shape[0]
                aoa_maps = []
                for i in range(num_paths):
                    # Take first channel and squeeze
                    map_2d = aoa_tensor[i, 0, :, :].squeeze().numpy()
                    aoa_maps.append(map_2d)
            elif aoa_tensor.dim() == 3:
                # 3D tensor: (num_paths, height, width)
                num_paths = aoa_tensor.shape[0]
                aoa_maps = [aoa_tensor[i, :, :].squeeze().numpy() for i in range(num_paths)]
            elif aoa_tensor.dim() == 2:
                # 2D tensor: single map
                aoa_maps = [aoa_tensor.squeeze().numpy()]
            else:
                # Higher dimensions, try to squeeze to 2D
                aoa_maps = [aoa_tensor.squeeze().numpy()]
        elif isinstance(aoa_maps, list) and len(aoa_maps) > 0:
            # If it's a list of tensors, convert each to numpy
            processed_maps = []
            for aoa_map in aoa_maps:
                if isinstance(aoa_map, torch.Tensor):
                    # Convert to CPU and squeeze extra dimensions
                    map_cpu = aoa_map.cpu().squeeze()
                    if map_cpu.dim() > 2:
                        # Take last 2 dimensions if still > 2D
                        map_cpu = map_cpu[-2:] if map_cpu.dim() == 3 else map_cpu
                    processed_maps.append(map_cpu.numpy())
                elif isinstance(aoa_map, np.ndarray):
                    # Already numpy array, squeeze all extra dimensions
                    squeezed_map = np.squeeze(aoa_map)
                    # If still more than 2D after squeezing, take the last 2 dimensions
                    while squeezed_map.ndim > 2:
                        squeezed_map = squeezed_map[0] if squeezed_map.shape[0] == 1 else squeezed_map[-1]
                    processed_maps.append(squeezed_map)
                else:
                    processed_maps.append(aoa_map)
            aoa_maps = processed_maps
        elif isinstance(aoa_maps, np.ndarray):
            # Single numpy array
            if aoa_maps.ndim > 2:
                aoa_maps = [np.squeeze(aoa_maps)]
            else:
                aoa_maps = [aoa_maps]

        num_paths = len(aoa_maps)

        if path_names is None:
            path_names = [f'Path {i+1}' for i in range(num_paths)]

        fig, axes = plt.subplots(1, num_paths + 1, figsize=(5 * (num_paths + 1), 4))

        if num_paths + 1 == 1:
            axes = [axes]
        elif not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        # Convert tensors to CPU numpy arrays for plotting
        X_cpu = self.X.cpu().numpy()
        Y_cpu = self.Y.cpu().numpy()
        bs_pos_cpu = self.bs_pos.cpu().numpy()
        
        # Verify grid and map dimensions match
        if len(aoa_maps) > 0:
            map_shape = aoa_maps[0].shape
            if map_shape != X_cpu.shape:
                print(f"Warning: Map shape {map_shape} doesn't match grid shape {X_cpu.shape}")
                # Try to resize if needed
                if len(map_shape) == 2 and len(X_cpu.shape) == 2:
                    print("Attempting to interpolate maps to match grid...")
                    try:
                        from scipy.ndimage import zoom
                        zoom_factors = [X_cpu.shape[i] / map_shape[i] for i in range(2)]
                        aoa_maps = [zoom(aoa_map, zoom_factors, order=1) for aoa_map in aoa_maps]
                    except ImportError:
                        print("scipy not available, cannot resize maps")

        # Plot AoA for each path
        for idx, (aoa_map, name) in enumerate(zip(aoa_maps, path_names)):
            ax = axes[idx]

            # Ensure aoa_map is 2D
            if aoa_map.ndim > 2:
                aoa_map = np.squeeze(aoa_map)
            elif aoa_map.ndim < 2:
                print(f"Warning: AoA map {idx} has insufficient dimensions: {aoa_map.shape}")
                continue
                
            # Final check that dimensions are exactly 2
            if aoa_map.ndim != 2:
                print(f"Error: Cannot plot AoA map {idx} with shape {aoa_map.shape}")
                continue

            # Use the CPU numpy arrays here
            im = ax.contourf(X_cpu, Y_cpu, aoa_map, levels=20, cmap='twilight')
            im.set_clim(-180, 180)
            ax.contour(X_cpu, Y_cpu, aoa_map, levels=10, colors='white', 
                    linewidths=0.5, alpha=0.3)

            # Plot buildings
            for building in self.buildings:
                rect = Rectangle((building['x'], building['y']), 
                            building['width'], building['height'],
                            linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
                ax.add_patch(rect)

            # Plot BS - use CPU numpy array
            ax.plot(bs_pos_cpu[0], bs_pos_cpu[1], 'r*', markersize=20, 
                label='Base Station', markeredgecolor='black', markeredgewidth=1)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{name} - AoA Map')
            ax.set_aspect('equal')
            ax.legend()

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('AoA (degrees)')

        # Plot LoS map
        ax = axes[num_paths]
        
        # Handle LoS map - ensure it's 2D
        if isinstance(los_map, torch.Tensor):
            los_map = los_map.cpu().numpy()
        if los_map.ndim > 2:
            los_map = los_map.squeeze()
        
        # Use CPU numpy arrays here too
        im = ax.contourf(X_cpu, Y_cpu, los_map.astype(float), levels=[0, 0.5, 1], 
                        cmap='hsv', alpha=0.6)

        for building in self.buildings:
            rect = Rectangle((building['x'], building['y']), 
                        building['width'], building['height'],
                        linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(rect)

        ax.plot(bs_pos_cpu[0], bs_pos_cpu[1], 'r*', markersize=20, 
            label='Base Station', markeredgecolor='white', markeredgewidth=1)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('LoS Condition')
        ax.set_aspect('equal')
        ax.legend()

        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.set_label('LoS')
        cbar.ax.set_yticklabels(['NLoS', 'LoS'])

        plt.tight_layout()
        plt.show()

    def plot_amplitude_map(self, amplitude_maps, path_names=None):
        """
        Plot amplitude maps for all paths, reflecting the strongest paths.

        Parameters:
        -----------
        amplitude_maps : list of 2D arrays or 4D tensor
            Amplitude values for each path
        path_names : list of str
            Names for each path (e.g., ['Strongest Path', 'Second Strongest', 'Third Strongest'])
        """
        # Handle both list of arrays and tensor input
        if isinstance(amplitude_maps, torch.Tensor):
            # Convert tensor to CPU and handle dimensions
            amp_tensor = amplitude_maps.cpu()
            if amp_tensor.dim() == 4:
                # 4D tensor: (num_paths, channels, height, width)
                num_paths = amp_tensor.shape[0]
                amplitude_maps = []
                for i in range(num_paths):
                    # Take first channel and squeeze
                    map_2d = amp_tensor[i, 0, :, :].squeeze().numpy()
                    amplitude_maps.append(map_2d)
            elif amp_tensor.dim() == 3:
                # 3D tensor: (num_paths, height, width)
                num_paths = amp_tensor.shape[0]
                amplitude_maps = [amp_tensor[i, :, :].squeeze().numpy() for i in range(num_paths)]
            elif amp_tensor.dim() == 2:
                # 2D tensor: single map
                amplitude_maps = [amp_tensor.squeeze().numpy()]
            else:
                # Higher dimensions, try to squeeze to 2D
                amplitude_maps = [amp_tensor.squeeze().numpy()]
        elif isinstance(amplitude_maps, list) and len(amplitude_maps) > 0:
            # If it's a list of tensors, convert each to numpy
            processed_maps = []
            for amp_map in amplitude_maps:
                if isinstance(amp_map, torch.Tensor):
                    # Convert to CPU and squeeze extra dimensions
                    map_cpu = amp_map.cpu().squeeze()
                    if map_cpu.dim() > 2:
                        # Take last 2 dimensions if still > 2D
                        map_cpu = map_cpu[-2:] if map_cpu.dim() == 3 else map_cpu
                    processed_maps.append(map_cpu.numpy())
                elif isinstance(amp_map, np.ndarray):
                    # Already numpy array, squeeze all extra dimensions
                    squeezed_map = np.squeeze(amp_map)
                    # If still more than 2D after squeezing, take the last 2 dimensions
                    while squeezed_map.ndim > 2:
                        squeezed_map = squeezed_map[0] if squeezed_map.shape[0] == 1 else squeezed_map[-1]
                    processed_maps.append(squeezed_map)
                else:
                    processed_maps.append(amp_map)
            amplitude_maps = processed_maps
        elif isinstance(amplitude_maps, np.ndarray):
            # Single numpy array
            if amplitude_maps.ndim > 2:
                amplitude_maps = [np.squeeze(amplitude_maps)]
            else:
                amplitude_maps = [amplitude_maps]

        num_paths = len(amplitude_maps)

        if path_names is None:
            path_names = [f'Path {i+1}' for i in range(num_paths)]

        fig, axes = plt.subplots(1, num_paths, figsize=(5 * num_paths, 4))

        if num_paths == 1:
            axes = [axes]
        elif not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        # Convert tensors to CPU numpy arrays for plotting
        X_cpu = self.X.cpu().numpy()
        Y_cpu = self.Y.cpu().numpy()
        bs_pos_cpu = self.bs_pos.cpu().numpy()
        
        # Verify grid and map dimensions match
        if len(amplitude_maps) > 0:
            map_shape = amplitude_maps[0].shape
            if map_shape != X_cpu.shape:
                print(f"Warning: Map shape {map_shape} doesn't match grid shape {X_cpu.shape}")
                # Try to resize if needed
                if len(map_shape) == 2 and len(X_cpu.shape) == 2:
                    print("Attempting to interpolate maps to match grid...")
                    try:
                        from scipy.ndimage import zoom
                        zoom_factors = [X_cpu.shape[i] / map_shape[i] for i in range(2)]
                        amplitude_maps = [zoom(amp_map, zoom_factors, order=1) for amp_map in amplitude_maps]
                    except ImportError:
                        print("scipy not available, cannot resize maps")
        
        # Calculate common colorbar range for all amplitude maps
        if len(amplitude_maps) > 0:
            all_valid_maps = [amp_map for amp_map in amplitude_maps if amp_map.ndim == 2]
            if all_valid_maps:
                global_min = min(amp_map.min() for amp_map in all_valid_maps)
                global_max = max(amp_map.max() for amp_map in all_valid_maps)
                
                # Ensure reasonable range
                if global_max - global_min < 1:
                    # Very narrow range, expand it
                    global_min, global_max = global_min - 5, global_max + 5
                elif global_min > -120 and global_max < -30:
                    # Already reasonable amplitude range, use as is
                    pass
                else:
                    # Extend range to reasonable defaults while encompassing data
                    global_min = min(-90, global_min)
                    global_max = max(-40, global_max)
                
                print(f"Using common colorbar range: [{global_min:.1f}, {global_max:.1f}] dB")
            else:
                global_min, global_max = -90, -40

        for idx, (amp_map, name) in enumerate(zip(amplitude_maps, path_names)):
            ax = axes[idx]

            # Ensure amp_map is 2D
            if amp_map.ndim > 2:
                amp_map = np.squeeze(amp_map)
            elif amp_map.ndim < 2:
                print(f"Warning: Amplitude map {idx} has insufficient dimensions: {amp_map.shape}")
                continue
                
            # Final check that dimensions are exactly 2
            if amp_map.ndim != 2:
                print(f"Error: Cannot plot amplitude map {idx} with shape {amp_map.shape}")
                continue

            # Use CPU numpy arrays here
            im = ax.contourf(X_cpu, Y_cpu, amp_map, levels=20, cmap='hot_r')
            
            # Use the common colorbar range for all amplitude maps
            im.set_clim(global_min, global_max)
            
            # Plot buildings
            for building in self.buildings:
                rect = Rectangle((building['x'], building['y']), 
                            building['width'], building['height'],
                            linewidth=2, edgecolor='black', facecolor='gray', alpha=0.7)
                ax.add_patch(rect)

            # Plot BS - use CPU numpy array
            ax.plot(bs_pos_cpu[0], bs_pos_cpu[1], 'r*', markersize=20, 
                label='Base Station', markeredgecolor='black', markeredgewidth=1)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'{name} - Amplitude')
            ax.set_aspect('equal')
            ax.legend()

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'Amplitude (dB)\nRange: [{global_min:.1f}, {global_max:.1f}]')

        plt.tight_layout()
        plt.show()


# For backward compatibility, replace the original class
RayTracingAoAMap = RayTracingAoAMapGPU


if __name__ == "__main__":
    # Performance test
    import time
    
    print("Testing GPU-accelerated RayTracingAoAMap...")
    
    # Create test scenario
    rt = RayTracingAoAMapGPU(map_size=(100, 100), grid_spacing=2, device='auto', verbose=True)
    rt.set_base_station(80, 30)
    rt.add_building(20, 20, 30, 15)
    rt.add_building(75, 56, 25, 19)
    
    print("Starting ray tracing computation...")
    start_time = time.time()
    
    aoa_maps, los_map = rt.generate_aoa_map_gpu(num_paths=3)
    amplitude_maps = rt.generate_amplitude_map_gpu(num_paths=3)

    path_names = ['Strongest Path', 'Second Strongest', 'Third Strongest']
    rt.plot_aoa_map(aoa_maps, los_map, path_names)
    rt.plot_amplitude_map(amplitude_maps, path_names)
    
    end_time = time.time()
    
    print(f"Generated maps in {end_time - start_time:.3f} seconds")
    print(f"Map shape: {aoa_maps[0].shape}")
    print(f"LOS percentage: {np.sum(los_map)/los_map.size*100:.1f}%")
    print(f"Device used: {rt.device}")
    
    # Test memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")