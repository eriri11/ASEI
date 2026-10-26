"""Fixed-core radial augmentation used by MR-to-CT MT."""
import math
from collections.abc import Mapping
import torch
import torch.nn.functional as F

def radial_radius(r, amplitude, inner=24., outer=60.):
    if not 0 <= inner < outer or abs(amplitude) * math.pi >= outer - inner:
        raise ValueError("Radial derivative must remain strictly positive")
    t = ((r - inner) / (outer - inner)).clamp(0, 1)
    bump = torch.sin(math.pi * t).square()
    # Avoid finite precision sin(pi) causing a displacement outside support.
    return r + amplitude * torch.where((r > inner) & (r < outer), bump, 0.)

def context_grids(height, width, phase=0, amplitude=6., inner=24., outer=60.,
                  period=128, device="cpu"):
    """Four regular phases cover seedless regions, independent of predictions.

    Circle centers are at image/cell boundaries or cell midpoints. The support
    radius is smaller than half a cell so neighboring maps never overlap.
    Returns pullback, exact numerical inverse, fixed-core mask, valid mask.
    """
    if outer >= period / 2 or phase not in range(4):
        raise ValueError("Nonoverlapping disks and phase 0..3 required")
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                          torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    ox, oy = (phase % 2) * period / 2 - .5, (phase // 2) * period / 2 - .5
    cx = ox + period * torch.round((x - ox) / period)
    cy = oy + period * torch.round((y - oy) / period)
    dx, dy = x - cx, y - cy
    r = torch.sqrt(dx.square() + dy.square())
    mapped = radial_radius(r, amplitude, inner, outer)
    lo, hi = torch.full_like(r, inner), torch.full_like(r, outer)
    for _ in range(24):
        mid = (lo + hi) / 2
        below = radial_radius(mid, amplitude, inner, outer) < r
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    inv_r = torch.where((r > inner) & (r < outer), (lo + hi) / 2, r)

    def coordinates(new_r):
        factor = new_r / r.clamp_min(1e-8)
        # The additive form is exact identity on the fixed core.
        sx, sy = x + dx * (factor - 1), y + dy * (factor - 1)
        valid = (sx >= -.5) & (sx <= width - .5) & (sy >= -.5) & (sy <= height - .5)
        grid = torch.stack((2 * (sx + .5) / width - 1, 2 * (sy + .5) / height - 1), -1)[None]
        return grid, valid

    grid, valid = coordinates(mapped)
    inverse, iv = coordinates(inv_r)
    return grid, inverse, r <= inner, valid & iv

def resample(value, grid):
    return F.grid_sample(value, grid.expand(len(value), -1, -1, -1),
                         mode="bilinear", padding_mode="border", align_corners=False)

class SpatialContextAugmentation:
    """Training geometry shared by the strong image and weak teacher target.

    A private CPU generator preserves the existing data/appearance RNG stream.
    Grids are cached; neither prototypes nor extra teacher predictions are used.
    """

    def __init__(self, config):
        if not isinstance(config, Mapping):
            raise TypeError("spatial_context must be a configuration mapping")
        options = dict(config)
        options.pop("enabled", None)
        name = options.pop("name", "fixed_core_radial")
        if name != "fixed_core_radial":
            raise ValueError("spatial_context.name must be fixed_core_radial")
        self.period = options.pop("period", 128)
        self.inner = float(options.pop("inner_radius", 24.))
        self.outer = float(options.pop("outer_radius", 60.))
        self.displacement = float(options.pop("max_displacement", 6.))
        self.phases = tuple(options.pop("phases", [0, 1, 2, 3]))
        self.seed = options.pop("seed", 420104)
        if options:
            raise ValueError(f"Unknown spatial_context options: {sorted(options)}")
        if type(self.period) is not int or self.period <= 0 or type(self.seed) is not int:
            raise ValueError("Spatial period must be a positive integer and seed an integer")
        if not all(math.isfinite(v) for v in (self.inner, self.outer, self.displacement)):
            raise ValueError("Spatial radii and displacement must be finite")
        if not 0 <= self.inner < self.outer < self.period / 2 or self.displacement < 0:
            raise ValueError("Require 0 <= inner_radius < outer_radius < period/2 and displacement >= 0")
        if self.displacement * math.pi >= self.outer - self.inner:
            raise ValueError("Spatial displacement would permit a folded radial map")
        if not self.phases or len(set(self.phases)) != len(self.phases) or any(
            type(p) is not int or p not in range(4) for p in self.phases
        ):
            raise ValueError("Spatial phases must be a nonempty unique subset of [0,1,2,3]")
        self.generator = torch.Generator().manual_seed(self.seed)
        self._cache = {}

    def __call__(self, strong, probabilities):
        if strong.shape[0] != probabilities.shape[0] or strong.shape[-2:] != probabilities.shape[-2:]:
            raise ValueError("Strong image and teacher probability coordinates must match")
        phase = self.phases[int(torch.randint(len(self.phases), (1,), generator=self.generator))]
        sign = (-1, 1)[int(torch.randint(2, (1,), generator=self.generator))]
        key = (*strong.shape[-2:], strong.device)
        if key not in self._cache:
            self._cache[key] = {(p, s): context_grids(*strong.shape[-2:], p,
                s * self.displacement, self.inner, self.outer, self.period, strong.device)
                for p in self.phases for s in (-1, 1)}
        grid, _, _, valid = self._cache[key][phase, sign]
        return resample(strong, grid), resample(probabilities, grid), valid, phase, sign

    def warp_like(self, value, phase, sign):
        """Align an auxiliary map with an already sampled strong-view grid."""
        key = (*value.shape[-2:], value.device)
        if key not in self._cache:
            raise RuntimeError("Sample the spatial view before warping auxiliary maps")
        return resample(value, self._cache[key][phase, sign][0])

    def valid_like(self, value, phase, sign):
        """Validity of the already sampled pullback, without sampling new geometry."""
        key = (*value.shape[-2:], value.device)
        if key not in self._cache:
            raise RuntimeError("Sample the spatial view before reading its validity")
        return self._cache[key][phase, sign][3]
