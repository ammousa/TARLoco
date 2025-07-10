# Copyright (c) 2025, Amr Mousa, University of Manchester
# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# This file is based on code from the isaaclab repository:
# https://github.com/isaac-sim/IsaacLab/
#
# The original code is licensed under the BSD 3-Clause License.
# See the `licenses/` directory for details.
#
# This version includes significant modifications by Amr Mousa (2025).

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .terrains import corridor_terrain, random_railway_tracks_terrain


@configclass
class RailwayTracksTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a railway tracks mesh terrain."""

    function = random_railway_tracks_terrain

    """The gauge (separation) of the railway tracks (in m)."""
    gauge_range: tuple[float, float] = MISSING

    """The height of the rail tracks (in m). Rails are placed on top of the sleepers,
    so the total height above the ground is rail_height + sleeper_height."""
    rail_height_range: tuple[float, float] = MISSING

    """The width of the single rail (in m)."""
    rail_width_range: tuple[float, float] = MISSING

    """The distance between the sleepers (in m) from center to center."""
    sleeper_spacing_range: tuple[float, float] = MISSING

    """The height of the sleepers (in m)."""
    sleeper_height_range: tuple[float, float] = MISSING

    """The width of the sleepers (in m) (direction parallel to rails)."""
    sleeper_width_range: tuple[float, float] = MISSING


@configclass
class CorridorTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a corridor mesh terrain."""

    function = corridor_terrain

    """The width of the corridor (in m)."""
    width_range: tuple[float, float] = MISSING

    """The height of the corridor walls (in m)."""
    height: float = MISSING


ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.5, 0.9),
    use_cache=True,
    sub_terrains={  # proportions are later normalized
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.18),
            step_width=0.5,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.18),
            step_width=0.5,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1,
            grid_width=0.45,
            grid_height_range=(0.025, 0.1),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "random_tracks": RailwayTracksTerrainCfg(
            proportion=0.5,
            gauge_range=(0.60, 1.80),
            rail_height_range=(0.10, 0.20),
            rail_width_range=(0.04, 0.09),
            sleeper_spacing_range=(0.50, 0.70),
            sleeper_height_range=(0.0, 0.05),
            sleeper_width_range=(0.20, 0.30),
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
