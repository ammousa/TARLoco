#  Copyright 2025 University of Manchester, Amr Mousa
#  SPDX-License-Identifier: CC-BY-SA-4.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from exts.tarloco.utils.terrains_cfg import (
        CorridorTerrainCfg,
        RailwayTracksTerrainCfg,
    )


def random_railway_tracks_terrain(
    difficulty: float, cfg: RailwayTracksTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a random railway tracks mesh terrain."""
    gauge = np.random.uniform(*cfg.gauge_range)
    rail_height = cfg.rail_height_range[0] + difficulty * (
        cfg.rail_height_range[1] - cfg.rail_height_range[0]
    )  # the harder, the higher
    rail_width = np.random.uniform(*cfg.rail_width_range)
    rail_length = cfg.size[0]
    sleeper_spacing = np.random.uniform(*cfg.sleeper_spacing_range)
    sleeper_height = cfg.sleeper_height_range[0] + difficulty * (
        cfg.sleeper_height_range[1] - cfg.sleeper_height_range[0]
    )  # the harder, the higher
    sleeper_width = np.random.uniform(*cfg.sleeper_width_range)
    sleeper_length = gauge * 1.2

    meshes = []

    max_track_width = cfg.gauge_range[1] * 2
    tracks_no = int(cfg.size[1] / max_track_width)
    track_width = cfg.size[1] / tracks_no

    for i in range(tracks_no):
        center_y = (i + 0.5) * track_width

        # constants for terrain generation
        terrain_height = 1.0

        # generate rails
        left_rail = center_y - gauge / 2
        right_rail = center_y + gauge / 2
        rail_meshes = trimesh.creation.box(
            (rail_length, rail_width, rail_height),
            trimesh.transformations.translation_matrix((rail_length / 2, left_rail, sleeper_height + rail_height / 2)),
        )
        rail_meshes += trimesh.creation.box(
            (rail_length, rail_width, rail_height),
            trimesh.transformations.translation_matrix((rail_length / 2, right_rail, sleeper_height + rail_height / 2)),
        )
        meshes.append(rail_meshes)

        # generate sleepers
        sleepers_no = int((rail_length - sleeper_width) / sleeper_spacing) + 1
        for i in range(sleepers_no):
            pos = (
                i * sleeper_spacing + sleeper_width / 2,
                center_y,
                sleeper_height / 2,
            )
            sleeper_meshes = trimesh.creation.box(
                (sleeper_width, sleeper_length, sleeper_height),
                trimesh.transformations.translation_matrix(pos),
            )
            meshes.append(sleeper_meshes)

        # perpendicular tracks
        center_x = center_y  # assuming square terrain

        # generate rails
        left_rail = center_x - gauge / 2
        right_rail = center_x + gauge / 2
        rail_meshes = trimesh.creation.box(
            (rail_width, rail_length, rail_height),
            trimesh.transformations.translation_matrix((left_rail, rail_length / 2, sleeper_height + rail_height / 2)),
        )
        rail_meshes += trimesh.creation.box(
            (rail_width, rail_length, rail_height),
            trimesh.transformations.translation_matrix((right_rail, rail_length / 2, sleeper_height + rail_height / 2)),
        )
        meshes.append(rail_meshes)

        # generate sleepers
        sleepers_no = int((rail_length - sleeper_width) / sleeper_spacing) + 1
        for i in range(sleepers_no):
            pos = (
                center_x,
                i * sleeper_spacing + sleeper_width / 2,
                sleeper_height / 2,
            )
            sleeper_meshes = trimesh.creation.box(
                (sleeper_length, sleeper_width, sleeper_height),
                trimesh.transformations.translation_matrix(pos),
            )
            meshes.append(sleeper_meshes)

    # generate the ground
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    ground_meshes = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes.append(ground_meshes)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], 0.0])

    return meshes, origin


def corridor_terrain(difficulty: float, cfg: CorridorTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a corridor mesh terrain."""
    width = cfg.width_range[1] - difficulty * (cfg.width_range[1] - cfg.width_range[0])
    height = cfg.height

    meshes = []

    center_y = cfg.size[1] / 2
    wall_thickness = 0.1

    # constants for terrain generation
    terrain_height = 1.0

    # generate the corridor
    left_wall = center_y - width / 2 - wall_thickness / 2
    right_wall = center_y + width / 2 + wall_thickness / 2
    corridor_meshes = trimesh.creation.box(
        (cfg.size[0], wall_thickness, height),
        trimesh.transformations.translation_matrix((cfg.size[0] / 2, left_wall, height / 2)),
    )
    corridor_meshes += trimesh.creation.box(
        (cfg.size[0], wall_thickness, height),
        trimesh.transformations.translation_matrix((cfg.size[0] / 2, right_wall, height / 2)),
    )
    meshes.append(corridor_meshes)

    # generate the ground
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
    ground_meshes = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes.append(ground_meshes)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], 0.0])

    return meshes, origin
