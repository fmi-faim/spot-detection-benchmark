from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def merge_z(
    volume,
    coords,
    scores,
    search_range=2,
    max_gap=0,
    min_planes=2,
    mass_radius=3,
):
    """(plane, y, x) detections -> one (z, y, x) per spot (the notebook's df_clean).

    Args:
        volume: the (z, y, x) image, used to measure each detection's mass.
        coords: (N, 3) detections, columns (plane, y, x).
        scores: (N,) score of each detection, or None.
        search_range: max (y, x) distance in px between two planes of a spot
            (notebook: search_range).
        max_gap: number of missing planes allowed inside a spot (notebook:
            gap_frames, trackpy's memory).
        min_planes: tracks seen on fewer planes are not merged (notebook:
            min_frames).
        mass_radius: radius in px of trackpy.refine_com (notebook:
            radius_refinement).

    Returns:
        (coords (M, 3), scores (M,) or None): each kept detection with its own
        score.
    """
    import trackpy as tp

    tp.quiet()
    coords = np.asarray(coords, float).reshape(-1, 3)
    if not len(coords):
        return np.empty((0, 3)), (None if scores is None else np.empty((0,)))
    planes = np.rint(coords[:, 0]).astype(int)

    pad_width = mass_radius
    frames = []
    for slice_ in np.unique(planes):
        rows = np.flatnonzero(planes == slice_)
        yx = coords[rows, 1:3]
        y, x = yx.T.copy()

        yx = yx + pad_width
        image_curr = np.pad(
            volume[slice_], pad_width=pad_width, mode="constant", constant_values=0
        )

        df_curr = tp.refine_com(
            raw_image=image_curr, image=image_curr, radius=mass_radius, coords=yx
        )
        df_curr["x"] = x
        df_curr["y"] = y
        df_curr["slice"] = slice_
        df_curr["row"] = rows
        frames.append(df_curr)
    df = pd.concat(frames, ignore_index=True)

    track = tp.link(
        df.rename({"slice": "frame"}, axis=1),
        search_range=search_range,
        memory=max_gap,
    )
    track = tp.filter_stubs(track, threshold=min_planes).rename(
        {"frame": "slice"}, axis=1
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        idx = track.groupby(["particle"])["mass"].transform(max) == track["mass"]
    df_nms = track[idx]

    df_without_track = df[
        ~df.set_index(["x", "y", "slice", "mass"]).index.isin(
            track.set_index(["x", "y", "slice", "mass"]).index
        )
    ]
    df_clean = pd.concat([df_nms, df_without_track]).reset_index(drop=True)

    out = df_clean[["slice", "y", "x"]].to_numpy(float)
    if scores is None:
        return out, None
    return out, np.asarray(scores, float)[df_clean["row"].to_numpy(int)]
