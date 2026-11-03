from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import yaml

from spot_detection_benchmark.utils import MODES, REPO_ROOT, get_device, set_piscis_dir

log = logging.getLogger(__name__)

with open(REPO_ROOT / "models.yaml", encoding="utf-8") as f:
    MODELS = yaml.safe_load(f)


def _log_h(threshold, image):
    h = int(round(threshold))
    if np.issubdtype(image.dtype, np.integer):
        h = min(h, np.iinfo(image.dtype).max)
    return h


def _sample(arr, coords):
    if not len(coords):
        return np.array([])
    idx = tuple(
        np.clip(np.rint(coords[:, k]).astype(int), 0, arr.shape[k] - 1)
        for k in range(coords.shape[1])
    )
    return np.asarray(arr)[idx]


def _piscis_peaks(coords, deltas, radius=2, tol=1e-3):
    coords = np.asarray(coords, float)
    if not len(coords):
        return np.empty(coords.shape, int), np.empty(0, bool)
    yx = coords[:, -2:]
    steps = np.arange(-radius, radius + 1)
    offsets = np.stack(np.meshgrid(steps, steps, indexing="ij"), -1).reshape(-1, 2)
    candidates = np.rint(yx).astype(int)[:, None, :] + offsets[None]
    candidates = np.clip(candidates, 0, np.array(deltas.shape[-2:]) - 1)
    y, x = candidates[..., 0], candidates[..., 1]
    if coords.shape[1] == 3:
        z = np.rint(coords[:, :1]).astype(int)
        shift = deltas[z, :, y, x]  # (N, K, 2)
    else:
        shift = np.moveaxis(deltas[:, y, x], 0, -1)  # (2, N, K) -> (N, K, 2)
    miss = np.abs(candidates + shift - yx[:, None, :]).max(axis=-1)
    best = miss.argmin(axis=1)
    peaks = candidates[np.arange(len(coords)), best]
    found = miss[np.arange(len(coords)), best] < tol
    if coords.shape[1] == 3:
        peaks = np.hstack((np.rint(coords[:, :1]).astype(int), peaks))
    return peaks, found


def _piscis_scores(coords, labels, deltas):
    peaks, found = _piscis_peaks(coords, deltas)
    scores = labels[tuple(peaks.T)].astype(float)
    scores[~found] = np.nan
    if not found.all():
        log.debug(f"[piscis] {np.sum(~found)} spot(s) on the tile padding: no score")
    return scores


def _predict_stack(detector, stack, threshold):
    coords = []
    scores = []
    for i, image in enumerate(stack):
        c, s = detector.predict_2d(image, threshold)
        c = np.hstack((np.full((len(c), 1), i), c))
        coords.append(c)
        scores.append(s)

    if detector.score_name is None:
        scores = None
    if not coords:
        return np.empty((0, 3)), (None if scores is None else np.empty((0,)))
    return np.vstack(coords), (None if scores is None else np.hstack(scores))


def _predict_planes_and_merge(detector, volume, threshold, merge_params=None):
    from spot_detection_benchmark.merge_z import merge_z

    coords, scores = _predict_stack(detector, volume, threshold)
    return merge_z(volume, coords, scores, **(merge_params or {}))


class LoG:
    score_name = None
    default_threshold = 20
    needs_model = False
    filter_modes = ()

    def __init__(
        self,
        lateral_sigma=1.5,
        axial_sigma=None,
        voxel_size=(1.0, 1.0, 1.0),
    ):
        self.lateral_sigma = lateral_sigma
        self.voxel_size = voxel_size
        self.axial_sigma = (
            lateral_sigma * voxel_size[-1] / voxel_size[0]
            if axial_sigma is None
            else axial_sigma
        )

    def predict_2d(self, image, threshold):
        from spot_detection_benchmark.log_detection import detect_spots_2d

        coords = detect_spots_2d(
            image, lateral_sigma=self.lateral_sigma, h=_log_h(threshold, image)
        )
        return coords, None

    def predict_3d(self, volume, threshold):
        from spot_detection_benchmark.log_detection import detect_spots

        coords = detect_spots(
            volume,
            axial_sigma=self.axial_sigma,
            lateral_sigma=self.lateral_sigma,
            h=_log_h(threshold, volume),
        )
        return coords, None


class DeepBlink:
    score_name = "probability"
    default_threshold = 0.5
    filter_modes = ("2d", "2d_stack")

    def __init__(self, model_path, merge_params=None):
        from deepblink.io import load_model

        self.model = load_model(str(model_path))
        self.merge_params = merge_params

    def predict_2d(self, image, threshold):
        from deepblink.inference import predict

        coords = predict(image=image, model=self.model, probability=threshold)
        if coords.ndim == 2 and coords.shape[1] > 2:
            return coords[:, :2], coords[:, 2]
        return coords.reshape(-1, 2), None

    def predict_3d(self, volume, threshold):
        return _predict_planes_and_merge(
            self, volume, threshold, merge_params=self.merge_params
        )


class Piscis:
    score_name = "confidence"
    default_threshold = 0.5
    filter_modes = ("2d", "2d_stack")

    def __init__(self, model_path):
        set_piscis_dir()

        from piscis import Piscis as PiscisModel

        self.model = PiscisModel(model_name=model_path, device=get_device())

    def predict_2d(self, image, threshold):
        coords, maps = self.model.predict(
            image, threshold=threshold, intermediates=True
        )
        coords = np.asarray(coords).reshape(-1, 2)
        maps = np.asarray(maps)  # (labels, delta y, delta x)
        return coords, _piscis_scores(coords, maps[0], maps[1:])

    def predict_3d(self, volume, threshold):
        coords, maps = self.model.predict(
            volume, stack=True, threshold=threshold, intermediates=True
        )
        coords = np.asarray(coords).reshape(-1, 3)
        maps = np.asarray(maps)  # (z, labels / delta y / delta x, y, x)
        return coords, _piscis_scores(coords, maps[:, 0], maps[:, 1:])


class UFish:
    score_name = "laplacian"
    default_threshold = 0.5
    filter_modes = ("2d", "2d_stack", "3d")

    def __init__(self, model_path, blend_3d=True):
        from ufish.api import UFish as UFishModel

        if not str(model_path).endswith(".pth"):
            raise ValueError(f"ufish: {model_path} is not a PyTorch .pth model")
        self.blend_3d = blend_3d
        self.model = UFishModel()
        self.model.load_weights_from_path(str(model_path))
        on_gpu = self.model.cuda

        log.info(f"[ufish] model loaded on {'GPU' if on_gpu else 'CPU'}")
        if not on_gpu:
            log.warning(
                "[ufish] running on CPU, inference will be slow. On the cluster, use "
                "the ufish-gpu environment on a GPU node (--gres=gpu:1)."
            )

    def predict_2d(self, image, threshold):
        from skimage.filters import laplace

        spots_df, enhanced = self.model.predict(image, intensity_threshold=threshold)
        coords = spots_df[["axis-0", "axis-1"]].to_numpy()
        return coords, _sample(laplace(enhanced), coords)

    def predict_3d(self, volume, threshold):
        from skimage.filters import laplace

        spots_df, enhanced = self.model.predict(
            volume,
            axes="zyx",
            blend_3d=self.blend_3d,
            intensity_threshold=threshold,
        )
        coords = spots_df[["axis-0", "axis-1", "axis-2"]].to_numpy()
        return coords, _sample(laplace(enhanced), coords)


class Spotiflow:
    score_name = "probability"
    default_threshold = None
    needs_3d_model = True
    filter_modes = ("2d", "2d_stack", "3d")

    def __init__(self, model_path):
        from spotiflow.model import Spotiflow as SpotiflowModel

        self.device = get_device()
        self.model = SpotiflowModel.from_folder(model_path, map_location=self.device)
        self.default_threshold = float(np.ravel(self.model._prob_thresh)[0])

    def predict_2d(self, image, threshold):
        if self.model.config.is_3d:
            raise ValueError(
                "--mode 2d / --mode 2d_stack requires a Spotiflow model trained "
                "in 2D, but this one is 3D only. Use a 2D Spotiflow model, or "
                "--mode 3d to predict in 3D."
            )
        coords, details = self.model.predict(
            image, device=self.device, prob_thresh=threshold
        )
        return np.asarray(coords).reshape(-1, 2), getattr(details, "prob", None)

    def predict_3d(self, volume, threshold):
        if not self.model.config.is_3d:
            raise ValueError(
                "--mode 3d requires a Spotiflow model trained in 3D, but this one is "
                "2D only. Use a 3D Spotiflow model, or --mode 2d / --mode 2d_stack "
                "to predict in 2D."
            )
        coords, details = self.model.predict(
            volume, device=self.device, prob_thresh=threshold
        )
        return np.asarray(coords).reshape(-1, 3), getattr(details, "prob", None)


DETECTORS = {
    "log": LoG,
    "deepblink": DeepBlink,
    "spotiflow": Spotiflow,
    "piscis": Piscis,
    "ufish": UFish,
}

TOOLS = list(DETECTORS)


def _model_in_yaml(training_type, dim, tool):
    return (MODELS[training_type].get(dim) or {}).get(tool)


def _full_path(model_path):
    path = REPO_ROOT / model_path
    return path if path.exists() else model_path


def default_model(tool, mode, pretrained=False):
    training_type = "pretrained" if pretrained else "trained"
    searched = f"{training_type} / {'3d' if mode == '3d' else '2d'}"

    if mode != "3d":
        model = _model_in_yaml(training_type, "2d", tool)
    else:
        model = _model_in_yaml(training_type, "3d", tool)
        needs_3d = getattr(DETECTORS[tool], "needs_3d_model", False)
        if model is None and not needs_3d:
            model = _model_in_yaml(training_type, "2d", tool)
            searched += " and 2d"
        if model is None and needs_3d and training_type == "trained":
            model = _model_in_yaml("pretrained", "3d", tool)
            searched += ", then pretrained / 3d"
            if model is not None:
                warnings.warn(
                    f"{tool}: no trained 3D model in models.yaml, "
                    f"using the pretrained one: {model}"
                )

    if model is None:
        raise ValueError(
            f"{tool}: no model in models.yaml for mode {mode} (searched {searched})"
        )
    return _full_path(model)


def make_detector(tool, mode, pretrained=False, model_path=None, **kwargs):
    if tool not in DETECTORS:
        raise ValueError(f"Unknown tool: {tool}. Available tools: {TOOLS}")
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}. Available modes: {MODES}")

    model_path = resolve_model(tool, mode, pretrained, model_path)
    if model_path is None:  # a tool without model (LoG)
        return DETECTORS[tool](**kwargs)
    return DETECTORS[tool](model_path=model_path, **kwargs)


def resolve_model(tool, mode, pretrained=False, model_path=None):
    if not getattr(DETECTORS[tool], "needs_model", True):
        return None
    if model_path is None:
        model_path = default_model(tool, mode, pretrained=pretrained)
    return model_path


def detect(detector, image, mode, threshold=None):
    if threshold is None:
        threshold = detector.default_threshold

    if mode == "2d":
        coords, scores = detector.predict_2d(image, threshold)
    elif mode == "2d_stack":
        coords, scores = _predict_stack(detector, image, threshold)
    elif mode == "3d":
        coords, scores = detector.predict_3d(image, threshold)
    else:
        raise ValueError(f"Unknown mode: {mode}. Available modes: {MODES}")

    ndim = 2 if mode == "2d" else 3
    coords = np.asarray(coords).reshape(-1, ndim)
    spots = pd.DataFrame(coords, columns=[f"axis-{k}" for k in range(ndim)])
    if detector.score_name is not None:
        # an image without spot may come back without scores: keep the column
        spots[detector.score_name] = (
            np.full(len(spots), np.nan)
            if scores is None
            else np.asarray(scores).reshape(-1)
        )
    return spots
