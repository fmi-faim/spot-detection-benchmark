from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.morphology import ball, disk, h_maxima
from skimage.util import img_as_float32

FWHM_TO_SIGMA = 2 * np.sqrt(2 * np.log(2))  # ~2.355


def compute_lateral_sigma(
    wavelength: float,
    NA: float,
    lateral_spacing: float,
) -> float:
    """Theoretical YX sigma (in pixels) of a diffraction limited spot.

    Uses Abbe's lateral resolution limit d = wavelength / (2 * NA) as the
    full width at half maximum of the spot.

    Parameters
    ----------
    wavelength :
        Emission wavelength.
    NA :
        Numerical aperture.
    lateral_spacing :
        Pixel size in YX, same unit as `wavelength`.

    Returns
    -------
    Sigma in pixels.
    """
    return wavelength / (2 * NA) / FWHM_TO_SIGMA / lateral_spacing


def compute_axial_sigma(
    wavelength: float,
    NA: float,
    axial_spacing: float,
    refractive_index: float = 1.33,
) -> float:
    """Theoretical Z sigma (in planes) of a diffraction limited spot.

    Uses the widefield axial resolution d_z = 2 * wavelength * n / NA**2 as the
    full width at half maximum of the spot (a confocal PSF is ~1.4x narrower).

    Parameters
    ----------
    wavelength :
        Emission wavelength.
    NA :
        Numerical aperture.
    axial_spacing :
        Z step between planes, same unit as `wavelength`.
    refractive_index :
        Refractive index of the immersion medium (1.0 air, 1.33 water, 1.515 oil).

    Returns
    -------
    Sigma in planes.
    """
    return 2 * wavelength * refractive_index / NA**2 / FWHM_TO_SIGMA / axial_spacing


@lru_cache(maxsize=None)  # depends on the sigmas only: computed once per sigmas
def estimate_log_rescale_factor(sigmas: tuple[float, ...]) -> float:
    """Rescale factor for a LoG filter response.

    After rescaling, the LoG response at the centre of a Gaussian spot with
    the given sigmas equals the spot amplitude in the input image.

    Parameters
    ----------
    sigmas :
        Sigmas used in the LoG filter, one per image axis.

    Returns
    -------
    Rescale factor.
    """
    size = int(max(sigmas) * 7)
    ndim = len(sigmas)

    spot = np.zeros((size,) * ndim, dtype=np.float32)
    spot[(size // 2,) * ndim] = 1
    spot = gaussian_filter(spot, sigmas)
    spot /= spot.max()

    return 1 / (-gaussian_laplace(spot, sigma=sigmas)).max()


def detect_spots(
    img: np.ndarray,
    axial_sigma: float,
    lateral_sigma: float,
    h: int,
    background_img: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Detect diffraction limited spots in 3D.

    The spot detection uses a Laplacian of Gaussian filter to detect
    spots of a given size. These detections are intensity filtered with
    a h-maxima filter.

    Parameters
    ----------
    img :
        Image containing spot signal, axes (Z, Y, X).
    axial_sigma :
        Z extension of the spots, in pixels.
    lateral_sigma :
        YX extension of the spots, in pixels.
    h :
        h-maxima threshold, in the intensity unit of `img`.
    background_img :
        Estimated background image. This is subtracted before the
        spot detection.
    mask :
        Foreground mask to restrict the spot detection.

    Returns
    -------
    Detected spots, shape (N, 3), columns (z, y, x).
    """
    image = (
        img_as_float32(img) - img_as_float32(background_img)
        if background_img is not None
        else img_as_float32(img)
    )

    rescale_factor = estimate_log_rescale_factor(
        sigmas=(axial_sigma, lateral_sigma, lateral_sigma)
    )
    log_img = (
        -gaussian_laplace(image, sigma=(axial_sigma, lateral_sigma, lateral_sigma))
        * rescale_factor
    )

    h_ = img_as_float32(np.array(h, dtype=img.dtype))
    h_detections = h_maxima(log_img, h=h_, footprint=ball(1))
    if mask is not None:
        h_detections = h_detections * mask
    return np.array(np.where(h_detections)).T


def detect_spots_2d(
    img: np.ndarray,
    lateral_sigma: float,
    h: int,
    background_img: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Detect diffraction limited spots in 2D.

    The spot detection uses a Laplacian of Gaussian filter to detect
    spots of a given size. These detections are intensity filtered with
    a h-maxima filter.

    Parameters
    ----------
    img :
        Image containing spot signal, axes (Y, X).
    lateral_sigma :
        YX extension of the spots, in pixels.
    h :
        h-maxima threshold, in the intensity unit of `img`.
    background_img :
        Estimated background image. This is subtracted before the
        spot detection.
    mask :
        Foreground mask to restrict the spot detection.

    Returns
    -------
    Detected spots, shape (N, 2), columns (y, x).
    """
    image = (
        img_as_float32(img) - img_as_float32(background_img)
        if background_img is not None
        else img_as_float32(img)
    )

    rescale_factor = estimate_log_rescale_factor(sigmas=(lateral_sigma, lateral_sigma))
    log_img = (
        -gaussian_laplace(image, sigma=(lateral_sigma, lateral_sigma)) * rescale_factor
    )

    h_ = img_as_float32(np.array(h, dtype=img.dtype))
    h_detections = h_maxima(log_img, h=h_, footprint=disk(1))
    if mask is not None:
        h_detections = h_detections * mask
    return np.array(np.where(h_detections)).T
