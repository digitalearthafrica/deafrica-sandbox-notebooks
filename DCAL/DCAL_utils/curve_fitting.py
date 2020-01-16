import numpy as np
from numpy import fft
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.ndimage.filters import gaussian_filter1d

from scale import np_scale
from plotter_utils_consts import n_pts_smooth, default_fourier_n_harm


def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gaussian_fit(x, y, x_smooth=None, n_pts=n_pts_smooth):
    """
    Fits a Gaussian to some data - x and y. Returns predicted interpolation values.

    Parameters
    ----------
    x: list-like
        The x values of the data to fit to. Must have range [0,1].
    y: list-like
        The y values of the data to fit to.
    x_smooth: list-like
        The exact x values to interpolate for. Supercedes `n_pts`.
    n_pts: int
        The number of evenly spaced points spanning the range of `x` to interpolate for.

    Returns
    -------
    x_smooth, y_smooth: numpy.ndarray
        The smoothed x and y values of the curve fit.
    """
    if x_smooth is None:
        x_smooth_inds = np.linspace(0, len(x), n_pts)
        x_smooth = np.interp(x_smooth_inds, np.arange(len(x)), x)
    mean, sigma = np.nanmean(y), np.nanstd(y)
    popt, pcov = curve_fit(gauss, np_scale(x), y, p0=[1, mean, sigma],
                           maxfev=np.iinfo(np.int32).max)
    y_smooth = gauss(np_scale(x_smooth), *popt)
    return x_smooth, y_smooth


def gaussian_filter_fit(x, y, x_smooth=None, n_pts=n_pts_smooth, sigma=None):
    """
    Fits a Gaussian filter to some data - x and y. Returns predicted interpolation values.
    Currently, smoothing is achieved by fitting a cubic spline to the gaussian filter fit
    of `x` and `y`.

    Parameters
    ----------
    x: list-like
        The x values of the data to fit to.
    y: list-like
        The y values of the data to fit to.
    x_smooth: list-like, optional
        The exact x values to interpolate for. Supercedes `n_pts`.
    n_pts: int, optional
        The number of evenly spaced points spanning the range of `x` to interpolate for.
    sigma: numeric, optional
        The standard deviation of the Gaussian kernel. A larger value yields a smoother curve,
        but also reduced the closeness of the fit. By default, it is `4 * np.std(y)`.

    Returns
    -------
    x_smooth, y_smooth: numpy.ndarray
        The smoothed x and y values of the curve fit.
    """
    if x_smooth is None:
        x_smooth_inds = np.linspace(0, len(x)-1, n_pts)
        x_smooth = np.interp(x_smooth_inds, np.arange(len(x)), x)
    sigma = sigma if sigma is not None else 4 * np.std(y)
    gauss_filter_y = gaussian_filter1d(y, sigma)
    cs = CubicSpline(x, gauss_filter_y)
    y_smooth = cs(x_smooth)
    return x_smooth, y_smooth


def poly_fit(x, y, degree, x_smooth=None, n_pts=n_pts_smooth):
    """
    Fits a polynomial of any positive integer degree to some data - x and y. Returns predicted interpolation values.

    Parameters
    ----------
    x: list-like
        The x values of the data to fit to.
    y: list-like
        The y values of the data to fit to.
    x_smooth: list-like
        The exact x values to interpolate for. Supercedes `n_pts`.
    n_pts: int
        The number of evenly spaced points spanning the range of `x` to interpolate for.
    degree: int
        The degree of the polynomial to fit.

    Returns
    -------
    x_smooth, y_smooth: numpy.ndarray
        The smoothed x and y values of the curve fit.
    """
    if x_smooth is None:
        x_smooth_inds = np.linspace(0, len(x), n_pts)
        x_smooth = np.interp(x_smooth_inds, np.arange(len(x)), x)
    y_smooth = np.array([np.array([coef * (x_val ** current_degree) for
                                   coef, current_degree in zip(np.polyfit(x, y, degree),
                                                               range(degree, -1, -1))]).sum() for x_val in x_smooth])
    return x_smooth, y_smooth


def fourier_fit(x, y, n_predict=0, x_smooth=None, n_pts=n_pts_smooth,
                n_harm=default_fourier_n_harm):
    """
    Creates a Fourier fit of a NumPy array. Also supports extrapolation.
    Credit goes to https://gist.github.com/tartakynov/83f3cd8f44208a1856ce.

    Parameters
    ----------
    x, y: numpy.ndarray
        1D NumPy arrays of the x and y values to fit to.
        Must not contain NaNs.
    n_predict: int
        The number of points to extrapolate.
        The points will be spaced evenly by the mean spacing of values in `x`.
    x_smooth: list-like, optional
        The exact x values to interpolate for. Supercedes `n_pts`.
    n_pts: int, optional
        The number of evenly spaced points spanning the range of `x` to interpolate for.
    n_harm: int
        The number of harmonics to use. A higher value yields a closer fit.

    Returns
    -------
    x_smooth, y_smooth: numpy.ndarray
        The smoothed x and y values of the curve fit.
    """
    if x_smooth is None:
        x_smooth_inds = np.linspace(0, len(x), n_pts)
        x_smooth = np.interp(x_smooth_inds, np.arange(len(x)), x)
    n_predict_smooth = int((len(x_smooth) / len(x)) * n_predict)
    # These points are evenly spaced for the fourier fit implementation we use.
    # More points are selected than are in `x_smooth` so we can interpolate accurately.
    fourier_mult_pts = 2
    x_smooth_fourier = np.linspace(x_smooth.min(), x_smooth.max(),
                                   fourier_mult_pts * len(x_smooth))
    y_smooth_fourier = np.interp(x_smooth_fourier, x, y)
    n_predict_smooth_fourier = int((len(x_smooth_fourier) / len(x)) * n_predict)

    # Perform the Fourier fit and extrapolation.
    n = y_smooth_fourier.size
    t = np.arange(0, n)
    p = np.polyfit(t, y_smooth_fourier, 1)  # find linear trend in arr
    x_notrend = y_smooth_fourier - p[0] * t  # detrended arr
    x_freqdom = fft.fft(x_notrend)  # detrended arr in frequency domain
    f = fft.fftfreq(n)  # frequencies
    # sort indexes by frequency, lower -> higher
    indexes = list(range(n))
    indexes.sort(key=lambda i: np.absolute(x_freqdom[i]))
    indexes.reverse()
    t = np.arange(0, n + n_predict_smooth_fourier)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n  # amplitude
        phase = np.angle(x_freqdom[i])  # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    y_smooth_fourier = restored_sig + p[0] * t

    # Find the points in `x_smooth_fourier` that are near to points in `x_smooth`
    # and then interpolate the y values to match the new x values.
    x_smooth = x_smooth_fourier[np.searchsorted(x_smooth_fourier, x_smooth)]
    # Ensure `x_smooth` includes the extrapolations.
    mean_x_smooth_space = np.diff(x_smooth).mean()
    x_predict_smooth = np.linspace(x_smooth[-1] + mean_x_smooth_space,
                                   x_smooth[-1] + mean_x_smooth_space * n_predict_smooth,
                                   n_predict_smooth)
    x_smooth = np.concatenate((x_smooth, x_predict_smooth))
    # Ensure `x_smooth_fourier` includes the extrapolations.
    mean_x_smooth_fourier_space = np.diff(x_smooth).mean()
    x_predict_smooth_fourier = \
        np.linspace(
            x_smooth_fourier[-1] + mean_x_smooth_fourier_space,
            x_smooth_fourier[-1] + mean_x_smooth_fourier_space * n_predict_smooth_fourier,
            n_predict_smooth_fourier)
    x_smooth_fourier = np.concatenate((x_smooth_fourier, x_predict_smooth_fourier))
    y_smooth = np.interp(x_smooth, x_smooth_fourier, y_smooth_fourier)
    return x_smooth, y_smooth