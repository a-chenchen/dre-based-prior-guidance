from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
from scipy import fftpack
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sbi.utils.metrics import c2st

compute_c2st = c2st

def _linear_binning(samples: np.ndarray, grid_points: np.ndarray):
    """Fast computation of histogram counts using a linearly spaced grid.

    Parameters
    ----------
    samples : np.ndarray
        The samples to be binned.
    grid_points : np.ndarray
        The grid points represent the bin centers. The grid points need to be
        linearly spaced (no check is performed to ensure that).

    Returns
    -------
    counts : np.ndarray
        Number of samples in each bin.
    """
    samples = samples[
        np.logical_and(samples >= grid_points[0], samples <= grid_points[-1])
    ]
    dx = grid_points[1] - grid_points[0]
    idx = np.floor((samples - (grid_points[0] - 0.5 * dx)) / dx)
    u, u_counts = np.unique(idx, return_counts=True)
    counts = np.zeros(len(grid_points))
    counts[u.astype(int)] = u_counts

    return counts


def _fixed_point(t: float, N: int, irange_squared: np.ndarray, a2: np.ndarray):
    """Compute the fixed point according to Botev et al. (2010).

    This function implements the function t-zeta*gamma^[l](t). Based on an
    implementation by Daniel B. Smith:
    https://github.com/Daniel-B-Smith/KDE-for-SciPy/blob/master/kde.py
    Note that the factor of 2.0 in the definition of f is correct. See longer
    discussion here: https://github.com/tommyod/KDEpy/issues/95
    """
    irange_squared = np.asarray(irange_squared, dtype=np.float64)
    a2 = np.asarray(a2, dtype=np.float64)
    ell = 7
    f = (
        2.0
        * np.pi ** (2 * ell)
        * np.sum(
            np.power(irange_squared, ell)
            * a2
            * np.exp(-irange_squared * np.pi**2.0 * t)
        )
    )

    if f <= 0:
        return -1

    for s in reversed(range(2, ell)):
        odd_numbers_prod = np.prod(np.arange(1, 2 * s + 1, 2, dtype=np.float64))
        K0 = odd_numbers_prod / np.sqrt(2.0 * np.pi)
        const = (1.0 + (1.0 / 2.0) ** (s + 1.0 / 2.0)) / 3.0
        time = np.power((2 * const * K0 / (N * f)), (2.0 / (3.0 + 2.0 * s)))
        f = (
            2.0
            * np.pi ** (2.0 * s)
            * np.sum(
                np.power(irange_squared, s)
                * a2
                * np.exp(-irange_squared * np.pi**2.0 * time)
            )
        )

    t_opt = np.power(2.0 * N * np.sqrt(np.pi) * f, -2.0 / 5.0)

    return t - t_opt


def _root(function: callable, N: int, args: tuple):
    """Try to find the smallest root whenever there is more than one.

    Root finding algorithm based on the MATLAB code by Botev et al. (2010).
    """
    N = max(min(1050.0, N), 50.0)
    tol = 1e-12 + 0.01 * (N - 50.0) / 1000.0
    converged = False
    while not converged:
        try:
            x, res = brentq(function, 0, tol, args=args, full_output=True, disp=False)
            converged = bool(res.converged)
        except ValueError:
            x = 0.0
            tol *= 2.0
            converged = False
        if x <= 0.0:
            converged = False
        if tol >= 1:
            return None

    if x <= 0.0:
        return None
    return x


def _scottrule1d(samples: np.ndarray):
    """Compute the kernel bandwidth according to Scott's rule for 1D samples.

    Parameters
    ----------
    samples : np.ndarray
        The 1D samples for which Scott's rule is being computed.

    Returns
    -------
    bandwidth : float
        Scott's bandwidth.
    """
    sigma = np.std(samples, ddof=1)
    sigma_iqr = (
        np.quantile(samples, q=0.75) - np.quantile(samples, q=0.25)
    ) / 1.3489795003921634
    sigma = min(sigma, sigma_iqr)
    return sigma * np.power(len(samples), -1.0 / 5.0)


def _validate_kde1d_args(n, lower_bound, upper_bound):
    """
    _validate_kde1d_args and raise value exception
    """
    if n <= 0:
        raise ValueError("n cannot be <= 0")

    if lower_bound is not None and upper_bound is not None:
        if lower_bound > upper_bound:
            raise ValueError("lower_bound cannot be > upper_bound")


def kde1d(
    samples: np.ndarray,
    n: int = 2**14,
    lower_bound: float = None,
    upper_bound: float = None,
):
    r"""Reliable and extremely fast kernel density estimator for 1D data.

    One-dimensional kernel density estimator based on fast Fourier transform.
    A Gaussian kernel is assumed and the bandwidth is chosen automatically
    using the technique developed by Botev et al. (2010) [1]_.

    Parameters
    ----------
    samples : np.ndarray
        The samples from which the density estimate is computed.
    n : int, optional
        The number of mesh points used in the uniform discretization of the
        interval [lower_bound, upper_bound]; n has to be a power of two;
        if n is not a power of two, it is rounded up to the next power of two,
        i.e., n is set to n=2^ceil(log2(n)), by default 2**14.
    lower_bound : float, optional
        The lower bound of the interval in which the density is being computed,
        if not given the default value is lower_bound=min(samples)-range/10,
        where range=max(samples)-min(samples), by default None.
    upper_bound : float, optional
        The upper bound of the interval in which the density is being computed,
        if not given the default value is upper_bound=max(data)+Range/10,
        where range=max(samples)-min(samples), by default None.

    Returns
    -------
    density : np.ndarray
        1D vector of length n with the values of the kernel density estimate
        at the grid points.
    xmesh : np.ndarray
        1D vector of grid over which the density estimate is computed.
    bandwidth : np.ndarray
        The optimal bandwidth (Gaussian kernel assumed).

    Notes
    -----
    This implementation is based on the MATLAB implementation by Zdravko Botev,
    and was further inspired by the Python implementations by Daniel B. Smith
    and the bandwidth selection code in KDEpy [2]_. We thank Zdravko Botev for
    useful clarifications on the implementation of the fixed_point function.

    Unlike other implementations, this one is immune to problems caused by
    multimodal densities with widely separated modes (see example). The
    bandwidth estimation does not deteriorate for multimodal densities because
    a parametric model is never assumed for the data.

    References
    ----------
    .. [1] Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
       estimation via diffusion. The Annals of Statistics,
       38(5):2916-2957, 2010.
    .. [2] https://github.com/tommyod/KDEpy/blob/master/KDEpy/bw_selection.py

    Examples
    --------

    .. code-block:: python

        import numpy as np
        from numpy.random import randn

        samples = np.concatenate(
            (randn(100, 1), randn(100, 1) * 2 + 35, randn(100, 1) + 55)
        )
        kde1d(samples, 2 ** 14, min(samples) - 5, max(samples) + 5)

    """
    samples = samples.ravel()  # make samples a 1D array

    # validate values passed to the function
    _validate_kde1d_args(n, lower_bound, upper_bound)

    n = int(2 ** np.ceil(np.log2(n)))  # round up to the next power of 2
    if lower_bound is None or upper_bound is None:
        minimum = np.min(samples)
        maximum = np.max(samples)
        delta = maximum - minimum
        if lower_bound is None:
            lower_bound = np.array([minimum - 0.1 * delta])
        if upper_bound is None:
            upper_bound = np.array([maximum + 0.1 * delta])

    delta = upper_bound - lower_bound
    xmesh = np.linspace(lower_bound, upper_bound, n)
    N = len(np.unique(samples))

    initial_data = _linear_binning(samples, xmesh)
    initial_data = initial_data / np.sum(initial_data)

    # Compute the Discrete Cosine Transform (DCT) of the data
    a = fftpack.dct(initial_data, type=2)

    # Compute the bandwidth
    irange_squared = np.arange(1, n, dtype=np.float64) ** 2.0
    a2 = a[1:] ** 2.0 / 4.0
    t_star = _root(_fixed_point, N, args=(N, irange_squared, a2))

    if t_star is None:
        # Automated bandwidth selection failed, use Scott's rule
        bandwidth = _scottrule1d(samples)
        t_star = (bandwidth / delta) ** 2.0
    else:
        bandwidth = np.sqrt(t_star) * delta

    # Smooth the discrete cosine transform of initial data using t_star
    a_t = a * np.exp(-np.arange(n, dtype=float) ** 2 * np.pi**2.0 * t_star / 2.0)

    # Diving by 2 because of the implementation of fftpack.idct
    density = fftpack.idct(a_t) / (2.0 * delta)
    density[density < 0] = 0.0  # remove negatives due to round-off error

    return density.ravel(), xmesh.ravel(), bandwidth


def kldiv_mvn(mu1, sigma1, mu2, sigma2):
    """
    Compute the analytical Kullback-Leibler divergence between two multivariate
    normal pdfs.

    Parameters
    ----------
    mu1 : np.ndarray
        The k-dimensional mean vector of the first multivariate normal pdf.
    sigma1 : np.ndarray
        The covariance matrix of the first multivariate normal pdf.
    mu2 : np.ndarray
        The k-dimensional mean vector of the second multivariate normal pdf.
    sigma2 : np.ndarray
        The covariance matrix of the second multivariate normal pdf.

    Returns
    -------
    kldiv : np.array
        The computed Kullback-Leibler divergence.
    """
    D = mu1.size
    mu1 = mu1.reshape(-1, 1)
    mu2 = mu2.reshape(-1, 1)
    dmu = mu2 - mu1
    detq1 = np.linalg.det(sigma1)
    detq2 = np.linalg.det(sigma2)
    lndet = np.log(detq2 / detq1)
    a, _, _, _ = np.linalg.lstsq(sigma2, sigma1, rcond=None)
    b, _, _, _ = np.linalg.lstsq(sigma2, dmu, rcond=None)
    kl1 = 0.5 * (np.trace(a) + dmu.T @ b - D + lndet)
    a, _, _, _ = np.linalg.lstsq(sigma1, sigma2, rcond=None)
    b, _, _, _ = np.linalg.lstsq(sigma1, dmu, rcond=None)
    kl2 = 0.5 * (np.trace(a) + dmu.T @ b - D - lndet)
    return np.concatenate((kl1, kl2), axis=None)


def compute_rmse(
    ground_truth: np.ndarray,
    samples: np.ndarray,
) -> torch.Tensor:
    """
    ground_truth: (num_features)
    samples: (num_samples, num_features)
    """
    ground_truth = np.array(ground_truth)
    samples = np.array(samples)
    ground_truth = np.expand_dims(ground_truth, 0).repeat(samples.shape[0], axis=0)
    rmse = np.sqrt(np.mean(np.square(ground_truth - samples)))
    return rmse


def kernel_matrix(X, Y, lengthscale):
    """Compute the RBF kernel matrix between X and Y.

    Args:
        X: First set of samples
        Y: Second set of samples
        lengthscale: RBF kernel lengthscale

    Returns:
        Kernel matrix
    """
    X_sqnorms = np.sum(np.square(X), axis=1)
    Y_sqnorms = np.sum(np.square(Y), axis=1)
    XY = np.dot(X, Y.transpose())
    sqD = -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]
    return np.exp(-sqD / (2 * lengthscale**2))


def compute_mmd_unweighted(x, y, lengthscale):
    """Approximates the squared MMD between samples x_i ~ P and y_i ~ Q

    Args:
        x: Samples from distribution P
        y: Samples from distribution Q
        lengthscale: RBF kernel lengthscale

    Returns:
        Maximum Mean Discrepancy value
    """
    if len(x.shape) == 1:
        x = np.array(x, ndmin=2).transpose()
        y = np.array(y, ndmin=2).transpose()

    m = x.shape[0]
    n = y.shape[0]

    z = np.concatenate((x, y), axis=0)

    K = kernel_matrix(z, z, lengthscale)

    kxx = K[0:m, 0:m]
    kyy = K[m : (m + n), m : (m + n)]
    kxy = K[0:m, m : (m + n)]

    return (
        (1 / m**2) * np.sum(kxx)
        - (2 / (m * n)) * np.sum(kxy)
        + (1 / n**2) * np.sum(kyy)
    )


def mtv(
    X1: Optional[Union[np.ndarray, callable]] = None,
    X2: Optional[Union[np.ndarray, callable]] = None,
    posterior=None,
    *args,
    **kwargs,
) -> Union[np.ndarray, Exception]:
    """
    Marginal total variation distances between two sets of posterior samples.

    Compute the total variation distance between posterior samples X1 and
    posterior samples X2, separately for each dimension (hence
    "marginal" total variation distance, MTV).

    Parameters
    ----------
    X1 : np.ndarray or callable, optional
        A ``N1``-by-``D`` matrix of samples, typically N1 = 1e5.
        Alternatively, may be a callable ``X1(x, d)`` which returns the marginal
        pdf along dimension ``d`` at point(s) ``x``.
    X2 : np.ndarray or callable, optional
        Another ``N2``-by-``D`` matrix of samples, typically N2 = 1e5.
        Alternatively, may be a callable ``X2(x, d)`` which returns the marginal
        pdf along dimension ``d`` at point(s) ``x``.
    posterior: benchflow.posteriors.Posterior, optional
        The posterior object from a benchflow run. Used to obtain samples if
        ``X1`` or ``X2`` are ``None``.

    Returns
    -------
    mtv: np.ndarray
        A ``D``-element vector whose elements are the total variation distance
        between the marginal distributions of ``vp`` and ``vp1`` or ``samples``,
        for each coordinate dimension.

    Raises
    ------
    ValueError
        Raised if neither ``vp2`` nor ``samples`` are specified.

    Notes
    -----
    The total variation distance between two densities `p1` and `p2` is:

    .. math:: TV(p1, p2) = \\frac{1}{2} \\int | p1(x) - p2(x) | dx.

    """
    # If samples are not provided, fetch them from the posterior object:
    if all(a is None for a in [X1, X2, posterior]):
        raise ValueError("No samples/callable or posterior provided.")
    if posterior is not None:
        try:  # Get analytical marginals, if possible
            X1, bounds_1 = posterior.get_marginals()
        except AttributeError:  # Otherwise use samples
            X1 = posterior.get_samples()
            if isinstance(X1, Exception):
                return X1  # Record errors, if any
        try:  # Get analytical marginals, if possible
            X2, bounds_2 = posterior.task.get_marginals()
        except AttributeError:  # Otherwise use samples
            X2 = posterior.task.get_posterior_samples()
            if isinstance(X2, Exception):
                return X2  # Record errors, if any
        D = posterior.task.D
    else:
        D = X1.shape[1]

    nkde = 2**13
    mtv = np.zeros((D,))

    # Compute marginal total variation
    for d in range(D):

        if not callable(X1):
            yy1, x1mesh, _ = kde1d(X1[:, d], nkde)
            # Ensure normalization
            yy1 = yy1 / simpson(yy1, x1mesh)

            def f1(x):
                return interp1d(
                    x1mesh,
                    yy1,
                    kind="cubic",
                    fill_value=np.array([0]),
                    bounds_error=False,
                )(x)

        else:

            def f1(x):
                return X1(x, d).ravel()  # Analytical marginal

            x1mesh = bounds_1[:, d]  # Marginal bounds

        if not callable(X2):
            yy2, x2mesh, _ = kde1d(X2[:, d], nkde)
            # Ensure normalization
            yy2 = yy2 / simpson(yy2, x2mesh)

            def f2(x):
                return interp1d(
                    x2mesh,
                    yy2,
                    kind="cubic",
                    fill_value=np.array([0]),
                    bounds_error=False,
                )(x)

        else:

            def f2(x):
                return X2(x, d).ravel()  # Analytical marginal

            x2mesh = bounds_2[:, d]  # Marginal bounds

        def f(x):
            return np.abs(f1(x) - f2(x))

        lb = min(x1mesh[0], x2mesh[0])
        ub = max(x1mesh[-1], x2mesh[-1])
        if not np.isinf(lb) and not np.isinf(ub):
            # Grid integration (faster)
            grid = np.linspace(lb, ub, int(1e6))
            y_tot = f(grid)
            mtv[d] = 0.5 * simpson(y_tot, grid)
        else:
            # QUADPACK integration (slower)
            mtv[d] = 0.5 * quad(f, lb, ub)[0]
    return mtv


def compute_mmtv(
    X1: Optional[Union[np.ndarray, callable]] = None,
    X2: Optional[Union[np.ndarray, callable]] = None,
    posterior=None,
    *args,
    **kwargs,
) -> Union[float, Exception]:
    """
    Mean marginal total variation dist. between two set of posterior samples.
    """
    result = mtv(X1, X2, posterior)
    if isinstance(result, Exception):
        return result
    else:
        return result.mean()


def median_heuristic(X: np.ndarray, Y: np.ndarray) -> float:
    """RBF lengthscale via the median heuristic: median pairwise distance / sqrt(2)."""
    Z = np.concatenate([X, Y], axis=0)
    Z_sqnorms = np.sum(Z ** 2, axis=1)
    sq_dists = Z_sqnorms[:, None] + Z_sqnorms[None, :] - 2.0 * Z @ Z.T
    sq_dists = np.maximum(sq_dists, 0.0)
    median_sq = np.median(sq_dists[sq_dists > 0])
    return float(np.sqrt(median_sq / 2.0)) if median_sq > 0 else 1.0


def compute_mmd(X: np.ndarray, Y: np.ndarray) -> float:
    """MMD with RBF kernel, lengthscale chosen via median heuristic."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    ls = median_heuristic(X, Y)
    return float(compute_mmd_unweighted(X, Y, ls))


def compute_wasserstein2(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean marginal Wasserstein-2 distance between samples X and Y.

    Computes the 1-D W2 distance per dimension using sorted quantiles,
    then averages across dimensions.  For D=1 this equals the exact W2;
    for D>1 it is the mean of marginal W2 distances.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    D = X.shape[1]
    w2_per_dim = []
    for d in range(D):
        xs = np.sort(X[:, d])
        ys = np.sort(Y[:, d])
        # interpolate the smaller set onto the quantiles of the larger
        n, m = len(xs), len(ys)
        if n != m:
            quantiles = (np.arange(max(n, m)) + 0.5) / max(n, m)
            xs = np.interp(quantiles, (np.arange(n) + 0.5) / n, xs)
            ys = np.interp(quantiles, (np.arange(m) + 0.5) / m, ys)
        w2_per_dim.append(float(np.sqrt(np.mean((xs - ys) ** 2))))
    return float(np.mean(w2_per_dim))


def compute_gskl(
    X1: Optional[np.ndarray] = None,
    X2: Optional[np.ndarray] = None,
    posterior=None,
    *args,
    **kwargs,
):
    """ "Gaussianized" symmetric Kullback-Leibler divergence (gsKL) between two
    sets of samples.

    gsKL is the symmetric KL divergence between two multivariate normal
    distributions with the same moments as samples. The symmetric KL divergence
    is the average of forward and reverse KL divergence.

    Parameters
    ----------
    X1 : np.ndarray, optional
        A ``N1``-by-``D`` matrix of samples.
    X2 : np.ndarray, optional
        Another ``N2``-by-``D`` matrix of samples.
    posterior: benchflow.posteriors.Posterior, optional
        The posterior object from a benchflow run. Used to obtain samples if
        ``X1`` or ``X2`` are ``None``.

    Returns
    -------
    kl: float
        gsKL of the two sets of samples.

    Notes
    -----
    Since the KL divergence is not symmetric, the method returns the average of
    forward and the reverse KL divergence, that is KL(``vp1`` || ``vp2``) and
    KL(``vp2`` || ``vp1``).

    """
    # If samples are not provided, fetch them from the posterior object:
    if all(a is None for a in [X1, X2, posterior]):
        raise ValueError("No samples or posterior provided.")
    if posterior is not None:
        X1 = posterior.get_samples()
        X2 = posterior.task.get_posterior_samples()
        if isinstance(X1, Exception):  # Record errors, if any
            return X1
        if isinstance(X2, Exception):
            return X2

    q1mu = np.mean(X1, axis=0)
    q1sigma = np.cov(X1.T)
    q2mu = np.mean(X2, axis=0)
    q2sigma = np.cov(X2.T)

    kls = kldiv_mvn(q1mu, q1sigma, q2mu, q2sigma)

    # Correct for numerical errors
    kls[kls < 0] = 0
    return kls.mean()
