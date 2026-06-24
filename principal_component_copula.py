import numpy as np
from numpy.linalg import eigh
from scipy.linalg import eigh as scipy_eigh
from scipy.optimize import minimize, brentq
from scipy.special import gammaln, gamma as gamma_func, kv as bessel_kv
from scipy.stats import norm, genhyperbolic, t, invgamma, rankdata
from scipy.interpolate import interp1d
from statsmodels.stats.correlation_tools import corr_nearest
from typing import Literal, Tuple, Dict, Callable, List, Optional, TypedDict
import tqdm as tqdm_module
import warnings
import logging

logger = logging.getLogger(__name__)


# Typed containers for the fitted generator parameters (self.params).
# These document the runtime dict schemas; runtime dict construction is
# unchanged and these are not enforced at runtime.


class GHNormalParams(TypedDict):
    """Fitted parameters for the Hyperbolic-Normal PCC (cop_type='normal')."""

    lam: float
    chi: float
    psi: float
    mu: float
    beta_bar: float
    alpha_bar: float


class SkewTParams(TypedDict):
    """Fitted parameters for the Skew-t_k / t_{d-k} PCC (cop_type='t')."""

    nu1: float
    gamma1: np.ndarray  # length-k array
    Sigma11: np.ndarray  # length-k array
    mu1: np.ndarray  # length-k array
    nu_rest: float
    Sigma_diag: np.ndarray  # length-(d-k) array


class CrossIndependentParams(TypedDict):
    """Fitted parameters for the independent cross-asset PCC (dependent=False)."""

    per_pc: List[Dict]
    K: int
    dependent: bool


class CrossDependentParams(TypedDict):
    """Fitted parameters for the dependent cross-asset PCC (dependent=True)."""

    t_params_dep: Dict
    K: int
    dependent: bool


class PrincipalComponentCopula:
    """
    Principal Component Copula (PCC).

    Combines copula-based dependence modelling with PCA to capture tail
    dependence along the most important directions in high-dimensional data
    (Gubbels et al., 2025).

    Parameters
    --
    dim : int
        Dimension d of the copula.
    cop_type : {"t", "cross", "normal"}
        Generator family for the first principal component.
        "t"      -> Skew-tk/ t_{d-k} PCC (GH-skew-t first PC, t higher PCs).
        "cross"  -> Skew-t_k/Normal (Skew-t_1*k/Normal) j>K (GH-skew-t first PCs (independent), Gaussian higher PCs)
        "normal" -> Hyperbolic-Normal PCC (GH first PC, normal higher PCs).

    Attributes (populated after fitting)
    --
    params : dict
        Fitted generator parameters; the schema depends on ``cop_type``.
        See the ``GHNormalParams`` / ``SkewTParams`` /
        ``CrossIndependentParams`` / ``CrossDependentParams`` TypedDicts for
        the exact key/type layout.

        "normal": dict with keys
            ``{lam, chi, psi, mu, beta_bar, alpha_bar}``
            (the constrained GH parameters of the first PC; higher PCs are
            Gaussian with variances ``Lambda[1:]``).
        "t": dict with keys
            ``{nu1, gamma1, Sigma11, mu1, nu_rest, Sigma_diag}``
            where ``gamma1``, ``Sigma11`` and ``mu1`` are length-k arrays for
            the GH-skew-t leading block, ``nu_rest`` is the degrees of freedom
            of the higher symmetric-t PCs and ``Sigma_diag`` their diagonal
            scales.
        "cross": dict describing the cross-asset PCC. For the independent
            case (``dependent=False``):
                ``{"per_pc": [...], "K": int, "dependent": False}``
            For the dependent case (``dependent=True``):
                ``{"t_params_dep": {...}, "K": int, "dependent": True}``.
    W : np.ndarray, shape (d, d)
        Eigenvector matrix from PCA on normal scores (columns = eigenvectors,
        ordered by descending eigenvalue).
    Lambda : np.ndarray, shape (d,)
        Eigenvalues in descending order.
    method : str
        Estimation method used ("GMM" or "MLE").
    """

    def __init__(
        self, dim: int, cop_type: Literal["t", "cross", "normal"] = "t", k: int = 1
    ):
        if cop_type not in {"t", "cross", "normal"}:
            raise ValueError(
                f"cop_type must be one of 't', 'cross', 'normal'; got {cop_type!r}."
            )
        if not isinstance(dim, (int, np.integer)) or dim <= 0:
            raise ValueError(f"dim must be a positive int; got {dim!r}.")
        if not isinstance(k, (int, np.integer)) or not (1 <= k <= dim):
            raise ValueError(
                f"k must be a positive int with 1 <= k <= dim={dim}; got {k!r}."
            )
        self.type = cop_type
        self.dim = dim
        self.k = k
        self.params = None
        self.W = None
        self.Lambda = None
        self.method = None
        self.max_iters = None
        self.loglikelihood = None
        self.uncertainty = None

    # PCA

    def _pca_decomposition(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigenvalue decomposition of the correlation matrix

            rho = W  Lambda  W'

        with eigenvalues sorted in *descending* order.  A sign convention
        is enforced so that the entry of each eigenvector with the largest
        absolute value is positive  (Section 2.3 of the paper).

        Parameters
        --
        rho : (d, d) symmetric positive-semidefinite matrix

        Returns
        ---
        Lambda : (d,) eigenvalues in descending order
        W      : (d, d) orthogonal matrix  (columns are eigenvectors)
        """
        eigenvalues, eigenvectors = eigh(rho)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # sign convention
        for j in range(eigenvectors.shape[1]):
            if eigenvectors[np.argmax(np.abs(eigenvectors[:, j])), j] < 0:
                eigenvectors[:, j] *= -1

        eigenvalues = np.maximum(eigenvalues, 0.0)
        return eigenvalues, eigenvectors

    def _estimate_correlation(
        self,
        Y: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate the correlation matrix from data Y.

        Parameters
        --
        Y : (n,d) array
        """
        # Handle NaN values: replace with 0 or use safe defaults
        if np.any(~np.isfinite(Y)):
            # If Y contains NaN/inf, use correlation of valid rows or return identity
            valid_mask = np.all(np.isfinite(Y), axis=1)
            if np.sum(valid_mask) < max(2, Y.shape[1]):
                # Not enough valid data, return identity correlation
                return np.eye(Y.shape[1])
            Y = Y[valid_mask]

        return np.corrcoef(Y, rowvar=False)

    def _nearest_psd_corr(
        self, corr: np.ndarray, threshold: float = 1e-12
    ) -> np.ndarray:
        """Nearest PSD correlation matrix, robust to ill-conditioned inputs.

        Tries corr_nearest first; falls back to progressive diagonal jitter then
        manual eigenvalue clipping via scipy (more robust LAPACK driver).
        """
        # Primary path
        try:
            return corr_nearest(corr, threshold=threshold)
        except np.linalg.LinAlgError:
            pass

        # Jitter + retry: progressively perturb until corr_nearest converges
        d = corr.shape[0]
        for jitter in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
            try:
                c = corr + jitter * np.eye(d)
                s = np.sqrt(np.diag(c))
                c = c / np.outer(s, s)
                np.fill_diagonal(c, 1.0)
                return corr_nearest(c, threshold=threshold)
            except np.linalg.LinAlgError:
                continue

        # Final fallback: scipy eigh clips eigenvalues directly
        try:
            evals, evecs = scipy_eigh(corr)
            evals = np.maximum(evals, threshold)
            c = evecs @ np.diag(evals) @ evecs.T
            s = np.sqrt(np.diag(c))
            s = np.where(s > 0, s, 1.0)
            c = c / np.outer(s, s)
            np.fill_diagonal(c, 1.0)
            return c
        except Exception:
            return np.eye(d)

    # Score / density utilities

    def _pseudo_observations(self, X: np.ndarray) -> np.ndarray:
        """Rank-based pseudo-observations U=rank(X)/(n+1) with average ranks for ties."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        ranks = rankdata(X, method="average", axis=0)
        return ranks / (n + 1.0)

    def _normal_scores(self, U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Normal-score transform  Y = Phi^{-1}(U).

        Parameters
        --
        U : np.ndarray, shape (n, d) -- pseudo-observations in (0, 1).
        eps : float -- clipping bound for numerical stability.

        Returns
        ---
        np.ndarray, shape (n, d)
        """
        return norm.ppf(np.clip(U, eps, 1.0 - eps))

    def _normal_log_density(self, x: np.ndarray, variance: float) -> np.ndarray:
        """Log-density of N(0, variance)."""
        return -0.5 * np.log(2.0 * np.pi * variance) - x**2 / (2.0 * variance)

    def _cos_coefficients(
        self, cf_func: Callable, a: float, b: float, Nc: int
    ) -> np.ndarray:
        """
        COS expansion coefficients  c_k  (Eq. 21).

            c_k = (2/(b-a)) Re[ phi(k pi/(b-a)) exp(-i k a pi/(b-a)) ]
        """
        k = np.arange(Nc + 1)
        t_k = k * np.pi / (b - a)
        cf_vals = cf_func(t_k)
        c_k = (2.0 / (b - a)) * np.real(cf_vals * np.exp(-1j * k * np.pi * a / (b - a)))
        return c_k  # shape (Nc+1,)

    def _gig_mean_var(self, lam: float, chi: float, psi: float) -> Tuple[float, float]:
        """
        First two moments of W ~ GIG(lambda, chi, psi).

            E[W]   = sqrt(chi/psi) * K_{lam+1}(omega) / K_lam(omega)
            E[W^2] = (chi/psi)     * K_{lam+2}(omega) / K_lam(omega)

        with omega = sqrt(chi * psi).

        Uses log-Bessel ratios for numerical stability.

        Returns
        ---
        EW, VarW : float, float
        """
        chi = max(chi, 1e-15)
        psi = max(psi, 1e-15)
        omega = np.sqrt(chi * psi)
        ratio = np.sqrt(chi / psi)

        # Use real parts and guard against zero / inf
        lK0 = np.log(np.maximum(np.real(bessel_kv(lam, omega)), 1e-300))
        lK1 = np.log(np.maximum(np.real(bessel_kv(lam + 1, omega)), 1e-300))
        lK2 = np.log(np.maximum(np.real(bessel_kv(lam + 2, omega)), 1e-300))

        EW = ratio * np.exp(lK1 - lK0)
        EW2 = (chi / psi) * np.exp(lK2 - lK0)
        VarW = max(EW2 - EW**2, 0.0)
        return float(EW), float(VarW)

    def _gh_constrained_params(
        self, alpha_bar: float, beta_bar: float, lam: float, target_variance: float
    ):
        """
        Given free shape parameters (alpha_bar, beta_bar) and the target
        variance Lambda_1, compute the full set of constrained GH parameters.

        Convention: Sigma = 1, gamma = beta_bar  (see Section 2.2.1).

        Returns
        ---
        params : dict   keys  lam, chi, psi, mu, beta_bar, alpha_bar
        """
        psi = alpha_bar**2 - beta_bar**2
        chi = self._gh_solve_chi(alpha_bar, beta_bar, lam, target_variance)
        EW, _ = self._gig_mean_var(lam, chi, psi)
        mu = -beta_bar * EW  # E[P_1] = 0

        return dict(
            lam=lam, chi=chi, psi=psi, mu=mu, beta_bar=beta_bar, alpha_bar=alpha_bar
        )

    def _gh_solve_chi(
        self, alpha_bar, beta_bar, lam, target_variance, chi_lo=1e-4, chi_hi=1e5
    ):
        """
        Solve for chi such that  Var[P_1] = target_variance,
        where P_1 ~ GH with Sigma=1 convention.

            Var[P_1] = E[W] + beta_bar^2 * Var[W]

        with W ~ GIG(lam, chi, psi),  psi = alpha_bar^2 - beta_bar^2.

        Uses Brent's method (monotonicity guaranteed by the paper).
        Adaptively widens the bracket if needed.
        """
        psi = alpha_bar**2 - beta_bar**2

        def _residual(chi):
            EW, VarW = self._gig_mean_var(lam, chi, psi)
            return EW + beta_bar**2 * VarW - target_variance

        # adaptively find a bracket where the sign changes
        lo, hi = chi_lo, chi_hi
        f_lo, f_hi = _residual(lo), _residual(hi)

        # widen bracket if necessary
        for _ in range(20):
            if np.isfinite(f_lo) and np.isfinite(f_hi) and f_lo * f_hi < 0:
                break
            if not np.isfinite(f_lo) or f_lo > 0:
                lo *= 10
                f_lo = _residual(lo)
            if not np.isfinite(f_hi) or f_hi < 0:
                hi *= 10
                f_hi = _residual(hi)
            lo = max(lo, 1e-6)
        return brentq(_residual, lo, hi, xtol=1e-12, maxiter=300)

    # Skew-t1 / t_{d-1}  --  parameter helpers

    def _skewt_constrained_params(
        self, nu1: float, gamma1: float, Lambda1: float
    ) -> Dict:
        """
        Compute constrained parameters for the first k generator P_k
        so that E[P_k] = 0 and Var(P_k) = Lambda_l  (Eq. 10).

            mu_1    = - nu_1 gamma_1 / (nu_1 - 2)
            Sigma_11 = (nu-2)/nu * Lambda
                - 2 nu / ((nu-2)(nu-4)) * (gamma @ gamma.T)

        Requires nu_1 > 4 for finite Sigma_11 when gamma_1 != 0.

        Returns
        ---
        dict with keys: nu1, gamma1, Sigma11, mu1
        """
        gamma1 = np.asarray(gamma1)

        # Calculate location vector (mu)
        mu = -nu1 * gamma1 / (nu1 - 2.0)

        gamma_sq = gamma1**2

        # Calculate the vector of diagonal dispersion parameters
        Sigma_kk = ((nu1 - 2.0) / nu1 * Lambda1) - (
            (2.0 * nu1) / ((nu1 - 2.0) * (nu1 - 4.0)) * gamma_sq
        )

        return dict(nu1=nu1, gamma1=gamma1, Sigma11=Sigma_kk, mu1=mu)

    def _skewt_mvt_logpdf(
        self,
        x: np.ndarray,
        nu: float,
        gamma: np.ndarray,
        Sigma_kk: np.ndarray,
        mu: np.ndarray,
    ) -> np.ndarray:
        """
        Log-density of the k-D GH skew-t with a diagonal dispersion matrix.
        Assumes components are independent given the mixing variable V.

        Parameters
        --
        x        : np.ndarray, shape (N, k) or (k,)
        nu       : float, degrees of freedom
        gamma    : np.ndarray, shape (k,) skewness vector
        Sigma_kk : np.ndarray, shape (k,) diagonal dispersion vector
        mu       : np.ndarray, shape (k,) location vector
        """
        x = np.asarray(x, dtype=float)
        # Ensure x is always 2-D: (n,) batch input for k=1 becomes (n, 1).
        # This prevents np.sum(..., axis=-1) from collapsing a 1-D batch to a scalar.
        if x.ndim == 1:
            x = x[:, None]
        k = gamma.shape[0]

        if np.all(np.abs(gamma) < 1e-12):
            # When gamma = 0, P_{1:k} are symmetric and share mixing V, so their
            # joint density is the k-dim multivariate-t — NOT the product of
            # marginals (which would ignore the tail dependence from shared V).
            # mu = 0 when gamma = 0, so xc ≈ x.
            xc = x - mu
            return self._mvt_diag_logpdf(xc, nu, Sigma_kk)

        xc = x - mu

        sq_dist = np.sum(xc**2 / Sigma_kk, axis=-1)
        Q = np.sqrt(nu + sq_dist)
        skew_term = np.sum(gamma * xc / Sigma_kk, axis=-1)

        half_nu = nu / 2.0
        half_nupk = (nu + k) / 2.0

        gamma_norm_sigma = np.sqrt(np.sum(gamma**2 / Sigma_kk))

        arg_K = gamma_norm_sigma * Q

        log_det_sigma = np.sum(np.log(Sigma_kk))

        log_f = (
            half_nu * np.log(nu / 2.0)
            - gammaln(half_nu)
            - (k / 2.0) * np.log(2.0 * np.pi)
            - 0.5 * log_det_sigma
            + np.log(2.0)
            + skew_term
            + half_nupk * np.log(gamma_norm_sigma / Q)
            + np.log(np.maximum(np.real(bessel_kv(half_nupk, arg_K)), 1e-300))
        )
        return log_f

    def _mvt_diag_logpdf(
        self,
        P_rest: np.ndarray,
        nu: float,
        Sigma_diag: np.ndarray,
    ) -> np.ndarray:
        """
        Log-density of the (d-1)-dimensional multivariate t distribution
        with zero mean and diagonal scale matrix Sigma_diag  (Eq. 11).

            log f(p) = gammaln((nu+d-1)/2) - gammaln(nu/2)
                     - (d-1)/2 log(nu pi) - 0.5 sum log(Sigma_{j,j})
                     - (nu+d-1)/2 log(1 + sum p_j^2 / (nu Sigma_{j,j}))

        Parameters
        --
        P_rest     : (n, d-1)  PC scores for generators j >= 2
        nu         : float > 2   common DoF for higher generators
        Sigma_diag : (d-1,)      diagonal scale entries
        """
        n, dm1 = P_rest.shape
        half_nu = nu / 2.0
        half_nup = (nu + dm1) / 2.0

        maha = np.sum(P_rest**2 / (nu * Sigma_diag[None, :]), axis=1)

        log_f = (
            gammaln(half_nup)
            - gammaln(half_nu)
            - dm1 / 2.0 * np.log(nu * np.pi)
            - 0.5 * np.sum(np.log(Sigma_diag))
            - half_nup * np.log1p(maha)
        )
        return log_f

    # CFs, PDFs, DFs and Inversion

    def _cos_density(
        self,
        y_grid: np.ndarray,
        cf_func: Callable,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
    ) -> np.ndarray:
        """
        Approximate the probability density via the COS method  (Eq. 20).

            f(y) ≈ c_0/2  +  sum_{k=1}^{Nc} c_k cos(k pi (y-a)/(b-a))

        Parameters
        --
        y_grid  : 1-D array of evaluation points
        cf_func : callable  phi(t) -> complex array
        a, b    : float     truncation interval  (default [-10, 10])
        Nc      : int       number of cosine terms

        Returns
        ---
        density : 1-D array, same length as y_grid  (floored at 1e-300)
        """
        y_grid = np.asarray(y_grid, dtype=float)
        c_k = self._cos_coefficients(cf_func, a, b, Nc)
        k = np.arange(Nc + 1)

        # (n_pts, Nc+1) matrix of cosine arguments
        arg = np.outer(y_grid - a, k * np.pi / (b - a))
        cos_mat = np.cos(arg)
        cos_mat[:, 0] = 0.5  # accounts for c_0 / 2

        density = cos_mat @ c_k
        return np.maximum(density, 1e-300)

    def _cos_cdf(
        self,
        y_grid: np.ndarray,
        cf_func: Callable,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
    ) -> np.ndarray:
        """
        Approximate the CDF via the COS method  (Eq. 22).

            F(y) ≈ c_0/2 (y-a)
                + sum_{k=1}^{Nc} c_k (b-a)/(k pi) sin(k pi (y-a)/(b-a))

        Returns
        ---
        cdf : 1-D array clipped to [0, 1]
        """
        y_grid = np.asarray(y_grid, dtype=float)
        c_k = self._cos_coefficients(cf_func, a, b, Nc)

        # k = 0 term
        cdf = c_k[0] / 2.0 * (y_grid - a)

        # k >= 1 terms
        k_pos = np.arange(1, len(c_k))
        arg = np.outer(y_grid - a, k_pos * np.pi / (b - a))
        sin_mat = np.sin(arg)
        coeffs = c_k[1:] * (b - a) / (k_pos * np.pi)
        cdf += sin_mat @ coeffs

        return np.clip(cdf, 0.0, 1.0)

    def _build_generator_cfs(
        self, Lambda: np.ndarray, gh_params: Dict
    ) -> List[Callable]:
        """
        Build list of d generator characteristic functions for the GH-Normal PCC.

        For the GH-Normal PCC (Lemma 1, Case 1 — all generators independent):
            - cf_generators[0] : GH characteristic function for P_1
            - cf_generators[j] : Normal CF with variance Lambda_j  for j >= 1

        To extend to another PCC family (e.g. Skew-t + t), write an analogous
        function that returns the appropriate list of CFs.

        Parameters
        --
        Lambda    : (d,)  eigenvalues
        gh_params : dict  output of gh_constrained_params

        Returns
        ---
        cf_generators : list of d callables,  each  t -> complex array
        """
        d = len(Lambda)
        gp = gh_params

        # first generator: GH
        cf_generators = [
            lambda t, _gp=gp: self._cf_gh_1d(
                t, _gp["lam"], _gp["chi"], _gp["psi"], _gp["mu"], _gp["beta_bar"]
            )
        ]

        # higher generators: Normal with variance Lambda_j
        for j in range(1, d):
            lam_j = Lambda[j]
            cf_generators.append(lambda t, v=lam_j: self._cf_normal(t, v))

        return cf_generators

    def _build_generator_cfs_cross(
        self, Lambda: np.ndarray, per_pc_params: List[Dict], K: int
    ) -> List[Callable]:
        """
        Build list of d generator CFs for the cross-asset PCC (§3).

        j < K  : independent Skew-t_1 CF — reuses _cf_skewt with scalar params.
        j >= K : Normal CF with variance Lambda[j] — reuses _cf_normal.

        Because all K generators are mutually independent (Case 1),
        the result is passed directly to the existing _make_marginal_cf.

        Parameters
        --
        Lambda        : (d,) eigenvalues
        per_pc_params : list of K dicts {nu, gamma, Sigma, mu}
        K             : int, number of Skew-t_1 generators
        """
        d = len(Lambda)
        cfs: List[Callable] = []
        for j in range(K):
            p = per_pc_params[j]
            nu_j, g_j, s_j, m_j = p["nu"], p["gamma"], p["Sigma"], p["mu"]
            cfs.append(
                lambda t, _nu=nu_j, _g=g_j, _s=s_j, _m=m_j: self._cf_skewt(
                    t, _nu, _g, _s, _m
                )
            )
        for j in range(K, d):
            lam_j = Lambda[j]
            cfs.append(lambda t, v=lam_j: self._cf_normal(t, v))
        return cfs

    def _cf_normal(self, t: np.ndarray, variance: float = 1.0) -> np.ndarray:
        """
        Characteristic function of N(0, variance).

            phi(t) = exp(-0.5 * variance * t^2)

        Parameters
        --
        t : array_like (real or complex)
        variance : float > 0

        Returns
        ---
        phi : ndarray (complex)
        """
        t = np.asarray(t, dtype=complex)
        return np.exp(-0.5 * variance * t**2)

    def _cf_gh_1d(
        self,
        t: np.ndarray,
        lam: float,
        chi: float,
        psi: float,
        mu: float,
        beta_bar: float,
    ) -> np.ndarray:
        """
        Characteristic function of the one-dimensional Generalized Hyperbolic
        distribution in the Sigma=1 convention (Eq. 8 specialised to d=1).

            phi(t) = exp(i t mu) * (psi / z)^{lam/2}
                    * K_lam(sqrt(chi * z)) / K_lam(sqrt(chi * psi))

        where z = psi + t^2 - 2 i t beta_bar.

        Parameters
        --
        t         : array_like  (will be cast to complex)
        lam       : float       GH shape index (lambda=1  -> Hyperbolic,
                                lambda=-0.5 -> NIG, lambda=-nu/2 -> skew-t)
        chi       : float > 0   scale parameter
        psi       : float > 0   psi = alpha_bar^2 - beta_bar^2
        mu        : float        location (chosen so that mean = 0)
        beta_bar  : float        skewness parameter (|beta_bar| < alpha_bar)

        Returns
        ---
        phi : ndarray (complex)
        """
        t = np.asarray(t, dtype=complex)
        z = psi + t**2 - 2j * t * beta_bar

        sqrt_chi_psi = np.sqrt(chi * psi + 0j)
        sqrt_chi_z = np.sqrt(chi * z)

        K_num = bessel_kv(lam, sqrt_chi_z)
        K_den = bessel_kv(lam, sqrt_chi_psi)

        phi = np.exp(1j * t * mu) * (psi / z) ** (lam / 2.0) * (K_num / K_den)
        return phi

    def _cf_skewt(
        self,
        t: np.ndarray,
        nu: float,
        gamma: np.ndarray,
        Sigma_kk: np.ndarray,
        mu: np.ndarray,
    ) -> np.ndarray:
        """
        Multivariate Characteristic Function for GH Skew-t with diagonal dispersion.
        Reduces to 1-D when inputs are scalars and handles symmetric t when gamma=0.
        """
        t = np.asarray(t, dtype=complex)
        gamma = np.asarray(gamma)
        Sigma_kk = np.asarray(Sigma_kk)
        mu = np.asarray(mu)

        half_nu = nu / 2.0

        if t.ndim > 1:
            dot_mu = np.sum(t * mu, axis=-1)
            q = np.sum(Sigma_kk * t**2, axis=-1) - 2j * np.sum(t * gamma, axis=-1)
            t_norm = np.linalg.norm(t, axis=-1)
        else:
            dot_mu = t * mu
            q = Sigma_kk * t**2 - 2j * gamma * t
            t_norm = np.abs(t)

        sqrt_nu_q = np.sqrt(nu * q + 0j)
        K_val = bessel_kv(half_nu, sqrt_nu_q)

        phi = (
            np.exp(1j * dot_mu)
            * 2.0
            / gamma_func(half_nu)
            * (sqrt_nu_q / 2.0) ** half_nu
            * K_val
        )

        phi = np.where(t_norm < 1e-15, 1.0 + 0j, phi)
        return phi

    def _cf_symm_t_1d(self, t: np.ndarray, nu: float, sigma_sq: float) -> np.ndarray:
        """
        Characteristic function of the 1-D symmetric Student-t with
        mean 0 and scale sigma_sq  (Var = nu/(nu-2) * sigma_sq).

        Special case of _cf_skewt with gamma = 0, mu = 0:
            phi(t) = 2 / Gamma(nu/2)
                     * (sqrt(nu sigma_sq) |t| / 2)^{nu/2}
                     * K_{nu/2}(sqrt(nu sigma_sq) |t|)

        Used for the multivariate-t contribution to marginal CFs
        under Lemma 1 Case 2, where the effective scale is
        sigma_sq_eff = sum_{j>=2} w_{i,j}^2  Sigma_{j,j}.
        """
        return self._cf_skewt(t, nu, 0.0, sigma_sq, 0.0)

    def _build_inverse_cdf(
        self,
        cf_func: Callable,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
        n_grid: int = 2000,
    ) -> Callable:
        """
        Build  F^{-1}_{Y_i}  by evaluating the COS-CDF on a fine grid
        and constructing a monotone linear interpolant.

        Returns
        ---
        inv_cdf : callable   u  ->  y = F^{-1}(u)
        """
        y_grid = np.linspace(a, b, n_grid)
        cdf_vals = self._cos_cdf(y_grid, cf_func, a, b, Nc)

        # enforce strict monotonicity
        cdf_vals = np.maximum.accumulate(cdf_vals)
        mask = np.concatenate([[True], np.diff(cdf_vals) > 1e-15])
        cdf_c = cdf_vals[mask]
        y_c = y_grid[mask]

        inv_cdf = interp1d(
            cdf_c, y_c, kind="linear", bounds_error=False, fill_value=(a, b)
        )
        return inv_cdf

    def _gh_1d_log_density(
        self,
        x: np.ndarray,
        lam: float,
        chi: float,
        psi: float,
        mu: float,
        alpha_bar: float,
        beta_bar: float,
    ) -> float:
        """
        Log-density of the 1-D Generalized Hyperbolic distribution
        with Sigma = 1 convention.

            log f(x) = (lam/2) log(psi/chi)
                    + (1/2 - lam) log(alpha_bar)
                    + beta_bar (x - mu)
                    - 0.5 log(2 pi)
                    - log K_lam(sqrt(chi psi))
                    + log K_{lam-1/2}(alpha_bar Q)
                    - (1/2 - lam) log(Q)

        where  Q = sqrt(chi + (x - mu)^2).
        """
        x = np.asarray(x, dtype=float)
        xc = x - mu
        Q = np.sqrt(chi + xc**2)
        omega = np.sqrt(chi * psi)

        log_f = (lam / 2.0) * np.log(psi / chi)
        log_f += (0.5 - lam) * np.log(max(alpha_bar, 1e-300))
        log_f += beta_bar * xc
        log_f -= 0.5 * np.log(2.0 * np.pi)

        K_denom = np.real(bessel_kv(lam, omega))
        log_f -= np.log(np.maximum(np.abs(K_denom), 1e-300))

        K_numer = np.real(bessel_kv(lam - 0.5, alpha_bar * Q))
        log_f += np.log(np.maximum(np.abs(K_numer), 1e-300))

        log_f -= (0.5 - lam) * np.log(np.maximum(Q, 1e-300))
        return log_f

    # HB-N  --  log-copula density & negative pseudo-log-likelihood

    def _compute_copula_log_likelihood(
        self,
        U: np.ndarray,
        W: np.ndarray,
        Lambda: np.ndarray,
        gh_params: Tuple,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
        n_grid: int = 2000,
    ) -> Tuple[float, np.ndarray, List[Callable]]:
        """
        Copula log-likelihood  (Eq. 18):

            l = sum_t [ log f_Y(Y_t) - sum_i log f_{Y_i}(Y_{i,t}) ]

        where Y_{i,t} = F^{-1}_{Y_i}(U_{i,t}).

        Returns
        ---
        ll       : float         log-likelihood value
        Y        : (n, d) array  implicit copula returns
        inv_cdfs : list of d callables   (reusable inverse CDFs)
        """
        n, d = U.shape
        gp = gh_params

        #  build inverse CDFs and marginal CFs for each index i
        cf_gens = self._build_generator_cfs(Lambda, gp)

        inv_cdfs = []
        marginal_cfs = []
        for i in range(d):
            cf_i = self._make_marginal_cf(i, W, cf_gens)
            marginal_cfs.append(cf_i)
            inv_cdfs.append(self._build_inverse_cdf(cf_i, a, b, Nc, n_grid))

        #  Y_{i,t} = F^{-1}_{Y_i}(U_{i,t})
        Y = np.empty_like(U)
        for i in range(d):
            Y[:, i] = inv_cdfs[i](U[:, i])

        #  log f_Y(Y_t)  via Eq. 17:  f_Y(y) = prod_j f_{P_j}(w_j' y)
        P = Y @ W  # (n, d):  P[t, j] = w_j' Y_t

        log_fY = self._gh_1d_log_density(
            P[:, 0],
            gp["lam"],
            gp["chi"],
            gp["psi"],
            gp["mu"],
            gp["alpha_bar"],
            gp["beta_bar"],
        )
        for j in range(1, d):
            log_fY += self._normal_log_density(P[:, j], Lambda[j])

        #  sum_i log f_{Y_i}(Y_{i,t})  via COS density (Eq. 15 ≈ 20)
        log_marg_sum = np.zeros(n)
        for i in range(d):
            dens_i = self._cos_density(Y[:, i], marginal_cfs[i], a, b, Nc)
            log_marg_sum += np.log(dens_i)

        ll = float(np.sum(log_fY - log_marg_sum))
        return ll, Y, inv_cdfs

    def _compute_copula_ll_t(
        self,
        U: np.ndarray,
        W: np.ndarray,
        Lambda: np.ndarray,
        t_params: Dict,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
        n_grid: int = 2000,
        k: int = 1,
    ) -> Tuple[float, np.ndarray, List[Callable]]:
        """
        Copula log-likelihood for the Skew-t1 / t_{d-1} PCC  (Eq. 18).

            l = sum_t [ log f_Y(Y_t) - sum_i log f_{Y_i}(Y_{i,t}) ]

        Numerator (Eq. 17): f_Y(y) = f_{P_1}(w_1' y) * f_{P_>}(W_>' y)
        Denominator:        prod_i f_{Y_i}(y_i)  via COS density inversion.
        """
        n, d = U.shape

        #  build marginal CFs and inverse CDFs (Case 2)
        inv_cdfs = []
        marginal_cfs = []
        for i in range(d):
            cf_i = self._make_marginal_cf_t(i, W, t_params, k)
            marginal_cfs.append(cf_i)
            inv_cdfs.append(self._build_inverse_cdf(cf_i, a, b, Nc, n_grid))

        #  Y_{i,t} = F^{-1}_{Y_i}(U_{i,t})
        Y = np.empty_like(U)
        for i in range(d):
            Y[:, i] = inv_cdfs[i](U[:, i])

        #  log f_Y(Y_t)  via Eq. 17
        P = Y @ W  # recover generators

        # first generator: 1-D skew-t
        log_fY = self._skewt_mvt_logpdf(
            P[:, :k],
            t_params["nu1"],
            t_params["gamma1"],
            t_params["Sigma11"],
            t_params["mu1"],
        )
        # higher generators: (d-1)-dim multivariate t
        log_fY += self._mvt_diag_logpdf(
            P[:, k:], t_params["nu_rest"], t_params["Sigma_diag"]
        )

        #  sum_i log f_{Y_i}(Y_{i,t}) via COS density (Eq. 15 ≈ 20)
        log_marg_sum = np.zeros(n)
        for i in range(d):
            dens_i = self._cos_density(Y[:, i], marginal_cfs[i], a, b, Nc)
            log_marg_sum += np.log(dens_i)

        ll = float(np.sum(log_fY - log_marg_sum))
        return ll, Y, inv_cdfs

    def _neg_copula_ll_shape(
        self,
        shape_vec: Tuple[float, float],
        U: np.ndarray,
        W: np.ndarray,
        Lambda: np.ndarray,
        lam_gh: float,
        a: float,
        b: float,
        Nc: int,
        n_grid: np.ndarray,
    ) -> float:
        """
        Negative copula log-likelihood as a function of the free shape
        parameters  [alpha_bar, beta_bar].

        Used inside Algorithm 1, Step 2(c).
        """
        alpha_bar, beta_bar = shape_vec

        # feasibility checks
        if alpha_bar <= np.abs(beta_bar) + 1e-8 or alpha_bar <= 1e-8:
            return 1e12

        try:
            gp = self._gh_constrained_params(alpha_bar, beta_bar, lam_gh, Lambda[0])
        except Exception:
            return 1e12

        try:
            ll, _, _ = self._compute_copula_log_likelihood(
                U, W, Lambda, gp, a, b, Nc, n_grid
            )
        except Exception:
            return 1e12

        if not np.isfinite(ll):
            return 1e12

        return -ll

    def _neg_copula_ll_shape_t(
        self,
        shape_vec: np.ndarray,
        U: np.ndarray,
        W: np.ndarray,
        Lambda: np.ndarray,
        a: float,
        b: float,
        Nc: int,
        n_grid: int,
        k: int,
    ) -> float:
        """
        Negative copula log-likelihood as a function of the free
        shape parameters  [eta1, gamma1, eta_rest]  (unconstrained).

        Reparametrisation:
            nu_1    = 4 + exp(eta1)     (ensures nu_1 > 4)
            gamma_1 = gamma1            (unconstrained)
            nu_rest = 2 + exp(eta_rest) (ensures nu_rest > 2)
        """
        eta1 = shape_vec[0]
        gamma1 = shape_vec[1 : 1 + k]  # shape (k,); scalar slice for k=1
        eta_rest = shape_vec[1 + k]
        nu1 = 4.0 + np.exp(eta1)
        nu_rest = 2.0 + np.exp(eta_rest)

        try:
            sp = self._skewt_constrained_params(nu1, gamma1, Lambda[:k])
        except Exception:
            return 1e12

        if np.any(sp["Sigma11"] <= 1e-10):
            return 1e12

        Sigma_diag = self._diag_scale_t(nu_rest, Lambda[k:])
        t_params = {**sp, "nu_rest": nu_rest, "Sigma_diag": Sigma_diag}

        try:
            ll, _, _ = self._compute_copula_ll_t(
                U, W, Lambda, t_params, a, b, Nc, n_grid, k
            )
        except Exception:
            return 1e12

        if not np.isfinite(ll):
            return 1e12

        return -ll

    # HB-N  --  Rebuild PCC Marginals through CF inversion

    def _make_marginal_cf(
        self, i: int, W: np.ndarray, cf_generators: List[Callable]
    ) -> Callable:
        """
        Factory for the marginal characteristic function  phi_{Y_i}(t).

        Implements Eq. 12 (Lemma 1, Case 1) in full generality:

            phi_{Y_i}(t) = prod_{j=0}^{d-1}  phi_{P_j}( w_{i,j} * t )

        Each generator CF is called individually via the cf_generators list,
        so this function is agnostic to the generator family.

        Parameters
        --
        i             : int       variable index (0-based)
        W             : (d, d)    eigenvector matrix
        cf_generators : list of d callables,  cf_generators[j](t) -> complex

        Returns
        ---
        cf_Yi : callable   t  ->  complex array
        """
        w_i = W[i, :]

        def cf_Yi(t):
            t = np.asarray(t, dtype=complex)
            result = np.ones_like(t)
            for j, cf_j in enumerate(cf_generators):
                result *= cf_j(w_i[j] * t)
            return result

        return cf_Yi

    # Skew-t1 / t_{d-1}  --  parameter helpers & negative pseudo-log-likelihood

    def _make_marginal_cf_t(
        self, i: int, W: np.ndarray, t_params: Dict, k: int = 1
    ) -> Callable:
        """
        Marginal CF for Y_i under the Skew-t_k / t_{d-k} PCC.

        Implements Lemma 1, Case 2.  Since the first k generators share a
        single IG mixing variable V, the linear combination
            X_1 = sum_{j=1}^k w_{i,j} P_j
        is itself a 1-D GH skew-t with effective parameters:

            gamma_eff     = w_{i,1:k} @ gamma1          (scalar)
            mu_eff        = w_{i,1:k} @ mu1              (scalar)
            sigma_sq_eff1 = sum_{j=1}^k w_{i,j}^2 Sigma_{j,j}  (scalar)

        The higher-PC block (independent mixing variable W) contributes a
        1-D symmetric t with:

            sigma_sq_eff2 = sum_{j>k} w_{i,j}^2 Sigma_{j,j}   (scalar)

        so the marginal CF is a product of two 1-D CFs and can be
        directly inverted with the COS method.
        """
        w_i = W[i, :]

        nu1 = t_params["nu1"]
        gamma1 = np.asarray(t_params["gamma1"])
        Sigma11 = np.asarray(t_params["Sigma11"])
        mu1 = np.asarray(t_params["mu1"])

        nu_rest = t_params["nu_rest"]
        Sigma_diag = np.asarray(t_params["Sigma_diag"])

        # Project the k-dim skewt block onto the scalar direction w_{i,1:k}
        gamma_eff = float(np.dot(w_i[:k], gamma1))
        mu_eff = float(np.dot(w_i[:k], mu1))
        skew_sigma_sq_eff = float(np.dot(w_i[:k] ** 2, Sigma11))
        sigma_sq_eff = float(np.dot(w_i[k:] ** 2, Sigma_diag))

        def cf_Yi(t_arg):
            t_arg = np.asarray(t_arg, dtype=complex)
            part1 = self._cf_skewt(t_arg, nu1, gamma_eff, skew_sigma_sq_eff, mu_eff)
            part2 = self._cf_symm_t_1d(t_arg, nu_rest, sigma_sq_eff)
            return part1 * part2

        return cf_Yi

    def _diag_scale_t(self, nu_rest: float, Lambda_rest: np.ndarray) -> np.ndarray:
        """
        Diagonal scale entries Sigma_{j,j} for higher-PC t generators so that
        Var(Pj) = Lambda_j:
            Sigma_{j,j} = (nu_rest - 2) / nu_rest * Lambda_j.

        Parameters
        --
        nu_rest     : float          degrees of freedom (> 2).
        Lambda_rest : np.ndarray, shape (d-1,)
        """
        return (nu_rest - 2.0) / nu_rest * Lambda_rest

    # Skew-t1 / t_{d-1}  --  Rebuild PCC Marginals through CF inversion

    # Public fit interface

    def fit(
        self,
        X: np.ndarray,
        max_iters: int = 10,
        method: Literal["GMM", "MLE"] = "GMM",
        dependent: bool = False,
    ):
        """
        Fit the PCC to data X.

        Parameters
        --
        X         : np.ndarray, shape (n, d)
            Raw observations; marginal distributions are handled via
            rank-based pseudo-observations internally.
        max_iters : int
            Maximum iterations for the L-BFGS-B optimizer.
        method    : {"GMM", "MLE"}
            "GMM" -- hybrid pseudo-MLE / moment estimator (Algorithm 1).
            "MLE" -- full MLE via COS-method marginals (not yet implemented).
        dependent : bool
            Only relevant when cop_type="cross".  If True, the K leading
            PCs share a single IG mixing variable (Lemma 1 Case 2 for the
            K-block); if False (default) each PC has its own independent
            mixing variable (Case 1 / independent cross-asset PCC).

        Returns
        ---
        result : dict
            The fit summary returned by the underlying ``fit_*`` routine
            (keys include ``parameters``, ``W``, ``Lambda``, ...). The fitted
            generator parameters are stored on ``self.params`` as a dict whose
            schema depends on ``cop_type``:

            "normal": ``{lam, chi, psi, mu, beta_bar, alpha_bar}``.
            "t":      ``{nu1, gamma1, Sigma11, mu1, nu_rest, Sigma_diag}``
                      (``gamma1``/``Sigma11``/``mu1`` are length-k arrays).
            "cross":  ``{"per_pc": [...], "K", "dependent": False}`` when
                      independent, or
                      ``{"t_params_dep": {...}, "K", "dependent": True}`` when
                      dependent.
        """
        X = np.asarray(X, float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array; got ndim={X.ndim}.")
        if X.shape[1] != self.dim:
            raise ValueError(
                f"X has {X.shape[1]} columns but the model dimension is {self.dim}."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values (no NaN or inf).")

        self.method = method
        self.max_iters = max_iters
        if self.type == "normal":
            return self._fit_normal(X)
        elif self.type == "cross":
            return self._fit_cross(X, dependent=dependent)
        return self._fit_t(X)

    def _fit_normal(self, X: np.ndarray):
        return (
            self.fit_normal_gmm(X) if self.method == "GMM" else self.fit_normal_mle(X)
        )

    def _fit_t(self, X: np.ndarray):
        return (
            self.fit_t_gmm(X, k=self.k) if self.method == "GMM" else self.fit_t_mle(X)
        )

    def _fit_cross(self, X: np.ndarray, dependent: bool = False):
        return (
            self.fit_cross_gmm(X, k=self.k, dependent=dependent)
            if self.method == "GMM"
            else self.fit_cross_mle(X)
        )

    # HB-N  --  GMM fit

    def fit_normal_gmm(
        self,
        X: np.ndarray,
        eps: float = 1e-12,
        tol: float = 1e-6,
        ftol: float = 1e-4,
        lam0: float = 1.0,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
        n_grid: int = 2000,
    ) -> Dict:
        """
        Fit the Hyperbolic-Normal PCC via pseudo-maximum-likelihood (GMM).

        Algorithm (Fig. 3a -> Algorithm 1 of Gubbels et al., 2025):
        1. Rank-based pseudo-observations  U = rank(X) / (n+1).
        2. Normal scores  Y = Phi^{-1}(U).
        3. PCA on the sample correlation of Y  ->  W, Lambda.
        4. PC scores  P = Y @ W.
        5. Initialise GH parameters from genhyperbolic.fit on P[:,0] (floc=0).
        6. Minimise negative pseudo-log-likelihood over GH shape parameters
           using Nelder-Mead on an unconstrained reparametrisation.

        Parameters
        --
        X   : np.ndarray, shape (n, d)
        eps : float -- clipping for Phi^{-1}.

        Returns
        ---
        gh_params : tuple -- (lambda, alpha, beta, 0, sigma) for genhyperbolic.
        W         : np.ndarray, shape (d, d)
        Lambda    : np.ndarray, shape (d,)
        """
        n, _ = X.shape
        U = self._pseudo_observations(X)
        Y = self._normal_scores(U, eps)

        corr = self._estimate_correlation(Y)
        corr = self._nearest_psd_corr(corr, threshold=1e-12)
        Lambda, W = self._pca_decomposition(corr)
        P = Y @ W
        p1 = P[:, 0]

        # Marginal GH fit on first PC (fix loc=0 per paper's zero-mean constraint)
        _, a0, b0, _, _ = genhyperbolic.fit(p1, floc=0.0, fp=lam0)
        # a0 = 1.0
        # b0 = -0.1 * np.sign(np.mean(p1**3))
        prev_shape = np.array([a0, b0], dtype=float)

        converged = False
        best_nll = 1e12
        nll_prev = 1e12
        best_ab = prev_shape.copy()
        best_W = W.copy()
        best_Lambda = Lambda.copy()
        W_new, Lambda_new = W.copy(), Lambda.copy()
        alpha_bar, beta_bar = a0, b0

        for k in tqdm_module.tqdm(
            range(self.max_iters), desc=f"Initialising ({self.type}) GMM algorithm:"
        ):
            # Step 2(a):  update correlation matrix  (Eq. 25, line 1)
            gp = self._gh_constrained_params(a0, b0, lam0, Lambda[0])
            cf_gens = self._build_generator_cfs(Lambda, gp)

            Y = np.empty((n, self.dim))
            for i in range(self.dim):
                cf_Yi = self._make_marginal_cf(i, W, cf_gens)
                inv_i = self._build_inverse_cdf(cf_Yi)
                Y[:, i] = inv_i(U[:, i])

            if np.any(np.std(Y, axis=0) < 1e-6):
                rho_new = self._estimate_correlation(self._normal_scores(U, eps))
                rho_new = self._nearest_psd_corr(rho_new, threshold=1e-12)
            else:
                rho_new = self._estimate_correlation(Y)
                rho_new = self._nearest_psd_corr(rho_new, threshold=1e-12)

            # Step 2(b):  PCA
            Lambda_new, W_new = self._pca_decomposition(rho_new)
            # Step 2(c): Shape Parameters
            res = minimize(
                self._neg_copula_ll_shape,
                x0=[a0, b0],
                args=(U, W_new, Lambda_new, lam0, a, b, Nc, n_grid),
                method="Nelder-Mead",
                options={"maxiter": 1_000, "xatol": 1e-9},
            )

            alpha_bar, beta_bar = res.x

            if res.fun < best_nll:
                best_nll = res.fun
                best_ab = np.array([alpha_bar, beta_bar])
                best_W = W_new.copy()
                best_Lambda = Lambda_new.copy()

            if (
                np.abs((res.fun - nll_prev) / max(abs(nll_prev), 1e-8)) < ftol
                and res.fun < 1e6
            ):
                converged = True
                break

            nll_prev = res.fun
            prev_shape = np.array([alpha_bar, beta_bar])
            a0, b0 = alpha_bar, beta_bar
            W, Lambda = W_new, Lambda_new

        if converged:
            logger.info("Algorithm Converged")

        if best_nll < 1e6:
            W, Lambda = best_W.copy(), best_Lambda.copy()
            alpha_bar, beta_bar = float(best_ab[0]), float(best_ab[1])
        else:
            W, Lambda = W_new.copy(), Lambda_new.copy()
        # Unpack parameters
        gp_final = self._gh_constrained_params(alpha_bar, beta_bar, lam0, Lambda[0])
        ll_final, _, _ = self._compute_copula_log_likelihood(
            U, W, Lambda, gp_final, a, b, Nc, n_grid
        )
        rho_Y = W @ np.diag(Lambda) @ W.T

        self.params = gp_final
        self.W = W
        self.Lambda = Lambda
        self.loglikelihood = ll_final

        return dict(
            parameters=gp_final,
            W=W,
            Lambda=Lambda,
            rho_Y=rho_Y,
            log_likelihood=ll_final,
            converged=converged,
            n_iter=k + 1,
        )

    def fit_normal_mle(self, X: np.ndarray):
        """
        Full MLE for the HB-N PCC.

        Requires the COS method (Eqs. 20-22 of Gubbels et al., 2025) to
        evaluate marginal densities and CDFs.  Not yet implemented.
        """
        raise NotImplementedError(
            "Full MLE requires the COS-method for marginal densities (Eqs. 20-22 "
            "of Gubbels et al., 2025).  Use method='GMM' instead."
        )

    # Skew-t1 / t_{d-1}  --  GMM fit

    def fit_t_gmm(
        self,
        X: np.ndarray,
        k: int = 1,
        eps: float = 1e-12,
        tol: float = 1e-6,
        ftol: float = 1e-4,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
        n_grid: int = 2000,
    ):
        """
        Fit the Skew-t1 / t_{d-k} PCC via Algorithm 1 (Gubbels et al., 2025).

        Generator specification (Eq. 11):
            P_k  ~ GH Skew-t_k(nu_1, Lambda_k, gamma_1)
            P_>  ~ t_{d-k}(nu_rest, Lambda_>)

        The first generator uses the GH skew-t characteristic function
        (Eq. 8 with lambda = -nu/2, chi = nu, psi -> 0).  The higher
        generators share a common inverse-gamma mixing variable (Eq. 9),
        giving rise to Lemma 1 Case 2 for the marginal CFs.

        Free shape parameters optimised by ML (step 2c):
            nu_1    : DoF for the first k PCs  (nu_1 > 4)
            gamma_1 : skewness of the first PCs
            nu_rest : DoF for higher PCs    (nu_rest > 2)

        Constrained parameters from Eq. 10:
            mu_1     = -nu_1 gamma_1 / (nu_1 - 2)
            Sigma_11 = (nu_1-2)/nu_1 Lambda_1
                       - 2 nu_1 / ((nu_1-2)(nu_1-4)) gamma_1^2
            Sigma_{j,j} = (nu_rest - 2) / nu_rest Lambda_j

        Parameters
        --
        X         : (n, d) array
        eps       : float   clipping for Phi^{-1}
        tol       : float   convergence tolerance on shape change
        a, b      : float   COS truncation interval
        Nc        : int     COS cosine terms
        n_grid    : int     inverse-CDF grid size
        shrinkage : None or "ledoit_wolf"

        Returns
        ---
        dict with keys: parameters, W, Lambda, rho_Y, log_likelihood,
                        converged, n_iter.
        """
        if k > self.dim:
            raise ValueError(f"Dependent block bigger than dim: k={k}>{self.dim}=d")

        n, _ = X.shape
        # Section 3.1:  pseudo-copula observations
        U = self._pseudo_observations(X)
        Y = self._normal_scores(U, eps)

        corr = self._estimate_correlation(Y)
        corr = self._nearest_psd_corr(corr, threshold=1e-12)
        Lambda, W = self._pca_decomposition(corr)
        P = Y @ W
        pk = P[:, k - 1]

        skew_sign = np.sign(np.mean(pk**3))
        nu1_0 = 8.0
        gamma1_0 = -0.3 * skew_sign
        nu_rest_0 = 10.0

        # unconstrained parameterisation:
        #   shape_vec = [eta1, gamma1_1, ..., gamma1_k, eta_rest]
        gamma1_0_vec = np.full(k, gamma1_0)
        prev_shape = np.concatenate(
            [[np.log(nu1_0 - 4.0)], gamma1_0_vec, [np.log(nu_rest_0 - 2.0)]]
        )
        rho_prev = corr.copy()
        converged = False
        nll_prev = 1e12
        best_nll = 1e12
        best_shape = prev_shape.copy()
        best_W = W.copy()
        best_Lambda = Lambda.copy()
        W_new, Lambda_new = W.copy(), Lambda.copy()
        cur_shape = prev_shape.copy()
        for it in tqdm_module.tqdm(
            range(self.max_iters), desc=f"Initialising GMM ({self.type}) algorithm:"
        ):
            eta1 = prev_shape[0]
            gamma1 = prev_shape[1 : 1 + k]
            eta_rest = prev_shape[1 + k]
            nu1 = 4.0 + np.exp(eta1)
            nu_rest = 2.0 + np.exp(eta_rest)

            #  Step 2(a):  F^{-1}_{Y_i}(U) -> Y,  then update rho
            sp = self._skewt_constrained_params(nu1, gamma1, Lambda[:k])
            if np.any(sp["Sigma11"] <= 1e-10):
                gamma1 = np.zeros(k)
                sp = self._skewt_constrained_params(nu1, gamma1, Lambda[:k])
            Sigma_diag = self._diag_scale_t(nu_rest, Lambda[k:])
            t_params = {**sp, "nu_rest": nu_rest, "Sigma_diag": Sigma_diag}

            Y = np.empty((n, self.dim))
            for i in range(self.dim):
                cf_Yi = self._make_marginal_cf_t(i, W, t_params, k)
                inv_i = self._build_inverse_cdf(cf_Yi)
                Y[:, i] = inv_i(U[:, i])

            if np.any(np.std(Y, axis=0) < 1e-6):
                rho_new = rho_prev.copy()
                Lambda_new, W_new = Lambda.copy(), W.copy()
            else:
                rho_new = self._estimate_correlation(Y)
                rho_new = self._nearest_psd_corr(rho_new, threshold=1e-12)
                Lambda_new, W_new = self._pca_decomposition(rho_new)

            # Step 2(c): Shape Parameters
            res = minimize(
                self._neg_copula_ll_shape_t,
                x0=prev_shape,
                args=(U, W_new, Lambda_new, a, b, Nc, n_grid, k),
                method="L-BFGS-B",
                options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-5},
            )

            cur_shape = res.x

            if res.fun < best_nll:
                best_nll = res.fun
                best_shape = cur_shape.copy()
                best_W = W_new.copy()
                best_Lambda = Lambda_new.copy()

            if (
                np.abs((res.fun - nll_prev) / max(abs(nll_prev), 1e-8)) < ftol
                and res.fun < 1e6
            ):
                converged = True
                break

            # Update Stopping Criterion
            nll_prev = res.fun
            rho_prev = rho_new

            prev_shape = cur_shape.copy()
            W, Lambda = W_new, Lambda_new

        if converged:
            logger.info("Algorithm 1 (Skew-t) converged.")

        #  final quantities: prefer best-seen state if final is degenerate
        if best_nll < 1e6:
            W, Lambda = best_W.copy(), best_Lambda.copy()
            cur_shape = best_shape.copy()
        else:
            W, Lambda = W_new.copy(), Lambda_new.copy()
        eta1_f = cur_shape[0]
        gamma1_f = cur_shape[1 : 1 + k]
        eta_rest_f = cur_shape[1 + k]
        nu1_f = 4.0 + np.exp(eta1_f)
        nu_rest_f = 2.0 + np.exp(eta_rest_f)

        sp_final = self._skewt_constrained_params(nu1_f, gamma1_f, Lambda[:k])
        Sigma_diag_f = self._diag_scale_t(nu_rest_f, Lambda[k:])
        t_params_final = {
            **sp_final,
            "nu_rest": nu_rest_f,
            "Sigma_diag": Sigma_diag_f,
        }

        ll_final, _, _ = self._compute_copula_ll_t(
            U, W, Lambda, t_params_final, a, b, Nc, n_grid, k
        )
        rho_Y = W @ np.diag(Lambda) @ W.T

        self.params = t_params_final
        self.W = W
        self.Lambda = Lambda
        self.loglikelihood = ll_final

        return dict(
            parameters=t_params_final,
            W=W,
            Lambda=Lambda,
            rho_Y=rho_Y,
            log_likelihood=ll_final,
            converged=converged,
            n_iter=it + 1,
        )

    def fit_t_mle(self, X: np.ndarray):
        """Full MLE for the Skew-t1 / t_{d-1} PCC (not yet implemented)."""
        raise NotImplementedError(
            "Full MLE requires the COS-method for marginal densities (Eqs. 20-22 "
            "of Gubbels et al., 2025).  Use method='GMM' instead."
        )

    # Cross-asset PCC  --  parameter helpers & log-likelihood

    def _per_pc_params_from_shape(
        self, shape_vec: np.ndarray, Lambda: np.ndarray, K: int
    ) -> Optional[List[Dict]]:
        """
        Unpack shape_vec [eta_1, gamma_1, ..., eta_K, gamma_K] into K per-PC
        parameter dicts {nu, gamma, Sigma, mu} (§6).

        Reparametrisation: nu_j = 4 + exp(clip(eta_j, -10, 10)).

        Returns None if any Sigma_jj is infeasible (<= 1e-10).
        """
        params: List[Dict] = []
        for j in range(K):
            eta_j = shape_vec[2 * j]
            gamma_j = shape_vec[2 * j + 1]
            nu_j = 4.0 + np.exp(np.clip(eta_j, -10.0, 5.0))
            sp = self._skewt_constrained_params(
                nu_j, np.array([gamma_j]), np.array([Lambda[j]])
            )
            if sp["Sigma11"][0] <= 1e-10:
                return None
            params.append(
                {
                    "nu": nu_j,
                    "gamma": gamma_j,
                    "Sigma": float(sp["Sigma11"][0]),
                    "mu": float(sp["mu1"][0]),
                }
            )
        return params

    def _neg_copula_ll_cross(
        self,
        shape_vec: np.ndarray,
        U: np.ndarray,
        W: np.ndarray,
        Lambda: np.ndarray,
        K: int,
        a: float,
        b: float,
        Nc: int,
        n_grid: int,
    ) -> float:
        """
        Negative copula log-likelihood as a function of shape_vec (§6).
        Fully analytic — no Monte Carlo.
        """
        per_pc = self._per_pc_params_from_shape(shape_vec, Lambda, K)
        if per_pc is None:
            # Smooth quadratic barrier so L-BFGS-B gets useful gradient near
            # the Sigma11>0 feasibility boundary instead of a hard cliff.
            penalty = 0.0
            for j in range(K):
                nu_j = 4.0 + np.exp(np.clip(shape_vec[2 * j], -10.0, 10.0))
                sp = self._skewt_constrained_params(
                    nu_j, np.array([shape_vec[2 * j + 1]]), np.array([Lambda[j]])
                )
                sig = float(sp["Sigma11"][0])
                if sig <= 1e-10:
                    penalty += (1e-10 - sig) ** 2
            return 1e6 * (1.0 + penalty)
        try:
            ll, _, _ = self._compute_copula_ll_cross(
                U, W, Lambda, per_pc, K, a, b, Nc, n_grid
            )
        except Exception:
            return 1e12
        return -ll if np.isfinite(ll) else 1e12

    def _compute_copula_ll_cross(
        self,
        U: np.ndarray,
        W: np.ndarray,
        Lambda: np.ndarray,
        per_pc_params: List[Dict],
        K: int,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
        n_grid: int = 2000,
    ) -> Tuple[float, np.ndarray, List[Callable]]:
        """
        Copula log-likelihood for the cross-asset PCC (§4, Eq. 18).

            l = sum_t [ log f_Y(Y_t) - sum_i log f_{Y_i}(Y_{i,t}) ]

        Numerator vine block   : K independent 1-D Skew-t log-densities.
        Numerator noise block  : Gaussian log-densities for j >= K.
        Denominator            : COS density (fully analytic — no MC).

        Reuses _build_generator_cfs_cross, _make_marginal_cf, _build_inverse_cdf,
               _cos_density, _skewt_mvt_logpdf, _normal_log_density.
        """
        n, d = U.shape

        # Analytic generator CFs (Case 1 — all independent)
        cfs = self._build_generator_cfs_cross(Lambda, per_pc_params, K)

        inv_cdfs: List[Callable] = []
        marginal_cfs: List[Callable] = []
        for i in range(d):
            cf_i = self._make_marginal_cf(i, W, cfs)
            marginal_cfs.append(cf_i)
            inv_cdfs.append(self._build_inverse_cdf(cf_i, a, b, Nc, n_grid))

        Y = np.empty_like(U)
        for i in range(d):
            Y[:, i] = inv_cdfs[i](U[:, i])

        P = Y @ W  # (n, d)  PC scores

        # Vine block: K independent 1-D Skew-t log-densities
        log_fY = np.zeros(n)
        for j in range(K):
            p = per_pc_params[j]
            log_fY += self._skewt_mvt_logpdf(
                P[:, j : j + 1],
                p["nu"],
                np.array([p["gamma"]]),
                np.array([p["Sigma"]]),
                np.array([p["mu"]]),
            )

        # Noise block: Gaussian
        for j in range(K, d):
            log_fY += self._normal_log_density(P[:, j], Lambda[j])

        # Denominator: sum_i log f_{Y_i}(Y_{i,t}) via COS density
        log_marg_sum = np.zeros(n)
        for i in range(d):
            dens_i = self._cos_density(Y[:, i], marginal_cfs[i], a, b, Nc)
            log_marg_sum += np.log(dens_i)

        ll = float(np.sum(log_fY - log_marg_sum))
        return ll, Y, inv_cdfs

    # Cross-asset PCC (dependent K-block) -- CF, LL, neg-LL helpers

    def _make_marginal_cf_cross_dep(
        self,
        i: int,
        W: np.ndarray,
        t_params_dep: Dict,
        k: int,
        Lambda: np.ndarray,
    ) -> Callable:
        """
        Marginal CF for Y_i under the *dependent* cross-asset PCC.

        The K leading PCs share one IG(nu1/2, nu1/2) mixing variable
        (Lemma 1, Case 2 applied to the K-block).  Higher PCs j >= K are
        independent Gaussians N(0, Lambda_j).

        Projecting the K-dim shared-mixing block onto the scalar direction
        w_{i, 1:K} yields a 1-D GH skew-t with effective parameters
        (same algebra as _make_marginal_cf_t, §6):

            gamma_eff    = w_{i,1:K} @ gamma1
            mu_eff       = w_{i,1:K} @ mu1
            sigma_sq_eff = sum_j  w_{i,j}^2  Sigma11_j    (j = 0..K-1)
            gauss_var    = sum_j  w_{i,j}^2  Lambda_j     (j = K..d-1)

        phi_{Y_i}(t) = phi_{skewt-1d-eff}(t) * phi_{N(0, gauss_var)}(t)
        """
        w_i = W[i, :]
        nu1 = t_params_dep["nu1"]
        gamma1 = np.asarray(t_params_dep["gamma1"])
        Sigma11 = np.asarray(t_params_dep["Sigma11"])
        mu1 = np.asarray(t_params_dep["mu1"])

        gamma_eff = float(np.dot(w_i[:k], gamma1))
        mu_eff = float(np.dot(w_i[:k], mu1))
        sigma_sq_eff = float(np.dot(w_i[:k] ** 2, Sigma11))
        gauss_var = float(np.dot(w_i[k:] ** 2, Lambda[k:]))

        def cf_Yi(t_arg):
            t_arg = np.asarray(t_arg, dtype=complex)
            part1 = self._cf_skewt(t_arg, nu1, gamma_eff, sigma_sq_eff, mu_eff)
            part2 = self._cf_normal(t_arg, gauss_var)
            return part1 * part2

        return cf_Yi

    def _compute_copula_ll_cross_dep(
        self,
        U: np.ndarray,
        W: np.ndarray,
        Lambda: np.ndarray,
        t_params_dep: Dict,
        k: int,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
        n_grid: int = 2000,
    ) -> Tuple[float, np.ndarray, List[Callable]]:
        """
        Copula log-likelihood for the *dependent* cross-asset PCC.

        Numerator (Eq. 17):
            f_Y(y) = f_{K-dim-skewt}(P_{1:K}) * prod_{j>=K} f_{N(0,Lambda_j)}(P_j)
        Denominator: COS density (analytic, no MC).

        The K-block density uses _skewt_mvt_logpdf; the Gaussian block reuses
        _normal_log_density (both already defined for the t-PCC, §3).
        """
        n, d = U.shape

        inv_cdfs: List[Callable] = []
        marginal_cfs: List[Callable] = []
        for i in range(d):
            cf_i = self._make_marginal_cf_cross_dep(i, W, t_params_dep, k, Lambda)
            marginal_cfs.append(cf_i)
            inv_cdfs.append(self._build_inverse_cdf(cf_i, a, b, Nc, n_grid))

        Y = np.empty_like(U)
        for i in range(d):
            Y[:, i] = inv_cdfs[i](U[:, i])

        P = Y @ W  # (n, d) PC scores

        # K-block: K-dim joint GH skew-t (shared mixing variable)
        log_fY = self._skewt_mvt_logpdf(
            P[:, :k],
            t_params_dep["nu1"],
            t_params_dep["gamma1"],
            t_params_dep["Sigma11"],
            t_params_dep["mu1"],
        )
        # Noise block: independent Gaussians
        for j in range(k, d):
            log_fY += self._normal_log_density(P[:, j], Lambda[j])

        log_marg_sum = np.zeros(n)
        for i in range(d):
            dens_i = self._cos_density(Y[:, i], marginal_cfs[i], a, b, Nc)
            log_marg_sum += np.log(dens_i)

        ll = float(np.sum(log_fY - log_marg_sum))
        return ll, Y, inv_cdfs

    def _neg_copula_ll_cross_dep(
        self,
        shape_vec: np.ndarray,
        U: np.ndarray,
        W: np.ndarray,
        Lambda: np.ndarray,
        k: int,
        a: float,
        b: float,
        Nc: int,
        n_grid: int,
    ) -> float:
        """
        Negative copula log-likelihood for the dependent cross-asset PCC.

        shape_vec = [eta1, gamma1_1, ..., gamma1_k]  (length 1+k)
        Reparametrisation: nu1 = 4 + exp(clip(eta1, -10, 10)).
        """
        eta1 = shape_vec[0]
        gamma1 = shape_vec[1 : 1 + k]
        nu1 = 4.0 + np.exp(np.clip(eta1, -10.0, 10.0))

        try:
            sp = self._skewt_constrained_params(nu1, gamma1, Lambda[:k])
        except Exception:
            return 1e12

        if np.any(sp["Sigma11"] <= 1e-10):
            # Smooth barrier: quadratic penalty proportional to infeasibility depth
            penalty = float(np.sum(np.maximum(0.0, 1e-10 - sp["Sigma11"]) ** 2))
            return 1e6 * (1.0 + penalty)

        try:
            ll, _, _ = self._compute_copula_ll_cross_dep(
                U, W, Lambda, sp, k, a, b, Nc, n_grid
            )
        except Exception:
            return 1e12

        return -ll if np.isfinite(ll) else 1e12

    def fit_cross_gmm(
        self,
        X: np.ndarray,
        k: int = 1,
        dependent: bool = False,
        eps: float = 1e-12,
        tol: float = 1e-6,
        ftol: float = 1e-4,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
        n_grid: int = 2000,
    ):
        """
        Fit the cross-asset PCC via Algorithm 1 (§5).

        Two generator structures are supported, selected via *dependent*:

        dependent=False  (Lemma 1 Case 1, default):
            Each of the K leading PCs has its own *independent* IG mixing
            variable.  Generators are mutually independent; marginal CFs are
            a product of 1-D CFs (Eq. 12).
            Shape vector: [eta_1, gamma_1, ..., eta_K, gamma_K]  (length 2K).

        dependent=True  (Lemma 1 Case 2 for the K-block):
            The K leading PCs share a *single* IG(nu1/2, nu1/2) mixing
            variable (Eq. 9).  This induces dependence within the K-block
            analogous to fit_t_gmm, but higher PCs j>K remain Gaussian
            (not t).  Marginal CF uses _make_marginal_cf_cross_dep.
            Shape vector: [eta1, gamma1_1, ..., gamma1_k]  (length 1+k).

        In both cases, higher PCs j > K are Gaussian N(0, Lambda_j).
        Reparametrisation: nu_j = 4 + exp(eta_j)  (ensures nu > 4).

        Parameters
        --
        X         : (n, d) array of raw observations
        k         : number of leading PCs with Skew-t distribution
        dependent : bool  -- see above
        eps, tol, ftol, a, b, Nc, n_grid : see fit_t_gmm

        Returns
        ---
        dict  keys: parameters, W, Lambda, rho_Y, log_likelihood, converged, n_iter
        """
        if k > self.dim:
            raise ValueError(f"Generator block bigger than dim: k={k}>{self.dim}=d")

        n, _ = X.shape
        U = self._pseudo_observations(X)
        Y = self._normal_scores(U, eps)

        corr = self._estimate_correlation(Y)
        corr = self._nearest_psd_corr(corr, threshold=1e-12)
        Lambda, W = self._pca_decomposition(corr)
        P = Y @ W

        if not dependent:
            # Independent cross-asset PCC (Case 1): shape_vec length 2K
            shape_vec_0 = np.zeros(2 * k)
            for j in range(k):
                nu_j_fit = max(float(t.fit(P[:, j], floc=0)[0]), 4.1)
                eta_j = np.clip(np.log(nu_j_fit - 4.0), -10.0, 5.0)
                nu_j = 4.0 + np.exp(eta_j)
                gamma_sign = float(np.sign(np.mean(P[:, j] ** 3)))
                # Ensure initial gamma keeps Sigma_jj = (nu-2)/nu*Lambda - 2nu/((nu-2)(nu-4))*gamma^2 > 0
                gamma_sq_max = (
                    (nu_j - 2.0) ** 2 * (nu_j - 4.0) / (2.0 * nu_j**2) * Lambda[j]
                )
                gamma_cap = 0.9 * np.sqrt(max(gamma_sq_max, 0.0))
                shape_vec_0[2 * j] = eta_j
                shape_vec_0[2 * j + 1] = gamma_sign * min(0.3, gamma_cap)

            prev_shape = shape_vec_0.copy()
            rho_prev = corr.copy()
            W_new, Lambda_new = W.copy(), Lambda.copy()
            converged = False
            nll_prev = 1e12
            best_nll = 1e12
            best_shape = prev_shape.copy()
            best_W = W.copy()
            best_Lambda = Lambda.copy()
            it = 0
            per_pc = None  # last valid per_pc for fallback

            for it in tqdm_module.tqdm(
                range(self.max_iters),
                desc=f"Initialising GMM ({self.type}, independent) algorithm:",
            ):
                per_pc = self._per_pc_params_from_shape(prev_shape, Lambda, k)
                if per_pc is None:
                    break
                cfs = self._build_generator_cfs_cross(Lambda, per_pc, k)

                Y = np.empty((n, self.dim))
                for i in range(self.dim):
                    cf_Yi = self._make_marginal_cf(i, W, cfs)
                    inv_i = self._build_inverse_cdf(cf_Yi, a, b, Nc, n_grid)
                    Y[:, i] = inv_i(U[:, i])

                if np.any(np.std(Y, axis=0) < 1e-6):
                    rho_new = rho_prev.copy()
                    Lambda_new, W_new = Lambda.copy(), W.copy()
                else:
                    rho_raw = self._estimate_correlation(Y)
                    # Dampen the correlation update to stabilise the outer loop.
                    # K independent mixing variables create competing feedback
                    # between shape and correlation that can cause oscillation.
                    rho_new = self._nearest_psd_corr(
                        0.5 * rho_raw + 0.5 * rho_prev, threshold=1e-12
                    )
                    Lambda_new, W_new = self._pca_decomposition(rho_new)

                # Alternate [eta_j, gamma_j] pairs; bound eta to keep nu < 4+e^5 ≈ 152
                # and avoid Bessel/Gamma overflow in _cf_skewt for large nu.
                _bnd = [(-10.0, 5.0), (None, None)] * k
                res = minimize(
                    self._neg_copula_ll_cross,
                    x0=np.clip(prev_shape, -10.0, 5.0),
                    args=(U, W_new, Lambda_new, k, a, b, Nc, n_grid),
                    method="L-BFGS-B",
                    bounds=_bnd,
                    options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-5},
                )

                cur_shape = res.x

                if res.fun < best_nll:
                    best_nll = res.fun
                    best_shape = cur_shape.copy()
                    best_W = W_new.copy()
                    best_Lambda = Lambda_new.copy()

                if (
                    it >= 2
                    and np.abs((res.fun - nll_prev) / max(abs(nll_prev), 1e-8)) < ftol
                    and res.fun < 1e6
                ):
                    converged = True
                    prev_shape = best_shape.copy()
                    W, Lambda = best_W, best_Lambda
                    break

                nll_prev = res.fun
                rho_prev = rho_new.copy()
                prev_shape = cur_shape.copy()
                W, Lambda = W_new, Lambda_new

            if converged:
                logger.info("Algorithm 1 (cross-asset PCC, independent) converged.")

            if best_nll < 1e6:
                W, Lambda = best_W.copy(), best_Lambda.copy()
                prev_shape = best_shape.copy()
            else:
                W, Lambda = W_new.copy(), Lambda_new.copy()
            per_pc_final = self._per_pc_params_from_shape(prev_shape, Lambda, k)
            if per_pc_final is None:
                per_pc_final = per_pc

            ll_final, _, _ = self._compute_copula_ll_cross(
                U, W, Lambda, per_pc_final, k, a, b, Nc, n_grid
            )
            rho_Y = W @ np.diag(Lambda) @ W.T

            self.params = {"per_pc": per_pc_final, "K": k, "dependent": False}
            self.W = W
            self.Lambda = Lambda
            self.loglikelihood = ll_final

            return dict(
                parameters={"per_pc": per_pc_final, "K": k, "dependent": False},
                W=W,
                Lambda=Lambda,
                rho_Y=rho_Y,
                log_likelihood=ll_final,
                converged=converged,
                n_iter=it + 1,
            )

        else:
            # Dependent cross-asset PCC (Case 2): shape_vec length 1+k
            # K leading PCs share one IG mixing variable; j>K Gaussian.
            nu1_0 = max(float(t.fit(P[:, 0], floc=0)[0]), 4.1)
            eta1_0 = np.clip(np.log(nu1_0 - 4.0), -10.0, 5.0)
            nu1_0_eff = 4.0 + np.exp(eta1_0)
            gamma1_0 = np.array(
                [
                    float(np.sign(np.mean(P[:, j] ** 3)))
                    * min(
                        0.3,
                        0.9
                        * np.sqrt(
                            max(
                                (nu1_0_eff - 2.0) ** 2
                                * (nu1_0_eff - 4.0)
                                / (2.0 * nu1_0_eff**2)
                                * Lambda[j],
                                0.0,
                            )
                        ),
                    )
                    for j in range(k)
                ]
            )
            shape_vec_0 = np.concatenate([[eta1_0], gamma1_0])

            prev_shape = shape_vec_0.copy()
            rho_prev = corr.copy()
            W_new, Lambda_new = W.copy(), Lambda.copy()
            converged = False
            nll_prev = 1e12
            best_nll = 1e12
            best_shape = prev_shape.copy()
            best_W = W.copy()
            best_Lambda = Lambda.copy()
            it = 0
            t_params_dep_fb = None  # fallback

            for it in tqdm_module.tqdm(
                range(self.max_iters),
                desc=f"Initialising GMM ({self.type}, dependent) algorithm:",
            ):
                eta1 = prev_shape[0]
                gamma1 = prev_shape[1 : 1 + k]
                nu1 = 4.0 + np.exp(np.clip(eta1, -10.0, 5.0))
                sp = self._skewt_constrained_params(nu1, gamma1, Lambda[:k])

                if np.any(sp["Sigma11"] <= 1e-10):
                    gamma1 = np.zeros(k)
                    sp = self._skewt_constrained_params(nu1, gamma1, Lambda[:k])

                t_params_dep_fb = {**sp}

                Y = np.empty((n, self.dim))
                for i in range(self.dim):
                    cf_Yi = self._make_marginal_cf_cross_dep(
                        i, W, t_params_dep_fb, k, Lambda
                    )
                    inv_i = self._build_inverse_cdf(cf_Yi, a, b, Nc, n_grid)
                    Y[:, i] = inv_i(U[:, i])

                if np.any(np.std(Y, axis=0) < 1e-6):
                    rho_new = rho_prev.copy()
                    Lambda_new, W_new = Lambda.copy(), W.copy()
                else:
                    rho_raw = self._estimate_correlation(Y)
                    rho_new = self._nearest_psd_corr(
                        0.5 * rho_raw + 0.5 * rho_prev, threshold=1e-12
                    )
                    Lambda_new, W_new = self._pca_decomposition(rho_new)

                # [eta1, gamma1_1..k]; bound eta1 same as independent case.
                _bnd = [(-10.0, 5.0)] + [(None, None)] * k
                res = minimize(
                    self._neg_copula_ll_cross_dep,
                    x0=np.concatenate(
                        [[np.clip(prev_shape[0], -10.0, 5.0)], prev_shape[1:]]
                    ),
                    args=(U, W_new, Lambda_new, k, a, b, Nc, n_grid),
                    method="L-BFGS-B",
                    bounds=_bnd,
                    options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-5},
                )

                cur_shape = res.x

                if res.fun < best_nll:
                    best_nll = res.fun
                    best_shape = cur_shape.copy()
                    best_W = W_new.copy()
                    best_Lambda = Lambda_new.copy()

                if (
                    it >= 2
                    and np.abs((res.fun - nll_prev) / max(abs(nll_prev), 1e-8)) < ftol
                    and res.fun < 1e6
                ):
                    converged = True
                    prev_shape = best_shape.copy()
                    W, Lambda = best_W, best_Lambda
                    break

                nll_prev = res.fun
                rho_prev = rho_new.copy()
                prev_shape = cur_shape.copy()
                W, Lambda = W_new, Lambda_new

            if converged:
                logger.info("Algorithm 1 (cross-asset PCC, dependent) converged.")

            if best_nll < 1e6:
                W, Lambda = best_W.copy(), best_Lambda.copy()
                prev_shape = best_shape.copy()
            else:
                W, Lambda = W_new.copy(), Lambda_new.copy()
            eta1_f = prev_shape[0]
            gamma1_f = prev_shape[1 : 1 + k]
            nu1_f = 4.0 + np.exp(np.clip(eta1_f, -10.0, 10.0))
            sp_final = self._skewt_constrained_params(nu1_f, gamma1_f, Lambda[:k])

            if np.any(sp_final["Sigma11"] <= 1e-10) and t_params_dep_fb is not None:
                sp_final = t_params_dep_fb

            ll_final, _, _ = self._compute_copula_ll_cross_dep(
                U, W, Lambda, sp_final, k, a, b, Nc, n_grid
            )
            rho_Y = W @ np.diag(Lambda) @ W.T

            self.params = {"t_params_dep": sp_final, "K": k, "dependent": True}
            self.W = W
            self.Lambda = Lambda
            self.loglikelihood = ll_final

            return dict(
                parameters={"t_params_dep": sp_final, "K": k, "dependent": True},
                W=W,
                Lambda=Lambda,
                rho_Y=rho_Y,
                log_likelihood=ll_final,
                converged=converged,
                n_iter=it + 1,
            )

    def fit_cross_mle(self, X: np.ndarray):
        """Full MLE for the Skew-t1*k-Normal PCC (not yet implemented)."""
        raise NotImplementedError(
            "Full MLE requires the COS-method for marginal densities (Eqs. 20-22 "
            "of Gubbels et al., 2025).  Use method='GMM' instead."
        )

    # Per-observation log-density (for Vuong test, GOF)

    def logpdf_obs(
        self,
        U: np.ndarray,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
        n_grid: int = 2000,
    ) -> np.ndarray:
        """
        Per-observation copula log-density. Shape (n,).

        Replicates the internal LL computation of each type without modifying
        the existing fit machinery. Used for Vuong (1989) test and
        Genest-Rémillard-Beaudoin (2009) GOF statistic.
        """
        if self.W is None or self.params is None:
            raise RuntimeError("Fit model before calling logpdf_obs.")
        U = np.asarray(U, dtype=float)
        if U.ndim != 2:
            raise ValueError(f"U must be a 2-D array; got ndim={U.ndim}.")
        if U.shape[1] != self.dim:
            raise ValueError(
                f"U has {U.shape[1]} columns but the model dimension is {self.dim}."
            )
        if np.any(U <= 0.0) or np.any(U >= 1.0):
            raise ValueError("U must lie strictly inside (0, 1).")
        n, d = U.shape
        W, Lambda = self.W, self.Lambda
        k = self.k

        if self.type == "normal":
            gp = self.params
            cfs = self._build_generator_cfs(Lambda, gp)
            inv_cdfs, marginal_cfs = [], []
            for i in range(d):
                cf_i = self._make_marginal_cf(i, W, cfs)
                marginal_cfs.append(cf_i)
                inv_cdfs.append(self._build_inverse_cdf(cf_i, a, b, Nc, n_grid))
            Y = np.column_stack([inv_cdfs[i](U[:, i]) for i in range(d)])
            P = Y @ W
            log_fY = self._gh_1d_log_density(
                P[:, 0],
                gp["lam"],
                gp["chi"],
                gp["psi"],
                gp["mu"],
                gp["alpha_bar"],
                gp["beta_bar"],
            )
            for j in range(1, d):
                log_fY += self._normal_log_density(P[:, j], Lambda[j])

        elif self.type == "t":
            tp = self.params
            inv_cdfs, marginal_cfs = [], []
            for i in range(d):
                cf_i = self._make_marginal_cf_t(i, W, tp, k)
                marginal_cfs.append(cf_i)
                inv_cdfs.append(self._build_inverse_cdf(cf_i, a, b, Nc, n_grid))
            Y = np.column_stack([inv_cdfs[i](U[:, i]) for i in range(d)])
            P = Y @ W
            log_fY = self._skewt_mvt_logpdf(
                P[:, :k],
                tp["nu1"],
                tp["gamma1"],
                tp["Sigma11"],
                tp["mu1"],
            )
            log_fY += self._mvt_diag_logpdf(P[:, k:], tp["nu_rest"], tp["Sigma_diag"])

        elif self.type == "cross" and not self.params.get("dependent", True):
            # independent cross-asset PCC  (params = {"per_pc": [...], ...})
            pp = self.params["per_pc"]
            cfs = self._build_generator_cfs_cross(Lambda, pp, k)
            inv_cdfs, marginal_cfs = [], []
            for i in range(d):
                cf_i = self._make_marginal_cf(i, W, cfs)
                marginal_cfs.append(cf_i)
                inv_cdfs.append(self._build_inverse_cdf(cf_i, a, b, Nc, n_grid))
            Y = np.column_stack([inv_cdfs[i](U[:, i]) for i in range(d)])
            P = Y @ W
            log_fY = np.zeros(n)
            for j in range(k):
                p = pp[j]
                log_fY += self._skewt_mvt_logpdf(
                    P[:, j : j + 1],
                    p["nu"],
                    np.array([p["gamma"]]),
                    np.array([p["Sigma"]]),
                    np.array([p["mu"]]),
                )
            for j in range(k, d):
                log_fY += self._normal_log_density(P[:, j], Lambda[j])

        else:  # dependent cross-asset PCC  (params = {"t_params_dep": {...}, ...})
            tp = self.params["t_params_dep"]
            inv_cdfs, marginal_cfs = [], []
            for i in range(d):
                cf_i = self._make_marginal_cf_cross_dep(i, W, tp, k, Lambda)
                marginal_cfs.append(cf_i)
                inv_cdfs.append(self._build_inverse_cdf(cf_i, a, b, Nc, n_grid))
            Y = np.column_stack([inv_cdfs[i](U[:, i]) for i in range(d)])
            P = Y @ W
            log_fY = self._skewt_mvt_logpdf(
                P[:, :k],
                tp["nu1"],
                tp["gamma1"],
                tp["Sigma11"],
                tp["mu1"],
            )
            for j in range(k, d):
                log_fY += self._normal_log_density(P[:, j], Lambda[j])

        log_marg_sum = np.zeros(n)
        for i in range(d):
            dens_i = self._cos_density(Y[:, i], marginal_cfs[i], a, b, Nc)
            log_marg_sum += np.log(np.maximum(dens_i, 1e-300))

        return log_fY - log_marg_sum

    def _marginal_cfs(self) -> List[Callable]:
        """
        Build the list of d marginal characteristic functions phi_{Y_i} for the
        fitted model, dispatching on ``self.type`` exactly as ``logpdf_obs`` and
        the simulators do.  Requires a fitted model.
        """
        if self.W is None or self.params is None:
            raise RuntimeError("Fit model before building marginal CFs.")
        W, Lambda, k, d = self.W, self.Lambda, self.k, self.dim

        if self.type == "normal":
            cfs = self._build_generator_cfs(Lambda, self.params)
            return [self._make_marginal_cf(i, W, cfs) for i in range(d)]
        if self.type == "t":
            tp = self.params
            return [self._make_marginal_cf_t(i, W, tp, k) for i in range(d)]
        if self.type == "cross" and not self.params.get("dependent", True):
            cfs = self._build_generator_cfs_cross(Lambda, self.params["per_pc"], k)
            return [self._make_marginal_cf(i, W, cfs) for i in range(d)]
        # dependent cross-asset PCC
        tp = self.params["t_params_dep"]
        return [self._make_marginal_cf_cross_dep(i, W, tp, k, Lambda) for i in range(d)]

    def cos_truncation_report(
        self,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
        n_grid: int = 2000,
        tol: float = 1e-3,
    ) -> Dict:
        """
        Diagnostics for the COS-method truncation of the fitted marginals.

        For each marginal Y_i the COS-CDF F_i is built from its characteristic
        function and checked at the truncation bounds: a well-truncated F should
        satisfy  F_i(a) ~ 0  and  F_i(b) ~ 1.  The per-margin tail-mass
        residuals are

            left_tail  = F_i(a)            (mass below the lower bound)
            right_tail = 1 - F_i(b)        (mass above the upper bound)

        Large residuals indicate the interval [a, b] is too narrow (or Nc too
        small) for that marginal.  A warning is issued if any residual exceeds
        ``tol`` (~1e-3 by default).  Requires a fitted model.

        Parameters
        --
        a, b   : float   COS truncation interval (should match the fit).
        Nc     : int     number of COS cosine terms.
        n_grid : int     grid resolution used to evaluate the COS-CDF.
        tol    : float   residual threshold above which a warning is raised.

        Returns
        ---
        dict with keys
            ``a``, ``b``, ``Nc``, ``tol``,
            ``left_tail``  : (d,) array of F_i(a),
            ``right_tail`` : (d,) array of 1 - F_i(b),
            ``max_residual`` : float, the worst residual over all margins,
            ``ok``         : bool, True if max_residual <= tol.
        """
        if self.W is None or self.params is None:
            raise RuntimeError("Fit model before calling cos_truncation_report.")

        d = self.dim
        marginal_cfs = self._marginal_cfs()

        # Evaluate each COS-CDF on a fine, monotonised grid and read off F(a),
        # F(b) at the truncation bounds.
        y_grid = np.linspace(a, b, n_grid)
        left_tail = np.empty(d)
        right_tail = np.empty(d)
        for i in range(d):
            cdf = self._cos_cdf(y_grid, marginal_cfs[i], a, b, Nc)
            cdf = np.maximum.accumulate(cdf)
            left_tail[i] = float(cdf[0])
            right_tail[i] = float(1.0 - cdf[-1])

        residuals = np.concatenate([np.abs(left_tail), np.abs(right_tail)])
        max_residual = float(np.max(residuals)) if residuals.size else 0.0
        ok = max_residual <= tol

        if not ok:
            warnings.warn(
                f"COS truncation tail mass exceeds tol={tol:g} "
                f"(max residual {max_residual:g}); widen [a, b] or raise Nc.",
                RuntimeWarning,
            )

        return {
            "a": a,
            "b": b,
            "Nc": Nc,
            "tol": tol,
            "left_tail": left_tail,
            "right_tail": right_tail,
            "max_residual": max_residual,
            "ok": ok,
        }

    # Parameter uncertainty  --  Wald confidence intervals

    @staticmethod
    def _numerical_hessian(
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        rel_step: float = 1e-3,
        abs_step: float = 1e-5,
    ) -> np.ndarray:
        """
        Central finite-difference Hessian of a scalar function f at x.

            H_ii = [f(x+h e_i) - 2 f(x) + f(x-h e_i)] / h_i^2
            H_ij = [f(x+h_i e_i +h_j e_j) - f(x+h_i e_i -h_j e_j)
                    - f(x-h_i e_i +h_j e_j) + f(x-h_i e_i -h_j e_j)]
                   / (4 h_i h_j)

        A per-coordinate step  h_i = rel_step * max(|x_i|, 1) + abs_step  keeps
        the perturbation well scaled.  ``rel_step`` is deliberately on the
        coarse side (1e-3) because the COS-method marginals are evaluated on a
        finite grid and carry ~1e-6 numerical noise; too small a step would let
        that noise dominate the second difference.
        """
        x = np.asarray(x, dtype=float)
        p = x.size
        h = rel_step * np.maximum(np.abs(x), 1.0) + abs_step
        H = np.zeros((p, p))
        f0 = float(f(x))

        # diagonal
        for i in range(p):
            xp = x.copy()
            xp[i] += h[i]
            xm = x.copy()
            xm[i] -= h[i]
            H[i, i] = (float(f(xp)) - 2.0 * f0 + float(f(xm))) / (h[i] ** 2)

        # off-diagonal (symmetric)
        for i in range(p):
            for j in range(i + 1, p):
                xpp = x.copy()
                xpp[i] += h[i]
                xpp[j] += h[j]
                xpm = x.copy()
                xpm[i] += h[i]
                xpm[j] -= h[j]
                xmp = x.copy()
                xmp[i] -= h[i]
                xmp[j] += h[j]
                xmm = x.copy()
                xmm[i] -= h[i]
                xmm[j] -= h[j]
                val = (
                    float(f(xpp)) - float(f(xpm)) - float(f(xmp)) + float(f(xmm))
                ) / (4.0 * h[i] * h[j])
                H[i, j] = H[j, i] = val

        return H

    def _shape_uncertainty_spec(
        self,
        U: np.ndarray,
        W: np.ndarray,
        Lambda: np.ndarray,
        k: int,
        a: float,
        b: float,
        Nc: int,
        n_grid: int,
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, Callable]:
        """
        Assemble, for the fitted PCC type, everything needed for the Wald
        covariance of the *shape* parameters:

        Returns
        ---
        theta   : (p,)  fitted shape vector on the **unconstrained** scale on
                  which the model is optimised (e.g. eta1 = log(nu1 - 4)).
        names   : list of p natural-parameter labels.
        est_nat : (p,)  fitted shape vector on the **natural** scale.
        jac     : (p,)  diagonal of  d(natural)/d(unconstrained)  (the
                  reparametrisation is coordinate-wise, so J is diagonal).
        objective : callable  s -> negative copula log-likelihood, evaluated
                  with W and Lambda held fixed at their fitted values.  This is
                  exactly the per-type objective minimised in step 2(c) of the
                  fit, so its Hessian at ``theta`` is the observed information.
        """
        t = self.type

        if t == "normal":
            gp = self.params
            lam_gh = gp["lam"]
            theta = np.array([float(gp["alpha_bar"]), float(gp["beta_bar"])])
            names = ["alpha_bar", "beta_bar"]
            est_nat = theta.copy()
            jac = np.ones(2)  # optimisation is directly on the natural params

            def objective(s):
                return self._neg_copula_ll_shape(
                    s, U, W, Lambda, lam_gh, a, b, Nc, n_grid
                )

            return theta, names, est_nat, jac, objective

        if t == "t":
            tp = self.params
            nu1 = float(tp["nu1"])
            gamma1 = np.asarray(tp["gamma1"], dtype=float).ravel()
            nu_rest = float(tp["nu_rest"])
            theta = np.concatenate(
                [[np.log(nu1 - 4.0)], gamma1, [np.log(nu_rest - 2.0)]]
            )
            names = ["nu1"] + [f"gamma1_{j + 1}" for j in range(k)] + ["nu_rest"]
            est_nat = np.concatenate([[nu1], gamma1, [nu_rest]])
            # nu1 = 4 + exp(eta1)  -> d nu1/d eta1 = nu1 - 4 ; gamma1 identity
            jac = np.concatenate([[nu1 - 4.0], np.ones(k), [nu_rest - 2.0]])

            def objective(s):
                return self._neg_copula_ll_shape_t(s, U, W, Lambda, a, b, Nc, n_grid, k)

            return theta, names, est_nat, jac, objective

        if t == "cross" and not self.params.get("dependent", False):
            per_pc = self.params["per_pc"]
            K = int(self.params.get("K", k))
            theta = np.empty(2 * K)
            est_nat = np.empty(2 * K)
            jac = np.empty(2 * K)
            names: List[str] = []
            for j in range(K):
                nu_j = float(per_pc[j]["nu"])
                g_j = float(per_pc[j]["gamma"])
                theta[2 * j] = np.log(nu_j - 4.0)
                theta[2 * j + 1] = g_j
                est_nat[2 * j] = nu_j
                est_nat[2 * j + 1] = g_j
                jac[2 * j] = nu_j - 4.0  # nu_j = 4 + exp(eta_j)
                jac[2 * j + 1] = 1.0
                names += [f"nu_{j + 1}", f"gamma_{j + 1}"]

            def objective(s):
                return self._neg_copula_ll_cross(s, U, W, Lambda, K, a, b, Nc, n_grid)

            return theta, names, est_nat, jac, objective

        if t == "cross":  # dependent K-block
            tp = self.params["t_params_dep"]
            nu1 = float(tp["nu1"])
            gamma1 = np.asarray(tp["gamma1"], dtype=float).ravel()
            theta = np.concatenate([[np.log(nu1 - 4.0)], gamma1])
            names = ["nu1"] + [f"gamma1_{j + 1}" for j in range(k)]
            est_nat = np.concatenate([[nu1], gamma1])
            jac = np.concatenate([[nu1 - 4.0], np.ones(k)])

            def objective(s):
                return self._neg_copula_ll_cross_dep(
                    s, U, W, Lambda, k, a, b, Nc, n_grid
                )

            return theta, names, est_nat, jac, objective

        raise ValueError(f"Unknown copula type: {t!r}")

    def parameter_uncertainty(
        self,
        X: np.ndarray,
        alpha: float = 0.05,
        a: float = -10.0,
        b: float = 10.0,
        Nc: int = 100,
        n_grid: int = 2000,
        rel_step: float = 1e-3,
        eps: float = 1e-12,
    ) -> Dict:
        """
        Wald confidence intervals for every free parameter of the fitted PCC.

        Handles all PCC families (``normal``, ``t``, ``cross`` independent and
        dependent) through a single objective: for each type the negative
        copula log-likelihood minimised in step 2(c) of the fit is rebuilt as a
        function of the *unconstrained* shape vector, with ``W`` and ``Lambda``
        held at their fitted values.

        Two sources of uncertainty are reported:

        1. **Shape parameters** (MLE).  The observed Fisher information is the
           Hessian of the negative log-likelihood at the optimum,
           ``I_n = -d^2 l / d theta d theta'``.  The asymptotic covariance is
           ``Cov(theta_hat) = I_n^{-1}`` (van der Vaart, 1998, Thm 5.39).  It is
           mapped from the unconstrained to the natural scale by the delta
           method, ``Cov(g) = J Cov(theta) J'`` with ``J`` diagonal.  The Wald
           interval is ``g_hat +/- z_{1-alpha/2} * SE(g_hat)``.

        2. **Covariance / correlation estimation.**  The d(d-1)/2 off-diagonal
           entries of ``rho_Y = W diag(Lambda) W'`` are sample correlations of
           the normal scores.  Their large-sample (delta-method) standard error
           is ``SE(r) = (1 - r^2) / sqrt(n)`` (Kendall & Stuart), giving the
           Wald interval ``r +/- z * SE(r)`` clipped to [-1, 1].

        The full result is stored on ``self.uncertainty`` and also returned.

        Parameters
        --
        X        : (n, d) array -- the same raw data used to fit the model;
                   rank-based pseudo-observations are rebuilt internally exactly
                   as in ``fit``.
        alpha    : float -- significance level (0.05 -> 95% intervals).
        a, b, Nc, n_grid : COS-method settings; should match those used to fit.
        rel_step : float -- relative step for the finite-difference Hessian.
        eps      : float -- clipping bound for the normal-score transform.

        Returns
        ---
        dict with keys
            ``type``, ``method`` ("wald"), ``alpha``, ``z``, ``n``,
            ``shape``       : per-shape-parameter block (names, estimate, se,
                              ci_lower, ci_upper, cov_natural,
                              cov_unconstrained, theta_unconstrained, hessian),
            ``correlation`` : per-correlation block (pairs, names, estimate, se,
                              ci_lower, ci_upper),
            ``table``       : flat concatenation of correlation + shape rows
                              (names, estimate, se, ci_lower, ci_upper).
        """
        if self.W is None or self.params is None:
            raise RuntimeError("Fit model before calling parameter_uncertainty.")

        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if d != self.dim:
            raise ValueError(
                f"X has {d} columns but the model dimension is {self.dim}."
            )

        # Rank-based pseudo-observations -- identical to the fit routines.
        U = self._pseudo_observations(X)

        W, Lambda, k = self.W, self.Lambda, self.k
        z = float(norm.ppf(1.0 - alpha / 2.0))

        #  1. Shape parameters: observed-information Wald CIs
        theta, names, est_nat, jac, objective = self._shape_uncertainty_spec(
            U, W, Lambda, k, a, b, Nc, n_grid
        )

        f0 = float(objective(theta))
        if not np.isfinite(f0) or f0 >= 1e6:
            warnings.warn(
                "Stored parameters evaluate to a penalised/infinite "
                "log-likelihood; the Wald covariance may be unreliable.",
                RuntimeWarning,
            )

        # Hessian of the NEGATIVE log-likelihood == observed Fisher information.
        H = self._numerical_hessian(objective, theta, rel_step=rel_step)
        H = 0.5 * (H + H.T)
        try:
            cov_unc = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov_unc = np.linalg.pinv(H)
        cov_unc = 0.5 * (cov_unc + cov_unc.T)

        # Delta method to the natural scale (J diagonal -> outer product form).
        cov_nat = np.outer(jac, jac) * cov_unc
        var_nat = np.diag(cov_nat)
        if np.any(var_nat <= 0) or not np.all(np.isfinite(var_nat)):
            warnings.warn(
                "Non-positive or non-finite variance from the Hessian inverse; "
                "the corresponding standard errors are set to NaN. Try adjusting "
                "rel_step.",
                RuntimeWarning,
            )
        se_nat = np.sqrt(np.where(var_nat > 0, var_nat, np.nan))
        ci_low = est_nat - z * se_nat
        ci_high = est_nat + z * se_nat

        shape_block = {
            "names": names,
            "estimate": est_nat,
            "se": se_nat,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "cov_natural": cov_nat,
            "cov_unconstrained": cov_unc,
            "theta_unconstrained": theta,
            "hessian": H,
        }

        #  2. Covariance estimation: Wald CIs for each correlation
        rho = W @ np.diag(Lambda) @ W.T
        iu, ju = np.triu_indices(d, k=1)
        r = rho[iu, ju]
        se_r = (1.0 - r**2) / np.sqrt(n)  # delta-method SE of a Pearson corr.
        r_low = np.clip(r - z * se_r, -1.0, 1.0)
        r_high = np.clip(r + z * se_r, -1.0, 1.0)
        pairs = list(zip(iu.tolist(), ju.tolist()))
        corr_names = [f"rho_{i + 1}_{j + 1}" for (i, j) in pairs]

        corr_block = {
            "pairs": pairs,
            "names": corr_names,
            "estimate": r,
            "se": se_r,
            "ci_lower": r_low,
            "ci_upper": r_high,
        }

        #  Flat combined table: correlations first, then shape params
        table = {
            "names": corr_names + names,
            "estimate": np.concatenate([r, est_nat]),
            "se": np.concatenate([se_r, se_nat]),
            "ci_lower": np.concatenate([r_low, ci_low]),
            "ci_upper": np.concatenate([r_high, ci_high]),
        }

        result = {
            "type": self.type,
            "method": "wald",
            "alpha": float(alpha),
            "z": z,
            "n": int(n),
            "shape": shape_block,
            "correlation": corr_block,
            "table": table,
        }

        self.uncertainty = result
        return result

    def parameter_uncertanity(self, *args, **kwargs):
        """Deprecated alias for :meth:`parameter_uncertainty` (kept for
        backward compatibility; the original method name was misspelled)."""
        warnings.warn(
            "parameter_uncertanity is deprecated; use parameter_uncertainty.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parameter_uncertainty(*args, **kwargs)

    # Simulation

    def simulate(
        self,
        n_samples: int,
        return_y: bool = False,
        return_p: bool = False,
        seed: int | np.random.Generator | None = 42,
    ) -> np.ndarray:
        """
        Simulate n_samples copula observations from the fitted PCC.

        Follows the data generating process of Fig. 3b (right -> left):
        1. Sample P from the fitted generator distributions.
        2. Rotate:  Y = W P  (row-wise: y_row = p_row @ W').
        3. Map to copula observations:  U_i = Phi(Y_i).

        Step 3 uses standard-normal marginals consistent with the pseudo-
        likelihood fitting scheme.  For exact copula marginals F_{Y_i}
        derived via characteristic functions (Eqs. 14-16 of the paper),
        replace Phi with the COS-method CDF.

        Parameters
        --
        n_samples : int
        return_y  : bool
            If True, also return the latent normal-score matrix Y.
        seed : int, np.random.Generator, or None
            Seed for reproducibility.  An integer creates a new Generator;
            a Generator is used directly; None (default) gives random results.

        Returns
        ---
        U : np.ndarray, shape (n_samples, d) -- copula samples in (0, 1)^d.
        Y : np.ndarray, shape (n_samples, d) -- only returned when return_y=True.

        Raises
        --
        RuntimeError  if the model has not been fitted yet.
        """
        if self.W is None or self.Lambda is None or self.params is None:
            raise RuntimeError("Fit distribution before simulating.")
        rng = np.random.default_rng(seed)
        if self.type == "normal":
            return self._simulate_normal(n_samples, return_y, return_p, rng)
        elif self.type == "cross":
            return self._simulate_cross(n_samples, return_y, return_p, rng)
        return self._simulate_t(n_samples, return_y, return_p, rng)

    def _simulate_normal(
        self, n_samples: int, return_y: bool, return_p: bool, rng: np.random.Generator
    ):
        """
        Simulate from the fitted Hyperbolic-Normal PCC.

            P1 ~ GH(lambda, alpha, beta, 0, sigma)     (first PC)
            Pj ~ N(0, Lambda_j)  for j > 1             (remaining PCs)

        Rotation:  Y = P W^T
        Marginals: U_i = F_{Y_i}(Y_i) computed via COS-CDF using the
                fitted marginal characteristic functions, exactly as
                in _simulate_t.
        """

        Lambda = self.Lambda
        W = self.W
        params = self.params
        d = self.dim

        #  First PC: GH distribution (already fitted in fit_normal_gmm)
        # params is the dict {lam, chi, psi, mu, beta_bar, alpha_bar} from
        # _gh_constrained_params; map it to scipy's positional convention
        # genhyperbolic.rvs(p, a, b, loc, scale) in the Sigma=1 convention
        # where natural alpha == alpha_bar and delta == sqrt(chi).
        chi = params["chi"]
        delta = np.sqrt(chi)
        P1 = genhyperbolic.rvs(
            params["lam"],
            params["alpha_bar"] * delta,
            params["beta_bar"] * delta,
            loc=params["mu"],
            scale=delta,
            size=n_samples,
            random_state=rng,
        )

        #  Higher PCs: Gaussian with variances Lambda_j (j > 1)
        P_rest = rng.normal(
            loc=0.0,
            scale=np.sqrt(Lambda[1:]),
            size=(n_samples, d - 1),
        )

        # Combine PCs
        P = np.column_stack([P1, P_rest])

        # Rotate back to Y-space
        Y = P @ W.T

        #  Compute U_i = F_{Y_i}(Y_i) using COS–CDF marginals
        U = np.empty_like(Y)

        cf_gens = self._build_generator_cfs(Lambda, params)

        for i in range(d):
            # build marginal CF for Y_i under the HB–Normal PCC
            cf_Yi = self._make_marginal_cf(i, W, cf_gens)

            # COS grid
            y_fine = np.linspace(-10.0, 10.0, 2000)

            # COS–CDF (monotonised)
            cdf_fine = self._cos_cdf(y_fine, cf_Yi)
            cdf_fine = np.maximum.accumulate(cdf_fine)

            # interpolation to evaluate U_i = F(Y_i)
            fwd = interp1d(
                y_fine,
                cdf_fine,
                kind="linear",
                bounds_error=False,
                fill_value=(0.0, 1.0),
            )

            U[:, i] = np.clip(fwd(Y[:, i]), 1e-10, 1 - 1e-10)
        if return_y and return_p:
            return (U, Y, P)
        elif return_y:
            return (U, Y)
        elif return_p:
            return (U, P)
        else:
            return U

    def _simulate_cross(
        self, n_samples: int, return_y: bool, return_p: bool, rng: np.random.Generator
    ):
        """
        Simulate from the fitted cross-asset PCC.

        dependent=False (Case 1):
            Each of the K leading PCs draws its own independent mixing variable:
                V_j ~ IG(nu_j/2, nu_j/2),  Z_j ~ N(0, 1)
                P_j = mu_j + gamma_j V_j + sqrt(Sigma_j V_j) Z_j

        dependent=True (Case 2 for the K-block):
            All K leading PCs share one mixing variable (Eq. 9):
                V   ~ IG(nu1/2, nu1/2)
                Z_j ~ N(0, 1)  independently
                P_j = mu1_j + gamma1_j V + sqrt(Sigma11_j V) Z_j

        In both cases, higher PCs j = K+1,...,d are N(0, Lambda_j).
        Marginals U_i = F_{Y_i}(Y_i) via COS-CDF (analytic, no MC).
        """
        k = self.params["K"]
        dep = self.params.get("dependent", False)
        Lambda = self.Lambda
        W = self.W
        d = self.dim

        P = np.empty((n_samples, d))

        if not dep:
            per_pc = self.params["per_pc"]
            for j in range(k):
                p = per_pc[j]
                V_j = invgamma.rvs(
                    a=p["nu"] / 2.0,
                    scale=p["nu"] / 2.0,
                    size=n_samples,
                    random_state=rng,
                )
                Z_j = rng.standard_normal(n_samples)
                P[:, j] = p["mu"] + p["gamma"] * V_j + np.sqrt(p["Sigma"] * V_j) * Z_j
        else:
            tp = self.params["t_params_dep"]
            nu1 = tp["nu1"]
            gamma1 = np.asarray(tp["gamma1"])
            Sigma11 = np.asarray(tp["Sigma11"])
            mu1 = np.asarray(tp["mu1"])
            # shared mixing variable for all K PCs
            V = invgamma.rvs(
                a=nu1 / 2.0, scale=nu1 / 2.0, size=n_samples, random_state=rng
            )
            Z = rng.standard_normal((n_samples, k))
            for j in range(k):
                P[:, j] = mu1[j] + gamma1[j] * V + np.sqrt(Sigma11[j] * V) * Z[:, j]

        for j in range(k, d):
            P[:, j] = rng.normal(0.0, np.sqrt(Lambda[j]), n_samples)

        Y = P @ W.T

        # U_i = F_{Y_i}(Y_i) via COS-CDF (marginal CF depends on dependence mode)
        U = np.empty_like(Y)
        if not dep:
            per_pc = self.params["per_pc"]
            cfs = self._build_generator_cfs_cross(Lambda, per_pc, k)
            for i in range(d):
                cf_Yi = self._make_marginal_cf(i, W, cfs)
                y_fine = np.linspace(-10.0, 10.0, 2000)
                cdf_fine = np.maximum.accumulate(self._cos_cdf(y_fine, cf_Yi))
                fwd = interp1d(
                    y_fine,
                    cdf_fine,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(0.0, 1.0),
                )
                U[:, i] = np.clip(fwd(Y[:, i]), 1e-10, 1 - 1e-10)
        else:
            tp = self.params["t_params_dep"]
            for i in range(d):
                cf_Yi = self._make_marginal_cf_cross_dep(i, W, tp, k, Lambda)
                y_fine = np.linspace(-10.0, 10.0, 2000)
                cdf_fine = np.maximum.accumulate(self._cos_cdf(y_fine, cf_Yi))
                fwd = interp1d(
                    y_fine,
                    cdf_fine,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(0.0, 1.0),
                )
                U[:, i] = np.clip(fwd(Y[:, i]), 1e-10, 1 - 1e-10)

        if return_y and return_p:
            return (U, Y, P)
        elif return_y:
            return (U, Y)
        elif return_p:
            return (U, P)
        else:
            return U

    def _simulate_t(
        self, n_samples: int, return_y: bool, return_p: bool, rng: np.random.Generator
    ):
        """
        Simulate from the fitted Skew-t_k / t_{d-k} PCC.

        First k PCs — stochastic representation (Eq. 9), shared mixing V:
            V    ~ IG(nu_1/2, nu_1/2)   via invgamma(a=nu_1/2, scale=nu_1/2)
            Z_j  ~ N(0, 1)  independently for j=1,...,k
            P_j  = mu_j + gamma_j V + sqrt(Sigma_{j,j} V) Z_j

        Higher PCs — independent shared mixing variable (Eq. 9):
            W_mix ~ IG(nu_rest/2, nu_rest/2)
            Z_j   ~ N(0, 1)
            P_j   = sqrt(Sigma_{j,j}) sqrt(W_mix) Z_j

        Rotation and copula observations via COS-CDF marginals.
        """
        k = self.k
        Lambda = self.Lambda
        W = self.W
        p = self.params
        d = self.dim

        nu1 = p["nu1"]
        gamma1 = p["gamma1"]
        Sigma11 = p["Sigma11"]
        mu1 = p["mu1"]
        nu_rest = p["nu_rest"]
        Sigma_diag = p["Sigma_diag"]

        # --- First k PCs (Eq. 9, j=1,...,k): share mixing variable V ---
        V = invgamma.rvs(a=nu1 / 2.0, scale=nu1 / 2.0, size=n_samples, random_state=rng)
        Z1 = rng.standard_normal((n_samples, k))
        # gamma1, mu1, Sigma11 are shape (k,); V is (n_samples,) — broadcast over k
        P1 = (
            mu1[None, :]
            + gamma1[None, :] * V[:, None]
            + np.sqrt(Sigma11[None, :] * V[:, None]) * Z1
        )  # shape (n_samples, k)

        # --- Higher PCs (Eq. 9, j>1): shared mixing variable ---
        W_mix = invgamma.rvs(
            a=nu_rest / 2.0, scale=nu_rest / 2.0, size=n_samples, random_state=rng
        )
        Z_rest = rng.standard_normal((n_samples, d - k))
        P_rest = np.sqrt(Sigma_diag) * np.sqrt(W_mix[:, None]) * Z_rest

        P = np.column_stack([P1, P_rest])
        Y = P @ W.T

        # --- U_i = F_{Y_i}(Y_i) via COS-CDF ---
        U = np.empty_like(Y)
        for i in range(d):
            cf_Yi = self._make_marginal_cf_t(i, W, p, k)
            y_fine = np.linspace(-10.0, 10.0, 2000)
            cdf_fine = self._cos_cdf(y_fine, cf_Yi)
            cdf_fine = np.maximum.accumulate(cdf_fine)
            fwd = interp1d(
                y_fine,
                cdf_fine,
                kind="linear",
                bounds_error=False,
                fill_value=(0.0, 1.0),
            )
            U[:, i] = np.clip(fwd(Y[:, i]), 1e-10, 1 - 1e-10)

        if return_y and return_p:
            return (U, Y, P)
        elif return_y:
            return (U, Y)
        elif return_p:
            return (U, P)
        else:
            return U
