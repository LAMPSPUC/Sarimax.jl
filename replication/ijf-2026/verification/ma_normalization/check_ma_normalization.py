"""Check the moving-average normalization of Supplementary Section S2 against the implementation.

    python check_ma_normalization.py

Supplementary Section S2 states the stationary-history determinant factor of an MA polynomial
in reflection coordinates, D_reg(theta) = prod_j [1 - eta_j(theta)^2]^(-j/T), its seasonal
counterpart D_seas(Theta) = prod_w [1 - eta_w(Theta)^2]^(-s w/T), and for the multiplicative
polynomial theta(B) Theta(B^s) the complete factor

    D_MA = D_reg(theta) D_seas(Theta) [R(theta, Theta; s)^2]^(-1/T),
    R = prod_i Theta(alpha_i^s),  theta(B) = prod_i (1 - alpha_i B)       (eq. app_d_ma_complete)

The package (src/models/sarima.jl, the MA block of the `mse` objective under `:penalized` and
`:innovations`) multiplies the sum of squares by D_reg(theta) D_seas(Theta) only, for both the
multiplicative and the additive seasonal specification.

This script works on the T-th power of each factor (the limit of the MA covariance
determinant, independent of T) and checks, for random invertible coefficients:

1. the definition: prod_j (1 - eta_j^2)^(-j) of a polynomial equals the limit of
   det(Omega_n) of its MA covariance (Toeplitz, n large), computed directly;
2. the supplement's identity: the complete multiplicative polynomial's factor equals
   D_reg D_seas R^-2 (so eq. app_d_ma_complete is exact);
3. the implementation: D_reg D_seas, the factor the package applies, differs from the complete
   factor by R^-2 for the multiplicative specification and by another factor for the additive
   one, which the package does not compute.

It prints the largest errors of 1 and 2 and the size of the omitted factor at T = 200. It
reads nothing and writes nothing; exit 0 when 1 and 2 hold to 1e-8, 1 otherwise.
"""

from __future__ import annotations

import sys

import numpy as np

RNG = np.random.default_rng(20261024)
T = 200


def reflection(coef: np.ndarray) -> np.ndarray:
    """Reflection coordinates of 1 + c_1 x + ... + c_m x^m (inverse of reflectionToMA)."""
    c = list(coef)
    m = len(c)
    eta = np.zeros(m)
    for k in range(m, 0, -1):
        e = c[k - 1]
        eta[k - 1] = e
        if k > 1:
            c = [(c[i] - e * c[k - 2 - i]) / (1 - e * e) for i in range(k - 1)]
    return eta


def log_det_limit(coef: np.ndarray, mult: int = 1) -> float:
    """log prod_j (1 - eta_j^2)^(-mult j): the stationary-history determinant in reflection form."""
    eta = reflection(coef)
    if np.any(np.abs(eta) >= 1):
        raise ValueError("not invertible")
    return float(-sum(mult * (j + 1) * np.log(1 - e * e) for j, e in enumerate(eta)))


def log_det_direct(coef: np.ndarray, n: int) -> float:
    """log det of the n x n covariance (unit innovation variance) of the MA process."""
    psi = np.concatenate(([1.0], coef))
    gamma = np.array([psi[: len(psi) - k] @ psi[k:] for k in range(len(psi))])
    row = np.zeros(n)
    row[: min(n, len(gamma))] = gamma[:n]
    omega = row[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
    sign, ld = np.linalg.slogdet(omega)
    assert sign > 0
    return float(ld)


def invertible(order: int, scale: float) -> np.ndarray:
    """Random MA coefficients through reflection coordinates in (-scale, scale)."""
    eta = RNG.uniform(-scale, scale, order)
    th = [eta[0]]
    for m in range(1, order):
        th = [th[i] + eta[m] * th[m - 1 - i] for i in range(m)] + [eta[m]]
    return np.array(th)


def seasonal(Theta: np.ndarray, s: int) -> np.ndarray:
    """Coefficients of Theta(B^s) as a polynomial in B."""
    out = np.zeros(s * len(Theta))
    out[s - 1 :: s] = Theta
    return out


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.convolve(np.concatenate(([1.0], a)), np.concatenate(([1.0], b)))[1:]


def resultant(theta: np.ndarray, Theta: np.ndarray, s: int) -> float:
    """R = prod_i Theta(alpha_i^s), with theta(B) = prod_i (1 - alpha_i B)."""
    alpha = np.roots(np.concatenate(([1.0], theta)))       # roots of x^q + theta_1 x^(q-1) + ...
    vals = [1 + sum(Theta[w] * a ** (s * (w + 1)) for w in range(len(Theta))) for a in alpha]
    r = np.prod(vals)
    assert abs(r.imag) < 1e-10
    return float(r.real)


def main() -> int:
    # 1. reflection form against the direct determinant
    err1 = 0.0
    for q in (1, 2, 3):
        c = invertible(q, 0.7)
        err1 = max(err1, abs(log_det_limit(c) - log_det_direct(c, 1500)))
    # 2 and 3
    err2, omitted_mult, omitted_add = 0.0, [], []
    for s in (4, 12):
        for q in (1, 2):
            for Q in (1, 2):
                for _ in range(25):
                    theta, Theta = invertible(q, 0.9), invertible(Q, 0.9)
                    reg = log_det_limit(theta)
                    seas = log_det_limit(Theta, mult=s)
                    full = log_det_limit(multiply(theta, seasonal(Theta, s)))
                    cross = -2 * np.log(abs(resultant(theta, Theta, s)))
                    err2 = max(err2, abs(full - (reg + seas + cross)))
                    omitted_mult.append(np.exp(cross / T))
                    add = np.concatenate((theta, np.zeros(s * Q - q))) + seasonal(Theta, s)
                    try:
                        omitted_add.append(np.exp((log_det_limit(add) - reg - seas) / T))
                    except ValueError:
                        pass                                   # additive polynomial not invertible
    print(f"1. reflection form vs direct determinant (n = 1500): largest |log difference| {err1:.2e}")
    print(f"2. complete multiplicative factor vs D_reg D_seas R^-2: largest |log difference| {err2:.2e}")
    print(f"3. factor omitted by the package at T = {T}: multiplicative R^(-2/T) in "
          f"[{min(omitted_mult):.6f}, {max(omitted_mult):.6f}] over {len(omitted_mult)} draws; "
          f"additive in [{min(omitted_add):.6f}, {max(omitted_add):.6f}] over {len(omitted_add)} invertible draws")
    theta, Theta = np.array([0.8]), np.array([-0.6])
    print(f"   example q = Q = 1, s = 12, theta_1 = 0.8, Theta_1 = -0.6: R = {resultant(theta, Theta, 12):.6f}, "
          f"R^(-2/T) = {resultant(theta, Theta, 12) ** (-2 / T):.6f}")
    ok = err1 < 1e-8 and err2 < 1e-8
    print("PASS: the supplement's identity holds; the package omits the cross factor" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
