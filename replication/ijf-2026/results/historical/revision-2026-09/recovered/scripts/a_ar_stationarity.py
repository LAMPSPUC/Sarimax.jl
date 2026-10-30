"""Part A - AR(2) and AR(3) stationarity regions from the reflection parameterization.

The region is generated ONLY through the package's recursion (`reflectionToAR` in
src/models/sarima.jl), never by clipping phi:

    phi_m^(m) = kappa_m,   phi_i^(m) = phi_i^(m-1) - kappa_m phi_(m-i)^(m-1),   |kappa_m| < 1.

It is then validated INDEPENDENTLY through the roots of the AR polynomial.

POLYNOMIAL CONVENTION. The model is y_t = phi_1 y_{t-1} + ... + phi_p y_{t-p} + eps_t, with
AR polynomial phi(z) = 1 - phi_1 z - ... - phi_p z^p. Stationarity means every root of phi(z)
lies OUTSIDE the unit circle, |z| > 1. Equivalently the INVERSE roots - the roots of
z^p - phi_1 z^(p-1) - ... - phi_p - lie INSIDE it. Both statements are reported, because texts
differ on which one they call "the roots".

WHICH FACES OF THE KAPPA CUBE BECOME THE BOUNDARY (AR(3)). Substituting into the recursion:
    kappa_1 = +1  ->  phi_1 + phi_2 + phi_3 = 1          (a root at z = +1)
    kappa_1 = -1  ->  -phi_1 + phi_2 - phi_3 = 1         (a root at z = -1)
    kappa_2 = -1  ->  phi_2 = phi_3^2 - phi_1 phi_3 - 1  (a complex pair ON the unit circle)
while kappa_3 = +-1 and kappa_2 = +1 collapse to line segments. The boundary is therefore two
planes and one ruled (saddle) surface, and the script TESTS whether that makes the region
non-convex rather than assuming it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C

RNG = np.random.default_rng(20260911)


# ---- the map ----------------------------------------------------------------------------
def reflection_to_ar(k: np.ndarray) -> np.ndarray:
    """Vectorized over the leading axis. k has shape (..., p)."""
    k = np.asarray(k, float)
    p = k.shape[-1]
    prev = k[..., :1].copy()
    for m in range(2, p + 1):
        km = k[..., m - 1:m]
        cur = np.empty(k.shape[:-1] + (m,))
        cur[..., m - 1] = km[..., 0]
        for i in range(1, m):
            cur[..., i - 1] = prev[..., i - 1] - km[..., 0] * prev[..., m - i - 1]
        prev = cur
    return prev


def ar_to_reflection(phi: np.ndarray) -> np.ndarray:
    """Inverse (step-down) recursion, used only to cross-check the classification."""
    phi = np.array(phi, float)
    p = phi.size
    k = np.empty(p)
    cur = phi.copy()
    for m in range(p, 0, -1):
        km = cur[m - 1]
        k[m - 1] = km
        if m == 1:
            break
        den = 1.0 - km * km
        if abs(den) < 1e-15:
            k[: m - 1] = np.nan
            break
        cur = np.array([(cur[i] + km * cur[m - 2 - i]) / den for i in range(m - 1)])
    return k


def min_root_modulus(phi: np.ndarray) -> float:
    """Smallest |z| among roots of 1 - phi_1 z - ... - phi_p z^p."""
    coeffs = np.concatenate(([1.0], -np.asarray(phi, float)))       # increasing powers
    roots = np.roots(coeffs[::-1])                                   # numpy wants decreasing
    return float(np.min(np.abs(roots)))


def max_inverse_root_modulus(phi: np.ndarray) -> float:
    return float(np.max(np.abs(np.roots(np.concatenate(([1.0], -np.asarray(phi, float)))))))


# ---- validation -------------------------------------------------------------------------
def validate(n_interior=200_000, n_boundary=20_000, n_box=200_000, n_pairs=200_000):
    out = {}
    for p in (2, 3):
        tol = 1e-3
        k = RNG.uniform(-(1 - tol), 1 - tol, size=(n_interior, p))
        phi = reflection_to_ar(k)
        mr = np.array([min_root_modulus(x) for x in phi])
        imr = np.array([max_inverse_root_modulus(x) for x in phi])
        # boundary: one coordinate pinned to +-1, the rest interior
        kb = RNG.uniform(-(1 - tol), 1 - tol, size=(n_boundary, p))
        face = RNG.integers(0, p, size=n_boundary)
        sign = RNG.choice([-1.0, 1.0], size=n_boundary)
        kb[np.arange(n_boundary), face] = sign
        mrb = np.array([min_root_modulus(x) for x in reflection_to_ar(kb)])
        # two-way classification on a box that contains the region: roots vs inverse map
        box = RNG.uniform(-3.2, 3.2, size=(n_box, p))
        by_roots = np.array([min_root_modulus(x) > 1.0 for x in box])
        kk = np.array([ar_to_reflection(x) for x in box])
        by_kappa = np.all(np.abs(kk) < 1.0, axis=1)
        by_kappa &= np.all(np.isfinite(kk), axis=1)
        disagree = np.flatnonzero(by_roots != by_kappa)
        # the only admissible disagreements are numerically ON the boundary
        near = np.array([abs(min_root_modulus(box[i]) - 1.0) < 1e-6 for i in disagree])
        out[f"AR({p})"] = {
            "interior_samples": n_interior,
            "interior_tolerance": tol,
            "interior_failing_root_condition": int(np.sum(mr <= 1.0)),
            "interior_min_root_modulus": float(mr.min()),
            "interior_max_inverse_root_modulus": float(imr.max()),
            "boundary_samples": n_boundary,
            "boundary_max_abs(min|root|-1)": float(np.max(np.abs(mrb - 1.0))),
            "box_samples": n_box,
            "box_stationary_fraction": float(by_roots.mean()),
            "box_disagreements_roots_vs_inverse_map": int(disagree.size),
            "box_disagreements_not_on_boundary": int(np.sum(~near)) if disagree.size else 0,
        }
        # convexity: is the midpoint of two stationary points always stationary?
        a = reflection_to_ar(RNG.uniform(-1, 1, size=(n_pairs, p)))
        b = reflection_to_ar(RNG.uniform(-1, 1, size=(n_pairs, p)))
        mid = 0.5 * (a + b)
        bad = np.array([min_root_modulus(x) <= 1.0 for x in mid])
        rec = out[f"AR({p})"]
        rec["convexity_pairs_tested"] = n_pairs
        rec["convexity_nonstationary_midpoints"] = int(bad.sum())
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            rec["convexity_counterexample"] = {
                "phi_a": a[i].round(6).tolist(), "phi_b": b[i].round(6).tolist(),
                "midpoint": mid[i].round(6).tolist(),
                "midpoint_min_root_modulus": min_root_modulus(mid[i]),
                "a_min_root_modulus": min_root_modulus(a[i]),
                "b_min_root_modulus": min_root_modulus(b[i]),
            }
    # the three AR(3) faces: check each mapped face satisfies its closed-form equation
    g = np.linspace(-1, 1, 41)
    A, B = np.meshgrid(g, g)
    faces = {}
    for name, fix in (("kappa1=+1", (0, 1.0)), ("kappa1=-1", (0, -1.0)), ("kappa2=-1", (1, -1.0))):
        free = [i for i in range(3) if i != fix[0]]
        k = np.zeros(A.shape + (3,))
        k[..., fix[0]] = fix[1]
        k[..., free[0]] = A
        k[..., free[1]] = B
        f = reflection_to_ar(k)
        if name == "kappa1=+1":
            res = f[..., 0] + f[..., 1] + f[..., 2] - 1.0
        elif name == "kappa1=-1":
            res = -f[..., 0] + f[..., 1] - f[..., 2] - 1.0
        else:
            res = f[..., 1] - (f[..., 2] ** 2 - f[..., 0] * f[..., 2] - 1.0)
        faces[name] = float(np.max(np.abs(res)))
    out["AR(3)"]["face_equation_max_residual"] = faces
    # deterministic non-convexity witness: the chord between two VERTICES of the closure
    a, d = np.array([3.0, -3.0, 1.0]), np.array([-3.0, -3.0, -1.0])
    out["AR(3)"]["vertex_chord_witness"] = {
        "A": a.tolist(), "D": d.tolist(), "midpoint": ((a + d) / 2).tolist(),
        "A_min_root_modulus": min_root_modulus(a),
        "D_min_root_modulus": min_root_modulus(d),
        "midpoint_min_root_modulus": min_root_modulus((a + d) / 2),
    }
    return out


# ---- figures ----------------------------------------------------------------------------
def square_boundary(n=400):
    t = np.linspace(-1, 1, n)
    return np.concatenate([
        np.column_stack([t, -np.ones(n)]), np.column_stack([np.ones(n), t]),
        np.column_stack([t[::-1], np.ones(n)]), np.column_stack([-np.ones(n), t[::-1]])])


def figure_main(val):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    C.setup_matplotlib()
    fig = plt.figure(figsize=(6.9, 3.1))

    # (a) AR(2): image of the kappa square
    ax = fig.add_subplot(1, 2, 1)
    edge = reflection_to_ar(square_boundary())
    ax.add_patch(Polygon(edge, closed=True, facecolor="#cde2fb", edgecolor=C.PALETTE[0], lw=1.2))
    kk = RNG.uniform(-1, 1, size=(4000, 2))
    pts = reflection_to_ar(kk)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.6, color=C.PALETTE[0], alpha=0.35, lw=0, rasterized=True)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-1.25, 1.25)
    ax.axhline(0, color=C.GRID, lw=0.5, zorder=0); ax.axvline(0, color=C.GRID, lw=0.5, zorder=0)
    ax.set_xlabel(r"$\phi_1$"); ax.set_ylabel(r"$\phi_2$")
    for (x, y), lab in (((-2, -1), r"$(-2,-1)$"), ((2, -1), r"$(2,-1)$"), ((0, 1), r"$(0,1)$")):
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(0, 5 if y > 0 else -10), ha="center", fontsize=7, color=C.INK2)
    ax.set_aspect("equal", adjustable="box")

    # (b) AR(3) as a stack of EXACT cross-sections. phi_3 = kappa_3 identically, so the slice
    # at phi_3 = c is the image of the (kappa_1, kappa_2) square with kappa_3 = c: a triangle.
    # Stacking them shows the solid without a surface that hides half of it, and makes the
    # twist that breaks convexity visible: the triangles rotate as phi_3 moves.
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    levels = np.linspace(-0.9, 0.9, 7)
    ramp = ["#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf"]
    for c, col in zip(levels, ramp):
        sq = square_boundary(200)
        tri = reflection_to_ar(np.column_stack([sq, np.full(len(sq), c)]))
        ax3.add_collection3d(Poly3DCollection([tri], facecolor=col, alpha=0.30,
                                              edgecolor=col, linewidths=0.9))
    # The solid has four vertices (images of the cube's corners) and FIVE edges. The faces
    # are the triangle ABC (kappa_1 = +1), the triangle BCD (kappa_1 = -1) and the skew
    # quadrilateral A-C-D-B (kappa_2 = -1, the saddle). There is NO edge A-D: the straight
    # chord between those two vertices leaves the region - its midpoint (0,-3,0) gives
    # 1 + 3z^2, roots of modulus 0.577 - which is the non-convexity, shown directly.
    V = {"A": (3, -3, 1), "B": (-1, 1, 1), "C": (1, 1, -1), "D": (-3, -3, -1)}
    for a, b in (("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D")):
        p, q = np.array(V[a]), np.array(V[b])
        ax3.plot(*zip(p, q), color=C.INK2, lw=0.55, ls=":")
    pa, pd_ = np.array(V["A"]), np.array(V["D"])
    ax3.plot(*zip(pa, pd_), color=C.PALETTE[7], lw=0.9, ls="--")
    ax3.scatter(0, -3, 0, color=C.PALETTE[7], s=10, depthshade=False)
    ax3.text(-0.2, -3, 0.18, "midpoint\nnot stationary", fontsize=5.8,
             color=C.PALETTE[7], ha="right", va="bottom")
    for lab, v in V.items():
        ax3.scatter(*v, color=C.INK, s=6, depthshade=False)
        ax3.text(v[0], v[1], v[2] + 0.12, f"({v[0]},{v[1]},{v[2]})", fontsize=6,
                 color=C.INK2, ha="center")
    ax3.set_xlim(-3, 3); ax3.set_ylim(-3, 1.2); ax3.set_zlim(-1, 1)
    ax3.set_xlabel(r"$\phi_1$", labelpad=-4); ax3.set_ylabel(r"$\phi_2$", labelpad=-4)
    ax3.tick_params(pad=-3)
    ax3.set_box_aspect((1.25, 1.0, 0.75), zoom=0.86)
    ax3.view_init(elev=18, azim=-62)
    # mplot3d places the z label outside the figure at this aspect; put it in figure coords
    fig.text(0.965, 0.52, r"$\phi_3$", fontsize=8.5, ha="center", va="center")
    # panel titles placed in FIGURE coordinates so (a) and (b) share one baseline; a 3D
    # axes positions its own title differently from a 2D one
    fig.text(0.07, 0.84, r"(a) AR(2): image of $(\kappa_1,\kappa_2)\in(-1,1)^2$", fontsize=8.5)
    fig.text(0.53, 0.84, r"(b) AR(3): exact sections at fixed $\phi_3$", fontsize=8.5)
    fig.subplots_adjust(wspace=0.02, left=0.07, right=0.95, top=0.88)
    C.save(fig, "ar_stationarity_ar2_ar3")
    plt.close(fig)


def figure_sections():
    """AR(3) cross-sections at fixed phi_3. Because phi_3 = kappa_3 exactly, each section is
    the image of the (kappa_1, kappa_2) square with kappa_3 held fixed - no slab, no
    approximation. The dashed outline is computed INDEPENDENTLY from the roots."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    C.setup_matplotlib()
    levels = [-0.75, -0.25, 0.25, 0.75]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.05), sharex=True, sharey=True)
    x = np.linspace(-3.0, 3.0, 301); y = np.linspace(-2.75, 1.35, 206)
    X, Y = np.meshgrid(x, y)
    for ax, c in zip(axes, levels):
        sq = square_boundary(300)
        k = np.column_stack([sq, np.full(len(sq), c)])
        edge = reflection_to_ar(k)[:, :2]
        ax.add_patch(Polygon(edge, closed=True, facecolor="#cde2fb",
                             edgecolor=C.PALETTE[0], lw=1.1))
        Z = np.array([[min_root_modulus([X[i, j], Y[i, j], c]) for j in range(X.shape[1])]
                      for i in range(X.shape[0])])
        ax.contour(X, Y, Z, levels=[1.0], colors=[C.INK], linewidths=0.7, linestyles="--")
        ax.set_title(rf"$\phi_3={c:+.2f}$", fontsize=8)
        ax.axhline(0, color=C.GRID, lw=0.4, zorder=0); ax.axvline(0, color=C.GRID, lw=0.4, zorder=0)
        ax.set_xlabel(r"$\phi_1$")
        # slices reach phi_2 = -1 - 2|phi_3| and phi_1 = +-(2 + |phi_3|)
        ax.set_xlim(-3.0, 3.0); ax.set_ylim(-2.75, 1.35)
    axes[0].set_ylabel(r"$\phi_2$")
    fig.subplots_adjust(wspace=0.08)
    C.save(fig, "ar3_stationarity_cross_sections")
    plt.close(fig)


CAPTIONS = r"""% ---- Figure: AR(2) and AR(3) stationarity regions ----------------------------------------
\caption{Stationarity regions generated through the reflection (partial-autocorrelation)
parameterization used by the package. (a) AR(2): the image of the open square
$(\kappa_1,\kappa_2)\in(-1,1)^2$ under $\phi_2=\kappa_2$, $\phi_1=\kappa_1(1-\kappa_2)$ is the
familiar triangle with vertices $(-2,-1)$, $(2,-1)$ and $(0,1)$. (b) AR(3): the image of the
open cube $(-1,1)^3$, drawn as exact sections at seven values of $\phi_3$. Since
$\phi_3=\kappa_3$ identically, each section is the image of the square $(\kappa_1,\kappa_2)$ with
$\kappa_3$ fixed, and is a triangle. The solid has four vertices and five edges (dotted); its
faces are the planes $\pm\phi_1+\phi_2\pm\phi_3=1$ (images of $\kappa_1=\pm1$, a root of the AR
polynomial at $z=\pm1$) and the ruled surface $\phi_2=\phi_3^2-\phi_1\phi_3-1$ (image of
$\kappa_2=-1$, a complex pair on the unit circle). The sections rotate with $\phi_3$, and the
region is not convex: the dashed chord joining the vertices $(3,-3,1)$ and $(-3,-3,-1)$ passes
through $(0,-3,0)$, whose AR polynomial $1+3z^2$ has roots of modulus $0.577$. Convention: the
AR polynomial is
$1-\phi_1z-\phi_2z^2-\phi_3z^3$, and stationarity means all of its roots lie outside the unit
circle (equivalently, all inverse roots lie inside it).}
\label{fig:ar-stationarity}

% ---- Supplementary figure: AR(3) cross-sections --------------------------------------------
\caption{Cross-sections of the AR(3) stationarity region at fixed $\phi_3$. Because
$\phi_3=\kappa_3$ identically, each section is exactly the image of the square
$(\kappa_1,\kappa_2)\in(-1,1)^2$ with $\kappa_3=\phi_3$ held fixed (shaded). The dashed line is
computed independently as the set where the smallest root modulus of
$1-\phi_1z-\phi_2z^2-\phi_3z^3$ equals one. Each section is a triangle, whose shape and position
change with $\phi_3$.}
\label{fig:ar3-sections}
"""


def main():
    import os
    cache = C.DATA / "ar_stationarity_validation.json"
    if os.environ.get("SKIP_VALIDATION") and cache.exists():
        val = json.loads(cache.read_text())
    else:
        val = validate()
        cache.write_text(json.dumps(val, indent=2))
    figure_main(val)
    figure_sections()
    (C.FIG / "ar_stationarity_captions.tex").write_text(CAPTIONS, encoding="utf-8")
    print(json.dumps(val, indent=2))


if __name__ == "__main__":
    main()
