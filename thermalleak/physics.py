"""
Heat-transfer physics for thermal leak detection.

Everything here is the *forward* physics: given a device with known material
properties and heat sources, what temperature field results? The detection
pipeline later inverts this reasoning.

Model
-----
A thin plate of thickness d, viewed face-on by a thermal camera. In-plane
conduction, convective loss from both faces to ambient:

    div( k*d * grad(T) )  +  P(x,y)  -  2h*(T - T_amb)  =  0

  k      thermal conductivity            [W/(m K)]   (may vary spatially)
  d      plate thickness                 [m]
  P      injected surface power density  [W/m^2]
  h      convective coefficient          [W/(m^2 K)]
  T_amb  ambient temperature             [K]

Two consequences matter enormously for detection, and they are the reason a
physics-informed detector beats plain image differencing:

1.  A localised source produces a *specific* radial profile. For a point source
    of power Q on an infinite plate the exact steady solution is

        dT(r) = Q / (2*pi*k*d) * K0(r / L)

    where K0 is the modified Bessel function of the second kind and

        L = sqrt(k*d / (2h))            [m]

    is the **thermal length** -- the distance over which heat spreads before
    convection carries it away. A real leak must look like this. An emissivity
    patch, a sticker, a moving reflection, or a registration artefact does not.

2.  L is set by the *material*, not by the leak. So the fitted length scale of a
    candidate anomaly is a hard physical check: an anomaly far sharper than L
    cannot be a conductive heat source, no matter how bright it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import kv


# --------------------------------------------------------------------------
# Material properties
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Material:
    """Bulk thermal properties of the plate being imaged."""

    name: str
    conductivity: float      # k   [W/(m K)]
    thickness: float         # d   [m]
    convection: float        # h   [W/(m^2 K)]
    emissivity: float        # eps [-]

    def thermal_length(self) -> float:
        """L = sqrt(k*d / 2h)  [m]. The natural spread of a point source."""
        return float(np.sqrt(self.conductivity * self.thickness / (2.0 * self.convection)))


# A few presets covering the scenarios in the project brief.
#
# Note on `conductivity`: these are *effective* in-plane values for the outer
# skin plus whatever backing structure conducts with it, not handbook values for
# the bare polymer. A laptop's bottom case is not a free-standing ABS sheet --
# it is bonded to standoffs, shields and a spreader, and the assembly conducts
# far better than the plastic alone. Using the handbook 0.2 W/(m K) produces
# case temperatures above 400 K, which is a good sanity check that the number is
# wrong rather than that the solver is.
MATERIALS = {
    # Laptop / consumer-electronics casing assembly.
    "abs_casing": Material("abs_casing", conductivity=1.2, thickness=0.003,
                           convection=14.0, emissivity=0.94),
    # Painted steel pipe wall under lagging -- the industrial-leak scenario.
    "steel_pipe": Material("steel_pipe", conductivity=16.0, thickness=0.004,
                           convection=12.0, emissivity=0.90),
    # Bare aluminium chassis: very high conductivity, leaks spread far and
    # low emissivity makes the radiometry genuinely difficult.
    "aluminium": Material("aluminium", conductivity=205.0, thickness=0.002,
                          convection=10.0, emissivity=0.20),

    # --- outdoor ground surfaces, for scene-scale imagery -----------------
    #
    # These exist so that footage of a street or a yard can be scored against
    # something other than a laptop casing. Two modelling notes, because the
    # numbers are not handbook values applied naively:
    #
    # `thickness` is the *thermally active layer* that conducts laterally, not
    # a slab dimension. A pavement is effectively semi-infinite downward; what
    # sets the lateral spreading length is how deep the near-surface layer that
    # participates in daily heat exchange goes, which is order 0.1-0.3 m.
    #
    # `convection` is halved from the physical value. `thermal_length` assumes
    # loss from *both* faces (the 2h in sqrt(kd/2h)) because it was written for
    # a thin plate in air. Ground loses heat from one face only, so an outdoor
    # h of ~15 W/(m^2 K) in light wind enters here as 7.5 to keep 2h correct.
    # Getting this wrong by the factor of two moves the thermal length by 1.4x.
    "concrete": Material("concrete", conductivity=1.4, thickness=0.15,
                         convection=7.5, emissivity=0.92),
    "asphalt": Material("asphalt", conductivity=0.75, thickness=0.15,
                        convection=7.5, emissivity=0.95),
}


# --------------------------------------------------------------------------
# Analytic point-source solution
# --------------------------------------------------------------------------

def point_source_profile(r: np.ndarray, power: float, material: Material,
                         r_floor: float | None = None) -> np.ndarray:
    """
    Steady temperature rise at radius `r` from a point source of `power` watts.

        dT(r) = Q / (2 pi k d) * K0(r / L)

    K0 diverges logarithmically at r = 0, which is a artefact of idealising a
    finite source as a point. We clamp r at `r_floor` (default: L/50).

    Parameters
    ----------
    r : array, radius in metres.
    power : Q, watts.
    material : plate properties.
    r_floor : minimum radius used in the evaluation, metres.

    Returns
    -------
    Temperature rise above ambient, kelvin.
    """
    L = material.thermal_length()
    if r_floor is None:
        r_floor = L / 50.0
    r_safe = np.maximum(np.asarray(r, dtype=float), r_floor)
    amplitude = power / (2.0 * np.pi * material.conductivity * material.thickness)
    return amplitude * kv(0, r_safe / L)


def bessel_model(r: np.ndarray, amplitude: float, length: float,
                 offset: float, r_floor_px: float = 0.75) -> np.ndarray:
    """
    Fitting form of the point-source profile, parameterised in *pixels*.

    Used by the detector, which does not know the true material properties and
    must estimate the thermal length from the image itself.

        dT(r) = A * K0(r / L) + c
    """
    length = max(float(length), 1e-3)
    r_safe = np.maximum(np.asarray(r, dtype=float), r_floor_px)
    return amplitude * kv(0, r_safe / length) + offset


# --------------------------------------------------------------------------
# Steady-state solver
# --------------------------------------------------------------------------

def _face_conductance(kd: np.ndarray, axis: int) -> np.ndarray:
    """
    Harmonic-mean conductance between adjacent cells along `axis`.

    The harmonic mean is the correct averaging for series thermal resistance --
    an arithmetic mean lets a highly conductive cell wrongly short-circuit an
    insulating neighbour.
    """
    a = np.take(kd, np.arange(0, kd.shape[axis] - 1), axis=axis)
    b = np.take(kd, np.arange(1, kd.shape[axis]), axis=axis)
    return 2.0 * a * b / np.maximum(a + b, 1e-12)


# LU factorisations are cached because the system matrix depends only on the
# geometry and material, never on the heat sources. Generating a dataset solves
# the same operator hundreds of times with different right-hand sides -- the
# baseline frame, the leaking frame, every pair -- so factorising once and
# back-substituting turns the dominant cost into a rounding error.
_FACTOR_CACHE: dict = {}
_FACTOR_CACHE_LIMIT = 8


def _get_factorisation(conductivity: np.ndarray, material: Material, dx: float):
    key = (conductivity.shape, float(dx), material.name,
           material.conductivity, material.thickness, material.convection,
           hash(conductivity.tobytes()))
    cached = _FACTOR_CACHE.get(key)
    if cached is not None:
        return cached

    if len(_FACTOR_CACHE) >= _FACTOR_CACHE_LIMIT:
        _FACTOR_CACHE.pop(next(iter(_FACTOR_CACHE)))

    A, sink = _assemble(conductivity, material, dx)
    solve = spla.factorized(A.tocsc())
    _FACTOR_CACHE[key] = (solve, sink)
    return solve, sink


def _assemble(conductivity: np.ndarray, material: Material, dx: float):
    """Build the sparse system matrix. Returns (matrix, per-cell sink conductance)."""
    ny, nx = conductivity.shape
    n = ny * nx
    cell_area = dx * dx

    kd = conductivity * material.thickness
    gx = _face_conductance(kd, axis=1)
    gy = _face_conductance(kd, axis=0)

    idx = np.arange(n).reshape(ny, nx)
    rows, cols, vals = [], [], []
    diag = np.zeros((ny, nx), dtype=float)

    left, right = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    g = gx.ravel()
    rows.extend([left, right]); cols.extend([right, left]); vals.extend([g, g])
    diag[:, :-1] -= gx
    diag[:, 1:] -= gx

    up, down = idx[:-1, :].ravel(), idx[1:, :].ravel()
    g = gy.ravel()
    rows.extend([up, down]); cols.extend([down, up]); vals.extend([g, g])
    diag[:-1, :] -= gy
    diag[1:, :] -= gy

    sink = 2.0 * material.convection * cell_area
    diag -= sink

    rows.append(idx.ravel()); cols.append(idx.ravel()); vals.append(diag.ravel())

    A = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    return A, sink


def solve_steady_state(conductivity: np.ndarray, power: np.ndarray,
                       material: Material, dx: float,
                       T_ambient: float = 293.15) -> np.ndarray:
    """
    Solve the steady heat equation on a rectangular grid with insulated edges.

    Insulated (zero-flux Neumann) outer boundaries are the right default: the
    device is bounded by air, and convective loss is already handled by the
    volumetric sink term rather than by an edge condition.

    Parameters
    ----------
    conductivity : (ny, nx) spatially varying k [W/(m K)].
    power        : (ny, nx) injected surface power density [W/m^2].
    material     : supplies thickness and convection coefficient.
    dx           : cell size [m] (square cells).
    T_ambient    : ambient temperature [K].

    Returns
    -------
    (ny, nx) temperature field in kelvin.
    """
    solve, sink = _get_factorisation(conductivity, material, dx)
    cell_area = dx * dx
    rhs = -(power * cell_area).ravel() - sink * T_ambient
    return solve(rhs).reshape(conductivity.shape)


# --------------------------------------------------------------------------
# Source construction
# --------------------------------------------------------------------------

def gaussian_source(shape: tuple[int, int], centre_px: tuple[float, float],
                    power_watts: float, sigma_px: float, dx: float) -> np.ndarray:
    """
    A localised heat source of total power `power_watts`, spread as a Gaussian.

    The Gaussian is normalised so the integral over the plate equals the stated
    power regardless of sigma -- so leak *power* and leak *size* stay
    independent knobs, which matters when sweeping detection limits.
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    cy, cx = centre_px
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    kernel = np.exp(-r2 / (2.0 * sigma_px ** 2))
    total = kernel.sum() * dx * dx
    if total <= 0:
        return np.zeros(shape)
    return power_watts * kernel / total


def crack_source(shape: tuple[int, int], start_px: tuple[float, float],
                 end_px: tuple[float, float], power_watts: float,
                 width_px: float, dx: float) -> np.ndarray:
    """
    A line-shaped leak: a crack or seam failure rather than a puncture.

    Implemented as distance-to-segment with a Gaussian cross-section, so it
    produces the elongated signature a real seam leak gives -- useful for
    testing whether a detector's isotropy assumption is too strict.
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    y0, x0 = start_px
    y1, x1 = end_px

    seg = np.array([y1 - y0, x1 - x0], dtype=float)
    seg_len2 = float(seg @ seg)
    if seg_len2 < 1e-9:
        return gaussian_source(shape, start_px, power_watts, width_px, dx)

    t = ((yy - y0) * seg[0] + (xx - x0) * seg[1]) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    dist2 = (yy - (y0 + t * seg[0])) ** 2 + (xx - (x0 + t * seg[1])) ** 2

    kernel = np.exp(-dist2 / (2.0 * width_px ** 2))
    total = kernel.sum() * dx * dx
    if total <= 0:
        return np.zeros(shape)
    return power_watts * kernel / total
