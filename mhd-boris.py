#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

"""
2.5D Constrained Transport Magnetohydrodynamics
Philip Mocz, @PMocz
Adam Reyes
(2024)

Simulate the Orszag-Tang vortex MHD problem (+more)
with a Boris-like Integrator to control timesteps!

See equations (1) -- (8) of Matsumoto, Miyoshi & Takasao (2019),
https://arxiv.org/abs/1902.02810

MAIN DIFFERENCE from vanilla MHD is that:

* Momentum is now (rho + rho_A) * v  instead of  rho * v,  with rho_A = |B|^2/c^2
  (equivalently (1 + vA^2/c^2) * rho * v).  The fluxes are unchanged.
* The Riemann solver uses Boris-corrected wave speeds, and its dissipation
  operates on the Boris-corrected conserved state.
* The timestep is therefore increased by the Boris factor.

I mark points of change from the vanilla-MHD model with "!!!" in comments.

The original problem has cf_max ~ 1.9, v_max ~ 1.6
"""

# directions for np.roll()
#  -1    right/up
#   1    left/down

# pressure/density floors applied to reconstructed face states only
# (the conservative update itself is untouched, so the scheme stays conservative)
SMALL_RHO = 1.0e-12
SMALL_P = 1.0e-12


def get_curl(Az, dx):
    """
    Calculate the discrete curl
    Az       is matrix of nodal z-component of magnetic potential
    dx       is the cell size
    bx       is matrix of cell face x-component magnetic-field
    by       is matrix of cell face y-component magnetic-field
    """

    bx = (Az - np.roll(Az, 1, axis=1)) / dx  # = d Az / d y
    by = -(Az - np.roll(Az, 1, axis=0)) / dx  # =-d Az / d x

    return bx, by


def get_div(bx, by, dx):
    """
    Calculate the discrete divergence of each cell
    dx       is the cell size
    bx       is matrix of cell face x-component magnetic-field
    by       is matrix of cell face y-component magnetic-field
    """

    divB = (bx - np.roll(bx, 1, axis=0) + by - np.roll(by, 1, axis=1)) / dx

    return divB


def get_avg(bx, by):
    """
    Calculate the volume-averaged magnetic field
    bx       is matrix of cell face x-component magnetic-field
    by       is matrix of cell face y-component magnetic-field
    Bx       is matrix of cell Bx
    By       is matrix of cell By
    """

    Bx = 0.5 * (bx + np.roll(bx, 1, axis=0))
    By = 0.5 * (by + np.roll(by, 1, axis=1))

    return Bx, By


def get_conserved(rho, vx, vy, vz, p, Bx, By, Bz, gamma, vol, c_limit):
    """
    Calculate the conserved variable from the primitive
    rho      is matrix of cell densities
    vx       is matrix of cell x-velocity
    vy       is matrix of cell y-velocity
    vz       is matrix of cell z-velocity
    p        is matrix of cell GAS pressures
    Bx       is matrix of cell Bx
    By       is matrix of cell By
    Bz       is matrix of cell Bz
    gamma    is ideal gas gamma
    vol      is cell volume
    c_limit  is the Boris speed limit
    Mass     is matrix of mass in cells
    Momx     is matrix of (Boris) x-momentum in cells
    Momy     is matrix of (Boris) y-momentum in cells
    Momz     is matrix of (Boris) z-momentum in cells
    Energy   is matrix of energy in cells
    """
    B2 = Bx**2 + By**2 + Bz**2
    rho_A = B2 / c_limit**2  # !!!
    Mass = rho * vol
    Momx = (rho + rho_A) * vx * vol  # !!!
    Momy = (rho + rho_A) * vy * vol  # !!!
    Momz = (rho + rho_A) * vz * vol  # !!!
    Energy = (p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2) + 0.5 * B2) * vol

    return Mass, Momx, Momy, Momz, Energy


def get_primitive(Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol, c_limit):
    """
    Calculate the primitive variable from the conservative
    Mass     is matrix of mass in cells
    Momx     is matrix of (Boris) x-momentum in cells
    Momy     is matrix of (Boris) y-momentum in cells
    Momz     is matrix of (Boris) z-momentum in cells
    Energy   is matrix of energy in cells
    Bx       is matrix of cell Bx
    By       is matrix of cell By
    Bz       is matrix of cell Bz
    gamma    is ideal gas gamma
    vol      is cell volume
    c_limit  is the Boris speed limit
    rho      is matrix of cell densities
    vx       is matrix of cell x-velocity
    vy       is matrix of cell y-velocity
    vz       is matrix of cell z-velocity
    p        is matrix of cell GAS pressures
    """
    B2 = Bx**2 + By**2 + Bz**2
    rho = Mass / vol
    rho_A = B2 / c_limit**2  # !!!
    vx = Momx / (rho + rho_A) / vol  # !!!
    vy = Momy / (rho + rho_A) / vol  # !!!
    vz = Momz / (rho + rho_A) / vol  # !!!
    p = (Energy / vol - 0.5 * rho * (vx**2 + vy**2 + vz**2) - 0.5 * B2) * (gamma - 1.0)

    return rho, vx, vy, vz, p


def get_gradient(f, dx):
    """
    Calculate the gradients of a field
    f        is a matrix of the field
    dx       is the cell size
    f_dx     is a matrix of derivative of f in the x-direction
    f_dy     is a matrix of derivative of f in the y-direction
    """

    f_dx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dx)
    f_dy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)

    return f_dx, f_dy


def slope_limit(f, dx, f_dx, f_dy):
    """
    Apply slope limiter to slopes
    f        is a matrix of the field
    dx       is the cell size
    f_dx     is a matrix of derivative of f in the x-direction
    f_dy     is a matrix of derivative of f in the y-direction
    """

    f_dx = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, ((f - np.roll(f, 1, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0))
            ),
        )
        * f_dx
    )
    f_dx = (
        np.maximum(
            0.0,
            np.minimum(
                1.0,
                (-(f - np.roll(f, -1, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0)),
            ),
        )
        * f_dx
    )
    f_dy = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, ((f - np.roll(f, 1, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0))
            ),
        )
        * f_dy
    )
    f_dy = (
        np.maximum(
            0.0,
            np.minimum(
                1.0,
                (-(f - np.roll(f, -1, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0)),
            ),
        )
        * f_dy
    )

    return f_dx, f_dy


def extrapolate_in_space_to_face(f, f_dx, f_dy, dx):
    """
    Extrapolate a cell-centered field to the two sides of each cell face.
    f        is a matrix of the field
    f_dx     is a matrix of the field x-derivatives
    f_dy     is a matrix of the field y-derivatives
    dx       is the cell size
    f_Lx     is the state on the left  side of the x-face at i+1/2 (from cell i)
    f_Rx     is the state on the right side of the x-face at i+1/2 (from cell i+1)
    f_Ly     is the state below        the y-face at j+1/2 (from cell j)
    f_Ry     is the state above        the y-face at j+1/2 (from cell j+1)
    """

    f_Lx = f + f_dx * dx / 2.0
    f_Rx = np.roll(f - f_dx * dx / 2.0, -1, axis=0)

    f_Ly = f + f_dy * dx / 2.0
    f_Ry = np.roll(f - f_dy * dx / 2.0, -1, axis=1)

    return f_Lx, f_Rx, f_Ly, f_Ry


def dudt_fluxes(flux_X, flux_Y, dx):
    F = -dx * flux_X
    F += dx * np.roll(flux_X, 1, axis=0)
    F += -dx * flux_Y
    F += dx * np.roll(flux_Y, 1, axis=1)
    return F


def dudt_stokes(flux_By_X, flux_Bx_Y, dx):
    # Ez at top right node of cell = avg of 4 fluxes
    Ez = 0.25 * (
        -flux_By_X
        - np.roll(flux_By_X, -1, axis=1)
        + flux_Bx_Y
        + np.roll(flux_Bx_Y, -1, axis=0)
    )
    dbx, dby = get_curl(-Ez, dx)

    return dbx, dby


def apply_dudt(F, dudt, dt):
    return F + dt * dudt


def get_boris_speeds(rho, p, Bn, Bt1, Bt2, gamma, c_limit):
    """
    Boris-corrected fast magnetosonic speed along the face normal n, and the
    Boris (Lorentz-like) factor.  Matsumoto et al. (2019):

        gamma_A  = 1 / sqrt(1 + vA^2/c^2)
        cf_boris = gamma_A * sqrt( (a^2 + vA^2
                                    + sqrt((a^2+vA^2)^2 - 4 a^2 vA_n^2)) / 2 )

    Note that the *directional* fast speed is used, not the upper bound
    sqrt(a^2 + vA^2): the two differ by a lot whenever B is nearly aligned
    with the face normal, and using the bound needlessly diffuses the solution
    and shrinks the timestep.
    """
    B2 = Bn**2 + Bt1**2 + Bt2**2
    a2 = gamma * np.maximum(p, SMALL_P) / rho
    vA2 = B2 / rho
    vAn2 = Bn**2 / rho

    gamma_A = 1.0 / np.sqrt(1.0 + vA2 / c_limit**2)  # !!!
    tmp = a2 + vA2
    disc = np.maximum(tmp**2 - 4.0 * a2 * vAn2, 0.0)
    cf = np.sqrt(0.5 * (tmp + np.sqrt(disc)))

    return gamma_A * cf, gamma_A  # !!!


def get_state_1d(rho, vn, vt1, vt2, p, Bn, Bt1, Bt2, gamma, c_limit, galilean=False):
    """
    Boris-corrected conserved state and its flux, in the face-aligned frame
    (n, t1, t2).  The fluxes are the ordinary MHD ones -- only the conserved
    momentum carries the Boris correction.  The normal field Bn is shared by
    the left and right states (constrained transport keeps it single-valued),
    so the normal-field flux vanishes identically and is not returned.

    galilean=True carries the Boris inertia through the momentum *flux* too
    (rho_B v v instead of rho v v).  Matsumoto et al. use rho v v, which is
    what the semi-relativistic expansion gives, but it leaves rho_A multiplying
    dv/dt and not (v.grad)v -- so the momentum equation expands to

        M dv/dt + rho (v.grad)v + v drho_A/dt = -grad(p_T) + div(BB),

    which is not Galilean invariant: boost the frame by U and the Alfven speed
    measured relative to the fluid picks up a dependence on U (the two roots of
    M d^2 + U rho_A d - Bn^2 = 0 stop being symmetric).  Using rho_B in the
    flux makes the same expansion read

        M Dv/Dt + v [Drho_A/Dt + rho_A div(v)] = -grad(p_T) + div(BB),

    so the frame dependence vanishes wherever rho_A is advected like a mass
    density -- exactly, for instance, for a circularly polarized Alfven wave.
    A residual remains where the field is compressed or amplified; removing
    that entirely means evolving rho_A as its own advected variable.
    """
    B2 = Bn**2 + Bt1**2 + Bt2**2
    v2 = vn**2 + vt1**2 + vt2**2
    vB = vn * Bn + vt1 * Bt1 + vt2 * Bt2
    pT = p + 0.5 * B2
    en = p / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * B2
    rho_B = rho + B2 / c_limit**2  # !!! Boris-corrected inertia

    U = (
        rho,
        rho_B * vn,  # !!!
        rho_B * vt1,  # !!!
        rho_B * vt2,  # !!!
        en,
        Bt1,
        Bt2,
    )
    rho_F = rho_B if galilean else rho  # !!!
    F = (
        rho * vn,
        rho_F * vn * vn + pT - Bn * Bn,
        rho_F * vn * vt1 - Bn * Bt1,
        rho_F * vn * vt2 - Bn * Bt2,
        (en + pT) * vn - Bn * vB,
        Bt1 * vn - Bn * vt1,
        Bt2 * vn - Bn * vt2,
    )

    return U, F


def get_flux_llf(Bn, W_L, W_R, gamma, c_limit, galilean=False):
    """
    Calculate fluxes between 2 states with the local Lax-Friedrichs/Rusanov rule
    Bn       is the (single-valued) face-normal magnetic field
    W_L      is the left  state (rho, vn, vt1, vt2, p, Bt1, Bt2)
    W_R      is the right state (rho, vn, vt1, vt2, p, Bt1, Bt2)
    gamma    is the ideal gas gamma
    c_limit  is the Boris speed limit
    """
    rho_L, vn_L, vt1_L, vt2_L, p_L, Bt1_L, Bt2_L = W_L
    rho_R, vn_R, vt1_R, vt2_R, p_R, Bt1_R, Bt2_R = W_R

    U_L, F_L = get_state_1d(
        rho_L, vn_L, vt1_L, vt2_L, p_L, Bn, Bt1_L, Bt2_L, gamma, c_limit, galilean
    )
    U_R, F_R = get_state_1d(
        rho_R, vn_R, vt1_R, vt2_R, p_R, Bn, Bt1_R, Bt2_R, gamma, c_limit, galilean
    )

    # !!! Boris-corrected signal speed
    cf_L, _ = get_boris_speeds(rho_L, p_L, Bn, Bt1_L, Bt2_L, gamma, c_limit)
    cf_R, _ = get_boris_speeds(rho_R, p_R, Bn, Bt1_R, Bt2_R, gamma, c_limit)
    C = np.maximum(np.abs(vn_L) + cf_L, np.abs(vn_R) + cf_R)

    # !!! the dissipation acts on the Boris-corrected conserved state U
    return tuple(
        0.5 * (fl + fr) - 0.5 * C * (ur - ul)
        for fl, fr, ul, ur in zip(F_L, F_R, U_L, U_R)
    )


def get_flux_hll(Bn, W_L, W_R, gamma, c_limit, galilean=False):
    """
    Calculate fluxes between 2 states with the HLL rule.  Same arguments as
    get_flux_llf.  Cheaper on dissipation than LLF because the two signal
    speeds are one-sided.
    """
    rho_L, vn_L, vt1_L, vt2_L, p_L, Bt1_L, Bt2_L = W_L
    rho_R, vn_R, vt1_R, vt2_R, p_R, Bt1_R, Bt2_R = W_R

    U_L, F_L = get_state_1d(
        rho_L, vn_L, vt1_L, vt2_L, p_L, Bn, Bt1_L, Bt2_L, gamma, c_limit, galilean
    )
    U_R, F_R = get_state_1d(
        rho_R, vn_R, vt1_R, vt2_R, p_R, Bn, Bt1_R, Bt2_R, gamma, c_limit, galilean
    )

    cf_L, gamma_A_L = get_boris_speeds(rho_L, p_L, Bn, Bt1_L, Bt2_L, gamma, c_limit)
    cf_R, gamma_A_R = get_boris_speeds(rho_R, p_R, Bn, Bt1_R, Bt2_R, gamma, c_limit)
    cf = np.maximum(cf_L, cf_R)

    # !!! The Boris system is not Galilean invariant: the advective eigenvalues
    # are shifted towards gamma_A^2 * vn (see eqn. 14 of Matsumoto et al. 2019,
    # whose Alfven speeds interpolate between gamma_A^2 S_M and S_M).  Bracket
    # both so that the HLL fan always contains the true wave fan.
    u_L = gamma_A_L**2 * vn_L
    u_R = gamma_A_R**2 * vn_R
    S_L = np.minimum(np.minimum(vn_L, u_L), np.minimum(vn_R, u_R)) - cf
    S_R = np.maximum(np.maximum(vn_L, u_L), np.maximum(vn_R, u_R)) + cf

    S_L = np.minimum(S_L, 0.0)
    S_R = np.maximum(S_R, 0.0)
    inv = 1.0 / (S_R - S_L)

    return tuple(
        (S_R * fl - S_L * fr + S_L * S_R * (ur - ul)) * inv
        for fl, fr, ul, ur in zip(F_L, F_R, U_L, U_R)
    )


# HLLD (adapted from Athena).  Vanilla MHD only -- no Boris correction.
def get_flux_hlld(Bn, W_L, W_R, gamma, c_limit, galilean=False):
    """
    Calculate fluxes between 2 states with the HLLD solver of
    Miyoshi & Kusano (2005).  Same arguments as get_flux_llf.

    NOTE: this solver assumes the vanilla MHD eigenstructure and is only used
    when the Boris correction is switched off (c_limit = inf).  Its Boris
    generalisation is the subject of Matsumoto et al. (2019) and is not
    implemented here.  `galilean` is accepted for a uniform solver signature
    and ignored: at c_limit = inf the Boris inertia is just rho.
    """
    del galilean

    SMALL_NUMBER = 1.0e-8

    rho_L, vx_L, vy_L, vz_L, P_L, By_L, Bz_L = W_L
    rho_R, vx_R, vy_R, vz_R, P_R, By_R, Bz_R = W_R
    Bxi = Bn

    Mx_L = rho_L * vx_L
    My_L = rho_L * vy_L
    Mz_L = rho_L * vz_L
    E_L = (
        P_L / (gamma - 1.0)
        + 0.5 * rho_L * (vx_L**2 + vy_L**2 + vz_L**2)
        + 0.5 * (Bxi**2 + By_L**2 + Bz_L**2)
    )

    Mx_R = rho_R * vx_R
    My_R = rho_R * vy_R
    Mz_R = rho_R * vz_R
    E_R = (
        P_R / (gamma - 1.0)
        + 0.5 * rho_R * (vx_R**2 + vy_R**2 + vz_R**2)
        + 0.5 * (Bxi**2 + By_R**2 + Bz_R**2)
    )

    # --- Step 2. ------------------------------------------------------------------
    #  Compute left & right wave speeds according to Miyoshi & Kusano, eqn. (67)
    #

    pbl = 0.5 * (Bxi**2 + By_L**2 + Bz_L**2)
    pbr = 0.5 * (Bxi**2 + By_R**2 + Bz_R**2)
    gpl = gamma * P_L
    gpr = gamma * P_R
    gpbl = gpl + 2.0 * pbl
    gpbr = gpr + 2.0 * pbr

    Bxsq = Bxi**2
    cfl = np.sqrt((gpbl + np.sqrt(gpbl**2 - 4.0 * gpl * Bxsq)) / (2.0 * rho_L))
    cfr = np.sqrt((gpbr + np.sqrt(gpbr**2 - 4.0 * gpr * Bxsq)) / (2.0 * rho_R))
    cfmax = np.maximum(cfl, cfr)

    spd1 = (vx_L - cfmax) * (vx_L <= vx_R) + (vx_R - cfmax) * (vx_L > vx_R)
    spd5 = (vx_R + cfmax) * (vx_L <= vx_R) + (vx_L + cfmax) * (vx_L > vx_R)

    # --- Step 3. ------------------------------------------------------------------
    # Compute L/R fluxes
    #

    # total pressure
    ptl = P_L + pbl
    ptr = P_R + pbr

    FL_d = Mx_L
    FL_Mx = Mx_L * vx_L + ptl - Bxsq
    FL_My = rho_L * vx_L * vy_L - Bxi * By_L
    FL_Mz = rho_L * vx_L * vz_L - Bxi * Bz_L
    FL_E = vx_L * (E_L + ptl - Bxsq) - Bxi * (vy_L * By_L + vz_L * Bz_L)
    FL_By = By_L * vx_L - Bxi * vy_L
    FL_Bz = Bz_L * vx_L - Bxi * vz_L
    FR_d = Mx_R
    FR_Mx = Mx_R * vx_R + ptr - Bxsq
    FR_My = rho_R * vx_R * vy_R - Bxi * By_R
    FR_Mz = rho_R * vx_R * vz_R - Bxi * Bz_R
    FR_E = vx_R * (E_R + ptr - Bxsq) - Bxi * (vy_R * By_R + vz_R * Bz_R)
    FR_By = By_R * vx_R - Bxi * vy_R
    FR_Bz = Bz_R * vx_R - Bxi * vz_R

    # --- Step 4. ------------------------------------------------------------------
    #  Return upwind flux if flow is supersonic
    #

    # deferred to the end

    # --- Step 5. ------------------------------------------------------------------
    #  Compute middle and Alfven wave speeds
    #

    sdl = spd1 - vx_L
    sdr = spd5 - vx_R

    # S_M: eqn (38) of Miyoshi & Kusano
    spd3 = (sdr * rho_R * vx_R - sdl * rho_L * vx_L - ptr + ptl) / (
        sdr * rho_R - sdl * rho_L
    )

    sdml = spd1 - spd3
    sdmr = spd5 - spd3
    # eqn (43) of Miyoshi & Kusano
    ULst_d = rho_L * sdl / sdml
    URst_d = rho_R * sdr / sdmr
    sqrtdl = np.sqrt(ULst_d)
    sqrtdr = np.sqrt(URst_d)

    # eqn (51) of Miyoshi & Kusano
    spd2 = spd3 - np.abs(Bxi) / sqrtdl
    spd4 = spd3 + np.abs(Bxi) / sqrtdr

    # --- Step 6. ------------------------------------------------------------------
    # Compute intermediate states
    #

    ptst = ptl + rho_L * sdl * (sdl - sdml)

    # Ul*
    # eqn (39) of M&K
    ULst_Mx = ULst_d * spd3
    # ULst_Bx = Bxi
    isDegen = np.abs(rho_L * sdl * sdml / Bxsq - 1.0) < SMALL_NUMBER

    # eqns (44) and (46) of M&K
    tmp = Bxi * (sdl - sdml) / (rho_L * sdl * sdml - Bxsq)
    ULst_My = (ULst_d * vy_L) * isDegen + (ULst_d * (vy_L - By_L * tmp)) * (~isDegen)
    ULst_Mz = (ULst_d * vz_L) * isDegen + (ULst_d * (vz_L - Bz_L * tmp)) * (~isDegen)

    # eqns (45) and (47) of M&K
    tmp = (rho_L * (sdl) ** 2 - Bxsq) / (rho_L * sdl * sdml - Bxsq)
    ULst_By = (By_L) * isDegen + (By_L * tmp) * (~isDegen)
    ULst_Bz = (Bz_L) * isDegen + (Bz_L * tmp) * (~isDegen)

    vbstl = (ULst_Mx * Bxi + ULst_My * ULst_By + ULst_Mz * ULst_Bz) / ULst_d
    # eqn (48) of M&K#
    ULst_E = (
        sdl * E_L
        - ptl * vx_L
        + ptst * spd3
        + Bxi * (vx_L * Bxi + vy_L * By_L + vz_L * Bz_L - vbstl)
    ) / sdml
    WLst_vy = ULst_My / ULst_d
    WLst_vz = ULst_Mz / ULst_d

    # Ur*
    # eqn (39) of M&K
    URst_Mx = URst_d * spd3
    # URst_Bx = Bxi
    isDegen = np.abs(rho_R * sdr * sdmr / Bxsq - 1.0) < SMALL_NUMBER

    # eqns (44) and (46) of M&K
    tmp = Bxi * (sdr - sdmr) / (rho_R * sdr * sdmr - Bxsq)
    URst_My = (URst_d * vy_R) * isDegen + (URst_d * (vy_R - By_R * tmp)) * (~isDegen)
    URst_Mz = (URst_d * vz_R) * isDegen + (URst_d * (vz_R - Bz_R * tmp)) * (~isDegen)

    # eqns (45) and (47) of M&K
    tmp = (rho_R * (sdr) ** 2 - Bxsq) / (rho_R * sdr * sdmr - Bxsq)
    URst_By = (By_R) * isDegen + (By_R * tmp) * (~isDegen)
    URst_Bz = (Bz_R) * isDegen + (Bz_R * tmp) * (~isDegen)

    vbstr = (URst_Mx * Bxi + URst_My * URst_By + URst_Mz * URst_Bz) / URst_d
    # eqn (48) of M&K
    URst_E = (
        sdr * E_R
        - ptr * vx_R
        + ptst * spd3
        + Bxi * (vx_R * Bxi + vy_R * By_R + vz_R * Bz_R - vbstr)
    ) / sdmr
    WRst_vy = URst_My / URst_d
    WRst_vz = URst_Mz / URst_d

    # Ul**  and Ur**  - if Bx is zero, same as  * -states
    #   if(Bxi == 0.0)
    isDegen = 0.5 * Bxsq / np.minimum(pbl, pbr) < (SMALL_NUMBER) ** 2
    ULdst_d = ULst_d * isDegen
    ULdst_Mx = ULst_Mx * isDegen
    ULdst_My = ULst_My * isDegen
    ULdst_Mz = ULst_Mz * isDegen
    ULdst_By = ULst_By * isDegen
    ULdst_Bz = ULst_Bz * isDegen
    ULdst_E = ULst_E * isDegen

    URdst_d = URst_d * isDegen
    URdst_Mx = URst_Mx * isDegen
    URdst_My = URst_My * isDegen
    URdst_Mz = URst_Mz * isDegen
    URdst_By = URst_By * isDegen
    URdst_Bz = URst_Bz * isDegen
    URdst_E = URst_E * isDegen

    # else
    invsumd = 1.0 / (sqrtdl + sqrtdr)
    Bxsig = np.where(Bxi > 0, 1.0, -1.0)

    ULdst_d = ULdst_d + ULst_d * (~isDegen)
    URdst_d = URdst_d + URst_d * (~isDegen)

    ULdst_Mx = ULdst_Mx + ULst_Mx * (~isDegen)
    URdst_Mx = URdst_Mx + URst_Mx * (~isDegen)

    # eqn (59) of M&K
    tmp = invsumd * (sqrtdl * WLst_vy + sqrtdr * WRst_vy + Bxsig * (URst_By - ULst_By))
    ULdst_My = ULdst_My + ULdst_d * tmp * (~isDegen)
    URdst_My = URdst_My + URdst_d * tmp * (~isDegen)

    # eqn (60) of M&K
    tmp = invsumd * (sqrtdl * WLst_vz + sqrtdr * WRst_vz + Bxsig * (URst_Bz - ULst_Bz))
    ULdst_Mz = ULdst_Mz + ULdst_d * tmp * (~isDegen)
    URdst_Mz = URdst_Mz + URdst_d * tmp * (~isDegen)

    # eqn (61) of M&K
    tmp = invsumd * (
        sqrtdl * URst_By
        + sqrtdr * ULst_By
        + Bxsig * sqrtdl * sqrtdr * (WRst_vy - WLst_vy)
    )
    ULdst_By = ULdst_By + tmp * (~isDegen)
    URdst_By = URdst_By + tmp * (~isDegen)

    # eqn (62) of M&K
    tmp = invsumd * (
        sqrtdl * URst_Bz
        + sqrtdr * ULst_Bz
        + Bxsig * sqrtdl * sqrtdr * (WRst_vz - WLst_vz)
    )
    ULdst_Bz = ULdst_Bz + tmp * (~isDegen)
    URdst_Bz = URdst_Bz + tmp * (~isDegen)

    # eqn (63) of M&K
    tmp = spd3 * Bxi + (ULdst_My * ULdst_By + ULdst_Mz * ULdst_Bz) / ULdst_d
    ULdst_E = ULdst_E + (ULst_E - sqrtdl * Bxsig * (vbstl - tmp)) * (~isDegen)
    URdst_E = URdst_E + (URst_E + sqrtdr * Bxsig * (vbstr - tmp)) * (~isDegen)

    # --- Step 7. ------------------------------------------------------------------
    #  Compute flux
    #

    flux_Mass = FL_d * (spd1 >= 0)
    flux_Momx = FL_Mx * (spd1 >= 0)
    flux_Momy = FL_My * (spd1 >= 0)
    flux_Momz = FL_Mz * (spd1 >= 0)
    flux_Energy = FL_E * (spd1 >= 0)
    flux_By = FL_By * (spd1 >= 0)
    flux_Bz = FL_Bz * (spd1 >= 0)

    flux_Mass += FR_d * (spd5 <= 0)
    flux_Momx += FR_Mx * (spd5 <= 0)
    flux_Momy += FR_My * (spd5 <= 0)
    flux_Momz += FR_Mz * (spd5 <= 0)
    flux_Energy += FR_E * (spd5 <= 0)
    flux_By += FR_By * (spd5 <= 0)
    flux_Bz += FR_Bz * (spd5 <= 0)

    # if(spd2 >= 0)
    # return Fl * #
    flux_Mass += (FL_d + spd1 * (ULst_d - rho_L)) * ((spd1 < 0) & (spd2 >= 0))
    flux_Momx += (FL_Mx + spd1 * (ULst_Mx - Mx_L)) * ((spd1 < 0) & (spd2 >= 0))
    flux_Momy += (FL_My + spd1 * (ULst_My - My_L)) * ((spd1 < 0) & (spd2 >= 0))
    flux_Momz += (FL_Mz + spd1 * (ULst_Mz - Mz_L)) * ((spd1 < 0) & (spd2 >= 0))
    flux_Energy += (FL_E + spd1 * (ULst_E - E_L)) * ((spd1 < 0) & (spd2 >= 0))
    flux_By += (FL_By + spd1 * (ULst_By - By_L)) * ((spd1 < 0) & (spd2 >= 0))
    flux_Bz += (FL_Bz + spd1 * (ULst_Bz - Bz_L)) * ((spd1 < 0) & (spd2 >= 0))

    # elseif(spd3 >= 0)
    # return Fl *  *
    tmp = spd2 - spd1
    flux_Mass += (FL_d - spd1 * rho_L - tmp * ULst_d + spd2 * ULdst_d) * (
        (spd2 < 0) & (spd3 >= 0)
    )
    flux_Momx += (FL_Mx - spd1 * Mx_L - tmp * ULst_Mx + spd2 * ULdst_Mx) * (
        (spd2 < 0) & (spd3 >= 0)
    )
    flux_Momy += (FL_My - spd1 * My_L - tmp * ULst_My + spd2 * ULdst_My) * (
        (spd2 < 0) & (spd3 >= 0)
    )
    flux_Momz += (FL_Mz - spd1 * Mz_L - tmp * ULst_Mz + spd2 * ULdst_Mz) * (
        (spd2 < 0) & (spd3 >= 0)
    )
    flux_Energy += (FL_E - spd1 * E_L - tmp * ULst_E + spd2 * ULdst_E) * (
        (spd2 < 0) & (spd3 >= 0)
    )
    flux_By += (FL_By - spd1 * By_L - tmp * ULst_By + spd2 * ULdst_By) * (
        (spd2 < 0) & (spd3 >= 0)
    )
    flux_Bz += (FL_Bz - spd1 * Bz_L - tmp * ULst_Bz + spd2 * ULdst_Bz) * (
        (spd2 < 0) & (spd3 >= 0)
    )

    # elseif(spd4 > 0)
    # return Fr *  *
    tmp = spd4 - spd5
    flux_Mass += (FR_d - spd5 * rho_R - tmp * URst_d + spd4 * URdst_d) * (
        (spd3 < 0) & (spd4 > 0)
    )
    flux_Momx += (FR_Mx - spd5 * Mx_R - tmp * URst_Mx + spd4 * URdst_Mx) * (
        (spd3 < 0) & (spd4 > 0)
    )
    flux_Momy += (FR_My - spd5 * My_R - tmp * URst_My + spd4 * URdst_My) * (
        (spd3 < 0) & (spd4 > 0)
    )
    flux_Momz += (FR_Mz - spd5 * Mz_R - tmp * URst_Mz + spd4 * URdst_Mz) * (
        (spd3 < 0) & (spd4 > 0)
    )
    flux_Energy += (FR_E - spd5 * E_R - tmp * URst_E + spd4 * URdst_E) * (
        (spd3 < 0) & (spd4 > 0)
    )
    flux_By += (FR_By - spd5 * By_R - tmp * URst_By + spd4 * URdst_By) * (
        (spd3 < 0) & (spd4 > 0)
    )
    flux_Bz += (FR_Bz - spd5 * Bz_R - tmp * URst_Bz + spd4 * URdst_Bz) * (
        (spd3 < 0) & (spd4 > 0)
    )

    # else
    # return Fr *
    flux_Mass += (FR_d + spd5 * (URst_d - rho_R)) * ((spd4 <= 0) & (spd5 > 0))
    flux_Momx += (FR_Mx + spd5 * (URst_Mx - Mx_R)) * ((spd4 <= 0) & (spd5 > 0))
    flux_Momy += (FR_My + spd5 * (URst_My - My_R)) * ((spd4 <= 0) & (spd5 > 0))
    flux_Momz += (FR_Mz + spd5 * (URst_Mz - Mz_R)) * ((spd4 <= 0) & (spd5 > 0))
    flux_Energy += (FR_E + spd5 * (URst_E - E_R)) * ((spd4 <= 0) & (spd5 > 0))
    flux_By += (FR_By + spd5 * (URst_By - By_R)) * ((spd4 <= 0) & (spd5 > 0))
    flux_Bz += (FR_Bz + spd5 * (URst_Bz - Bz_R)) * ((spd4 <= 0) & (spd5 > 0))

    return flux_Mass, flux_Momx, flux_Momy, flux_Momz, flux_Energy, flux_By, flux_Bz


RIEMANN_SOLVERS = {
    "llf": get_flux_llf,
    "hll": get_flux_hll,
    "hlld": get_flux_hlld,
}


def get_flux(method, Bn, W_L, W_R, gamma, c_limit, galilean=False):
    try:
        solver = RIEMANN_SOLVERS[method]
    except KeyError:
        raise ValueError(
            f"Unknown Riemann solver {method!r}; choose from {sorted(RIEMANN_SOLVERS)}"
        ) from None
    return solver(Bn, W_L, W_R, gamma, c_limit, galilean)


def get_dudt(rho, vx, vy, vz, p, bx, by, Bz, dx, gamma, c_limit, opts):
    """single stage of a runge-kutta method to return dudt of all these variables"""
    Bx, By = get_avg(bx, by)

    fields = {
        "rho": rho,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "p": p,
        "Bx": Bx,
        "By": By,
        "Bz": Bz,
    }

    faces = {}
    for name, f in fields.items():
        f_dx, f_dy = get_gradient(f, dx)
        if opts.slope_limiting:
            f_dx, f_dy = slope_limit(f, dx, f_dx, f_dy)
        faces[name] = extrapolate_in_space_to_face(f, f_dx, f_dy, dx)

    # keep the reconstructed states physical
    for name, floor in (("rho", SMALL_RHO), ("p", SMALL_P)):
        faces[name] = tuple(np.maximum(f, floor) for f in faces[name])

    def face(name, i):
        return faces[name][i]

    # x-direction: the normal field is the face-centered bx, which constrained
    # transport keeps single-valued -- no reconstruction, and hence no spurious
    # div(B) monopole force in the Riemann problem.
    (
        flux_Mass_X,
        flux_Momx_X,
        flux_Momy_X,
        flux_Momz_X,
        flux_Energy_X,
        flux_By_X,
        flux_Bz_X,
    ) = get_flux(
        opts.riemann_solver,
        bx,
        tuple(face(n, 0) for n in ("rho", "vx", "vy", "vz", "p", "By", "Bz")),
        tuple(face(n, 1) for n in ("rho", "vx", "vy", "vz", "p", "By", "Bz")),
        gamma,
        c_limit,
        opts.galilean,
    )
    # y-direction: cyclic permutation (x, y, z) -> (y, z, x)
    (
        flux_Mass_Y,
        flux_Momy_Y,
        flux_Momz_Y,
        flux_Momx_Y,
        flux_Energy_Y,
        flux_Bz_Y,
        flux_Bx_Y,
    ) = get_flux(
        opts.riemann_solver,
        by,
        tuple(face(n, 2) for n in ("rho", "vy", "vz", "vx", "p", "Bz", "Bx")),
        tuple(face(n, 3) for n in ("rho", "vy", "vz", "vx", "p", "Bz", "Bx")),
        gamma,
        c_limit,
        opts.galilean,
    )

    dudt_Mass = dudt_fluxes(flux_Mass_X, flux_Mass_Y, dx)
    dudt_Momx = dudt_fluxes(flux_Momx_X, flux_Momx_Y, dx)
    dudt_Momy = dudt_fluxes(flux_Momy_X, flux_Momy_Y, dx)
    dudt_Momz = dudt_fluxes(flux_Momz_X, flux_Momz_Y, dx)
    dudt_Energy = dudt_fluxes(flux_Energy_X, flux_Energy_Y, dx)
    dudt_Bz = dudt_fluxes(flux_Bz_X, flux_Bz_Y, dx) / dx**2
    dudt_bx, dudt_by = dudt_stokes(flux_By_X, flux_Bx_Y, dx)

    return (
        dudt_Mass,
        dudt_Momx,
        dudt_Momy,
        dudt_Momz,
        dudt_Energy,
        dudt_Bz,
        dudt_bx,
        dudt_by,
    )


def get_initial_conditions(prob_id, gamma, X, Y, c_limit, boris_eigenmode=True):
    """Primitive initial conditions (with GAS pressure p), plus Az and t_end."""
    if prob_id == 1:
        # Orszag-Tang vortex problem
        t_end = 0.5
        rho = (gamma**2 / (4.0 * np.pi)) * np.ones(X.shape)
        vx = -np.sin(2.0 * np.pi * Y)
        vy = np.sin(2.0 * np.pi * X)
        vz = np.zeros(X.shape)
        p = (gamma / (4.0 * np.pi)) * np.ones(X.shape)
        # (Az is at top-right node of each cell)
        Az = np.cos(4.0 * np.pi * X) / (4.0 * np.pi * np.sqrt(4.0 * np.pi)) + np.cos(
            2.0 * np.pi * Y
        ) / (2.0 * np.pi * np.sqrt(4.0 * np.pi))
        Bz = np.zeros(X.shape)
        b_uniform = (0.0, 0.0)
    elif prob_id == 2:
        # Circularly polarized Alfven wave travelling in -x, with B_x = 1.
        #
        # A circularly polarized wave leaves |B| (and hence rho_A) uniform and
        # constant, so it stays an *exact* nonlinear solution of the Boris
        # system -- but only if it is initialized as a Boris eigenmode:
        #
        #     dv_perp = +dB_perp / sqrt(rho + rho_A) = gamma_A dB_perp/sqrt(rho)
        #
        # The ideal-MHD relation dv_perp = dB_perp/sqrt(rho) is *not* a single
        # travelling wave of the Boris system.  It splits into counter-moving
        # Alfven waves of amplitude (1 +/- gamma_A)/2 whose beating makes |Bz|
        # swing up to amp/gamma_A -- an artifact of the setup, not of the
        # scheme.  Pass --no-boris-eigenmode to reproduce that.
        t_end = 5.0
        amp = 0.1
        rho = np.ones(X.shape)
        p = 0.2 * 0.1 * np.ones(X.shape)
        b_uniform = (1.0, 0.0)
        vA2 = (b_uniform[0] ** 2 + amp**2) / 1.0  # |B|^2/rho is uniform
        gamma_A = 1.0 / np.sqrt(1.0 + vA2 / c_limit**2) if boris_eigenmode else 1.0
        vx = np.zeros(X.shape)
        vy = gamma_A * amp * np.sin(2.0 * np.pi * X)  # !!!
        vz = gamma_A * amp * np.cos(2.0 * np.pi * X)  # !!!
        Az = amp / (2.0 * np.pi) * np.cos(2.0 * np.pi * X)
        Bz = amp * np.cos(2.0 * np.pi * X)
    elif prob_id == 3:
        # Magnetic Field Loop Test
        t_end = 1.0
        rho = np.ones(X.shape)
        p = np.ones(X.shape)
        vx = np.ones(X.shape) * np.sin(np.pi / 3.0)
        vy = np.ones(X.shape) * np.cos(np.pi / 3.0)
        vz = np.zeros(X.shape)
        Bz = np.zeros(X.shape)
        anorm = 1.0e-3
        r0 = 0.3
        r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
        Az = np.maximum(anorm * (r0 - r), 0)
        b_uniform = (0.0, 0.0)
    elif prob_id == 4:
        # Orszag-Tang with a strong uniform out-of-plane guide field.
        #
        # This is the regime the Boris correction is actually *for*: the guide
        # field makes the plasma low-beta, so vA >> max(c_s, |v|) and capping
        # the Alfven speed buys a real timestep while the in-plane dynamics is
        # still set by the flow.  Plain Orszag-Tang (problem 1) has vA ~ c_s ~
        # |v|, so there is nothing for Boris to win there.
        t_end = 0.5
        Bz0 = 3.0
        rho = (gamma**2 / (4.0 * np.pi)) * np.ones(X.shape)
        vx = -np.sin(2.0 * np.pi * Y)
        vy = np.sin(2.0 * np.pi * X)
        vz = np.zeros(X.shape)
        p = (gamma / (4.0 * np.pi)) * np.ones(X.shape)
        Az = np.cos(4.0 * np.pi * X) / (4.0 * np.pi * np.sqrt(4.0 * np.pi)) + np.cos(
            2.0 * np.pi * Y
        ) / (2.0 * np.pi * np.sqrt(4.0 * np.pi))
        Bz = Bz0 * np.ones(X.shape)
        b_uniform = (0.0, 0.0)
    else:
        raise ValueError(f"Problem ID {prob_id} not recognized")

    return rho, vx, vy, vz, p, Bz, Az, b_uniform, t_end


def get_timestep(rho, vx, vy, vz, p, Bx, By, Bz, gamma, dx, c_limit, courant_fac):
    """
    CFL timestep from the Boris-corrected fast magnetosonic speed.  The
    directional fast speeds (along x and along y) are used rather than the
    isotropic upper bound sqrt(a^2 + vA^2), which is what the wave fan of the
    Riemann problem actually sees.
    """
    cf_x, _ = get_boris_speeds(rho, p, Bx, By, Bz, gamma, c_limit)  # !!!
    cf_y, _ = get_boris_speeds(rho, p, By, Bz, Bx, gamma, c_limit)  # !!!
    cf = np.maximum(cf_x, cf_y)

    return courant_fac * np.min(dx / (cf + np.sqrt(vx**2 + vy**2)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="2.5D constrained-transport MHD with a Boris correction"
    )
    parser.add_argument(
        "prob_id",
        type=int,
        help="1 = Orszag-Tang, 2 = Alfven wave, 3 = field loop, "
        "4 = Orszag-Tang with a low-beta guide field",
    )
    parser.add_argument(
        "c_limit",
        type=float,
        help="Boris speed limit (use 'inf' to disable the correction)",
    )
    parser.add_argument("-N", "--resolution", type=int, default=128)
    parser.add_argument(
        "-s",
        "--riemann-solver",
        default="hll",
        choices=sorted(RIEMANN_SOLVERS),
        help="hlld is vanilla MHD only (requires c_limit = inf)",
    )
    parser.add_argument("--courant", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument(
        "--no-slope-limiting", dest="slope_limiting", action="store_false"
    )
    parser.add_argument(
        "--boost",
        type=float,
        default=0.0,
        help="add a uniform vx to the initial condition.  An exactly Galilean-"
        "invariant scheme gives the same solution in the boosted frame; the "
        "Boris correction does not, unless --galilean is set",
    )
    parser.add_argument(
        "--galilean",
        action="store_true",
        help="carry the Boris inertia through the momentum flux as well, which "
        "restores Galilean invariance (see get_state_1d); departs from "
        "Matsumoto et al. equations (1)-(8)",
    )
    parser.add_argument(
        "--no-boris-eigenmode",
        dest="boris_eigenmode",
        action="store_false",
        help="problem 2: use the ideal-MHD polarization instead of the Boris "
        "eigenmode, which makes the wave beat (see get_initial_conditions)",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot_in_real_time",
        action="store_false",
        help="skip the live plot (much faster for batch runs)",
    )
    parser.add_argument("-o", "--output-dir", default="output")
    opts = parser.parse_args(argv)

    if opts.riemann_solver == "hlld" and np.isfinite(opts.c_limit):
        parser.error(
            "the hlld solver does not carry the Boris correction; "
            "use --riemann-solver hll or llf for a finite c_limit"
        )
    return opts


def main(argv=None):
    """MHD Simulation"""

    opts = parse_args(argv)
    prob_id = opts.prob_id
    c_limit = opts.c_limit
    N = opts.resolution

    t = 0.0
    t_out = 0.01  # draw frequency

    # Mesh
    boxsize = 1.0
    dx = boxsize / N
    vol = dx**2
    xlin = np.linspace(0.5 * dx, boxsize - 0.5 * dx, N)
    X, Y = np.meshgrid(xlin, xlin, indexing="ij")

    # Generate Initial Conditions:
    gamma = 5.0 / 3.0  # ideal gas gamma
    rho, vx, vy, vz, p, Bz, Az, b_uniform, t_end = get_initial_conditions(
        prob_id, gamma, X, Y, c_limit, opts.boris_eigenmode
    )
    if opts.t_end is not None:
        t_end = opts.t_end
    vx = vx + opts.boost

    # a uniform field is divergence-free on its own, so it can simply be added
    # on top of the curl of Az (which need not be periodic for it)
    bx, by = get_curl(Az, dx)
    bx = bx + b_uniform[0]
    by = by + b_uniform[1]
    Bx, By = get_avg(bx, by)

    # Get conserved variables
    Mass, Momx, Momy, Momz, Energy = get_conserved(
        rho, vx, vy, vz, p, Bx, By, Bz, gamma, vol, c_limit
    )

    # keep track of timesteps
    dt_save = []
    warned_fast = False
    warned_pressure = False

    # prep figure
    plt.figure(figsize=(4, 4), dpi=80)
    out_count = 1

    # Simulation Main Loop
    while t < t_end:
        # get Primitive variables
        Bx, By = get_avg(bx, by)
        rho, vx, vy, vz, p = get_primitive(
            Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol, c_limit
        )

        # get time step (CFL) = dx / max signal speed
        dt = get_timestep(
            rho, vx, vy, vz, p, Bx, By, Bz, gamma, dx, c_limit, opts.courant
        )
        # what the timestep would have been without the Boris correction: the
        # ratio is the speedup Boris is actually buying right now.  It is < the
        # local Boris factor alpha whenever the flow speed, not the fast speed,
        # sets the CFL limit -- which is the case for most of Orszag-Tang.
        dt_mhd = get_timestep(
            rho, vx, vy, vz, p, Bx, By, Bz, gamma, dx, np.inf, opts.courant
        )
        dt = min(dt, t_end - t)

        c0 = np.sqrt(gamma * np.maximum(p, SMALL_P) / rho)
        ca = np.sqrt((Bx**2 + By**2 + Bz**2) / rho)
        cf = np.sqrt(c0**2 + ca**2)
        alpha = 1.0 / np.sqrt(1.0 + (ca / c_limit) ** 2)
        v_max = np.max(np.sqrt(vx**2 + vy**2 + vz**2))

        plot_this_turn = False
        if t + dt > out_count * t_out:
            plot_this_turn = True

        # first RK stage
        (
            dudt_Mass,
            dudt_Momx,
            dudt_Momy,
            dudt_Momz,
            dudt_Energy,
            dudt_Bz,
            dudt_bx,
            dudt_by,
        ) = get_dudt(rho, vx, vy, vz, p, bx, by, Bz, dx, gamma, c_limit, opts)

        # update solution
        Mass_1 = apply_dudt(Mass, dudt_Mass, dt)
        Momx_1 = apply_dudt(Momx, dudt_Momx, dt)
        Momy_1 = apply_dudt(Momy, dudt_Momy, dt)
        Momz_1 = apply_dudt(Momz, dudt_Momz, dt)
        Energy_1 = apply_dudt(Energy, dudt_Energy, dt)
        Bz_1 = apply_dudt(Bz, dudt_Bz, dt)
        bx_1 = apply_dudt(bx, dudt_bx, dt)
        by_1 = apply_dudt(by, dudt_by, dt)

        # start accumulating the n+1 result u^n+1 = [u^n + dt/2 L(u^n)] + dt/2 L(u^1)
        Mass = apply_dudt(Mass, dudt_Mass, 0.5 * dt)
        Momx = apply_dudt(Momx, dudt_Momx, 0.5 * dt)
        Momy = apply_dudt(Momy, dudt_Momy, 0.5 * dt)
        Momz = apply_dudt(Momz, dudt_Momz, 0.5 * dt)
        Energy = apply_dudt(Energy, dudt_Energy, 0.5 * dt)
        Bz = apply_dudt(Bz, dudt_Bz, 0.5 * dt)
        bx = apply_dudt(bx, dudt_bx, 0.5 * dt)
        by = apply_dudt(by, dudt_by, 0.5 * dt)

        # second stage: evaluated entirely on the predictor state u^1
        Bx_1, By_1 = get_avg(bx_1, by_1)
        rho_1, vx_1, vy_1, vz_1, p_1 = get_primitive(
            Mass_1,
            Momx_1,
            Momy_1,
            Momz_1,
            Energy_1,
            Bx_1,
            By_1,
            Bz_1,
            gamma,
            vol,
            c_limit,
        )
        (
            dudt_Mass,
            dudt_Momx,
            dudt_Momy,
            dudt_Momz,
            dudt_Energy,
            dudt_Bz,
            dudt_bx,
            dudt_by,
        ) = get_dudt(
            rho_1, vx_1, vy_1, vz_1, p_1, bx_1, by_1, Bz_1, dx, gamma, c_limit, opts
        )

        Mass = apply_dudt(Mass, dudt_Mass, 0.5 * dt)
        Momx = apply_dudt(Momx, dudt_Momx, 0.5 * dt)
        Momy = apply_dudt(Momy, dudt_Momy, 0.5 * dt)
        Momz = apply_dudt(Momz, dudt_Momz, 0.5 * dt)
        Energy = apply_dudt(Energy, dudt_Energy, 0.5 * dt)
        Bz = apply_dudt(Bz, dudt_Bz, 0.5 * dt)
        bx = apply_dudt(bx, dudt_bx, 0.5 * dt)
        by = apply_dudt(by, dudt_by, 0.5 * dt)

        # update time
        t += dt

        # save timestep
        dt_save.append(dt)

        # The Boris correction is a semi-relativistic approximation: it is only
        # valid while the flow is slow compared to the reduced light speed.
        # Matsumoto et al. (2019) find stability breaks down around |v| ~ 0.5c.
        if v_max > 0.5 * c_limit and not warned_fast:
            print(
                f"WARNING: v_max = {v_max:.3f} > 0.5 * c_limit = "
                f"{0.5 * c_limit:.3f}; the Boris correction is outside its "
                "regime of validity and the solution may be badly wrong"
            )
            warned_fast = True
        if np.min(p) <= 0.0 and not warned_pressure:
            print(f"WARNING: negative gas pressure ({np.min(p):.3e})")
            warned_pressure = True

        # check div B
        divB = get_div(bx, by, dx)
        print(
            "t=",
            f"{t:.4f}",
            " c0_max =",
            f"{np.max(c0):.4f}",
            " cf_max =",
            f"{np.max(cf):.4f}",
            " v_max=",
            f"{v_max:.4f}",
            " alpha=",
            f"{np.min(alpha):.4f}",
            " dt/dt_mhd=",
            f"{dt / dt_mhd:.4f}",
            " p_min=",
            f"{np.min(p):.3e}",
            " |divB|=",
            f"{np.mean(np.abs(divB)):.4f}",
            " |Bz|=",
            f"{np.mean(np.abs(Bz)):.4f}",
        )

        # plot in real time
        if (opts.plot_in_real_time and plot_this_turn) or (t >= t_end):
            plt.cla()
            if prob_id in (1, 4):
                plt.imshow(np.sqrt(bx**2 + by**2).T, cmap="jet")
                plt.clim(0.0, 0.8)
            elif prob_id == 2:
                plt.plot(Bz[:, N // 2])
                plt.ylim(-0.2, 0.2)
            elif prob_id == 3:
                plt.imshow(np.sqrt(bx**2 + by**2).T, cmap="jet")
                plt.clim(0.0, 0.0011)
            ax = plt.gca()
            if prob_id != 2:
                ax.invert_yaxis()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_aspect("equal")
            if opts.plot_in_real_time:
                plt.pause(0.001)
            out_count += 1

    print("done!")

    os.makedirs(opts.output_dir, exist_ok=True)
    prefix = os.path.join(opts.output_dir, "p" + str(prob_id) + "_")
    tag = str(c_limit)

    # Save figure
    plt.savefig(prefix + "P_B_" + tag + ".png", dpi=240)

    # Save rho, P_B, v, vA, cf, and dt_save
    Bx, By = get_avg(bx, by)
    rho, vx, vy, vz, p = get_primitive(
        Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol, c_limit
    )
    P_B = 0.5 * np.sqrt(Bx**2 + By**2 + Bz**2)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    c0 = np.sqrt(gamma * np.maximum(p, SMALL_P) / rho)
    ca = np.sqrt((Bx**2 + By**2 + Bz**2) / rho)
    cf = np.sqrt(c0**2 + ca**2)
    np.save(prefix + "data_rho_" + tag + ".npy", rho.T)
    np.save(prefix + "data_P_B_" + tag + ".npy", P_B.T)
    np.save(prefix + "data_Bz_" + tag + ".npy", Bz.T)
    np.save(prefix + "data_v_" + tag + ".npy", v.T)
    np.save(prefix + "data_ca_" + tag + ".npy", ca.T)
    np.save(prefix + "data_cf_" + tag + ".npy", cf.T)
    np.save(prefix + "data_dt_" + tag + ".npy", dt_save)


if __name__ == "__main__":
    main()
