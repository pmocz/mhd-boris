import matplotlib.pyplot as plt
import numpy as np
import sys

"""
2.5D Constrained Transport Magnetohydrodynamics
Philip Mocz, @PMocz
Adam Reyes
(2024)

Simulate the Orszag-Tang vortex MHD problem (+more)
with a Boris-like Integrator to control timesteps!

The original problem has cf_max ~ 1.9, u_max ~ 1.6
"""

# Simulation parameters (global)
N = 128  # resolution
boxsize = 1.0
gamma = 5.0 / 3.0  # ideal gas gamma
courant_fac = 0.4
t_end = 0.5
t_out = 0.01  # draw frequency
use_slope_limiting = True
plot_in_real_time = True
riemann_solver = "llf"

# directions for np.roll()
#  -1    right/up
#   1    left/down


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
    Calculate the discrete curl of each cell
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


def get_conserved(rho, vx, vy, vz, P, Bx, By, Bz, gamma, vol):
    """
    Calculate the conserved variable from the primitive
    rho      is matrix of cell densities
    vx       is matrix of cell x-velocity
    vy       is matrix of cell y-velocity
    P        is matrix of cell Total pressures
    Bx       is matrix of cell Bx
    By       is matrix of cell By
    gamma    is ideal gas gamma
    vol      is cell volume
    Mass     is matrix of mass in cells
    Momx     is matrix of x-momentum in cells
    Momy     is matrix of y-momentum in cells
    Energy   is matrix of energy in cells
    """
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Momz = rho * vz * vol
    Energy = (
        (P - 0.5 * (Bx**2 + By**2 + Bz**2)) / (gamma - 1.0)
        + 0.5 * rho * (vx**2 + vy**2 + vz**2)
        + 0.5 * (Bx**2 + By**2 + Bz**2)
    ) * vol

    return Mass, Momx, Momy, Momz, Energy


def get_primitive(Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol, cf_limit):
    """
    Calculate the primitive variable from the conservative
    Mass     is matrix of mass in cells
    Momx     is matrix of x-momentum in cells
    Momy     is matrix of y-momentum in cells
    Energy   is matrix of energy in cells
    Bx       is matrix of cell Bx
    By       is matrix of cell By
    gamma    is ideal gas gamma
    vol      is cell volume
    rho      is matrix of cell densities
    vx       is matrix of cell x-velocity
    vy       is matrix of cell y-velocity
    P        is matrix of cell Total pressures
    """
    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol
    vz = Momz / rho / vol
    P = (
        Energy / vol
        - 0.5 * rho * (vx**2 + vy**2 + vz**2)
        - 0.5 * (Bx**2 + By**2 + Bz**2)
    ) * (gamma - 1.0) + 0.5 * (Bx**2 + By**2 + Bz**2)

    # Try 2: apply boris factor in recovering the velocity
    # c0 = np.sqrt( gamma*(P-0.5*(Bx**2+By**2+Bz**2))/rho )
    c0 = np.sqrt(gamma * (np.maximum(P - 0.5 * (Bx**2 + By**2 + Bz**2), 1.0e-16)) / rho)
    ca = np.sqrt((Bx**2 + By**2 + Bz**2) / rho)
    cf = np.sqrt(c0**2 + ca**2)
    alpha = np.minimum(1.0, cf_limit / cf)
    vx *= alpha
    vy *= alpha

    return rho, vx, vy, vz, P


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
    Calculate the gradients of a field
    f        is a matrix of the field
    f_dx     is a matrix of the field x-derivatives
    f_dy     is a matrix of the field y-derivatives
    dx       is the cell size
    f_XL     is a matrix of spatial-extrapolated values on `left' face along x-axis
    f_XR     is a matrix of spatial-extrapolated values on `right' face along x-axis
    f_YL     is a matrix of spatial-extrapolated values on `left' face along y-axis
    f_YR     is a matrix of spatial-extrapolated values on `right' face along y-axis
    """

    f_XL = f - f_dx * dx / 2.0
    f_XL = np.roll(f_XL, -1, axis=0)
    f_XR = f + f_dx * dx / 2.0

    f_YL = f - f_dy * dx / 2.0
    f_YL = np.roll(f_YL, -1, axis=1)
    f_YR = f + f_dy * dx / 2.0

    return f_XL, f_XR, f_YL, f_YR


def dudt_fluxes(flux_X, flux_Y, dx):
    F = -dx * flux_X
    F += dx * np.roll(flux_X, 1, axis=0)
    F += -dx * flux_Y
    F += dx * np.roll(flux_Y, 1, axis=1)
    return F


def apply_fluxes(F, flux_F_X, flux_F_Y, dx, dt):
    """
    Apply fluxes to conserved variables
    F        is a matrix of the conserved variable field
    flux_F_X is a matrix of the x-dir fluxes
    flux_F_Y is a matrix of the y-dir fluxes
    dx       is the cell size
    dt       is the timestep
    """

    # update solution
    F += -dt * dx * flux_F_X
    F += dt * dx * np.roll(flux_F_X, 1, axis=0)
    F += -dt * dx * flux_F_Y
    F += dt * dx * np.roll(flux_F_Y, 1, axis=1)

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


# HLLD (adapted from Athena)
def get_flux_hlld(
    rho_L,
    rho_R,
    vx_L,
    vx_R,
    vy_L,
    vy_R,
    vz_L,
    vz_R,
    P_L,
    P_R,
    Bx_L,
    Bx_R,
    By_L,
    By_R,
    Bz_L,
    Bz_R,
    gamma,
    cf_limit,
):
    """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule
    rho_L        is a matrix of left-state  density
    rho_R        is a matrix of right-state density
    vx_L         is a matrix of left-state  x-velocity
    vx_R         is a matrix of right-state x-velocity
    vy_L         is a matrix of left-state  y-velocity
    vy_R         is a matrix of right-state y-velocity
    P_L          is a matrix of left-state  Total pressure
    P_R          is a matrix of right-state Total pressure
    Bx_L         is a matrix of left-state  x-magnetic-field
    Bx_R         is a matrix of right-state x-magnetic-field
    By_L         is a matrix of left-state  y-magnetic-field
    By_R         is a matrix of right-state y-magnetic-field
    gamma        is the ideal gas gamma
    flux_Mass    is the matrix of mass fluxes
    flux_Momx    is the matrix of x-momentum fluxes
    flux_Momy    is the matrix of y-momentum fluxes
    flux_Energy  is the matrix of energy fluxes
    """

    SMALL_NUMBER = 1.0e-8

    P_L -= 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2)
    P_R -= 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2)

    Bxi = 0.5 * (Bx_L + Bx_R)

    Mx_L = rho_L * vx_L
    My_L = rho_L * vy_L
    Mz_L = rho_L * vz_L
    E_L = (
        P_L / (gamma - 1.0)
        + 0.5 * rho_L * (vx_L**2 + vy_L**2 + vz_L**2)
        + 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2)
    )

    Mx_R = rho_R * vx_R
    My_R = rho_R * vy_R
    Mz_R = rho_R * vz_R
    E_R = (
        P_R / (gamma - 1.0)
        + 0.5 * rho_R * (vx_R**2 + vy_R**2 + vz_R**2)
        + 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2)
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
    # WLst_d = ULst_d
    # WLst_vx = ULst_Mx / ULst_d
    WLst_vy = ULst_My / ULst_d
    WLst_vz = ULst_Mz / ULst_d
    # WLst_Bx = ULst_Bx
    # WLst_By = ULst_By
    # WLst_Bz = ULst_Bz
    # WLst_P = (gamma - 1.0) * (
    #    ULst_E
    #    - 0.5 * WLst_d * (WLst_vx**2 + WLst_vy**2 + WLst_vz**2)
    #    - 0.5 * (WLst_Bx**2 + WLst_By**2 + WLst_Bz**2)
    # )

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
    # WRst_d = URst_d
    # WRst_vx = URst_Mx / URst_d
    WRst_vy = URst_My / URst_d
    WRst_vz = URst_Mz / URst_d
    # WRst_Bx = URst_Bx
    # WRst_By = URst_By
    # WRst_Bz = URst_Bz
    # WRst_P = (gamma - 1) * (
    #    URst_E
    #    - 0.5 * WRst_d * (WRst_vx**2 + WRst_vy**2 + WRst_vz**2)
    #    - 0.5 * (WRst_Bx**2 + WRst_By**2 + WRst_Bz**2)
    # )

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
    Bxsig = 0 * Bxi - 1
    Bxsig[Bxi > 0] = 1

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


def get_flux_llf(
    rho_L,
    rho_R,
    vx_L,
    vx_R,
    vy_L,
    vy_R,
    vz_L,
    vz_R,
    P_L,
    P_R,
    Bx_L,
    Bx_R,
    By_L,
    By_R,
    Bz_L,
    Bz_R,
    gamma,
    cf_limit,
):
    """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule
    rho_L        is a matrix of left-state  density
    rho_R        is a matrix of right-state density
    vx_L         is a matrix of left-state  x-velocity
    vx_R         is a matrix of right-state x-velocity
    vy_L         is a matrix of left-state  y-velocity
    vy_R         is a matrix of right-state y-velocity
    P_L          is a matrix of left-state  Total pressure
    P_R          is a matrix of right-state Total pressure
    Bx_L         is a matrix of left-state  x-magnetic-field
    Bx_R         is a matrix of right-state x-magnetic-field
    By_L         is a matrix of left-state  y-magnetic-field
    By_R         is a matrix of right-state y-magnetic-field
    gamma        is the ideal gas gamma
    flux_Mass    is the matrix of mass fluxes
    flux_Momx    is the matrix of x-momentum fluxes
    flux_Momy    is the matrix of y-momentum fluxes
    flux_Energy  is the matrix of energy fluxes
    """

    # left and right energies
    en_L = (
        (P_L - 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2)) / (gamma - 1.0)
        + 0.5 * rho_L * (vx_L**2 + vy_L**2 + vz_L**2)
        + 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2)
    )
    en_R = (
        (P_R - 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2)) / (gamma - 1.0)
        + 0.5 * rho_R * (vx_R**2 + vy_R**2 + vz_R**2)
        + 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2)
    )

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    momz_star = 0.5 * (rho_L * vz_L + rho_R * vz_R)
    en_star = 0.5 * (en_L + en_R)
    Bx_star = 0.5 * (Bx_L + Bx_R)
    By_star = 0.5 * (By_L + By_R)
    Bz_star = 0.5 * (Bz_L + Bz_R)

    P_star = (gamma - 1.0) * (
        en_star
        - 0.5 * (momx_star**2 + momy_star**2 + momz_star**2) / rho_star
        - 0.5 * (Bx_star**2 + By_star**2 + Bz_star**2)
    ) + 0.5 * (Bx_star**2 + By_star**2 + Bz_star**2)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star - Bx_star * Bx_star
    flux_Momy = momx_star * momy_star / rho_star - Bx_star * By_star
    flux_Momz = momx_star * momz_star / rho_star - Bx_star * Bz_star
    flux_Energy = (en_star + P_star) * momx_star / rho_star - Bx_star * (
        Bx_star * momx_star + By_star * momy_star + Bz_star * momz_star
    ) / rho_star
    flux_By = (By_star * momx_star - Bx_star * momy_star) / rho_star
    flux_Bz = (Bz_star * momx_star - Bx_star * momz_star) / rho_star

    # find wavespeeds
    # c0_L = np.sqrt( gamma*(P_L-0.5*(Bx_L**2+By_L**2+Bz_L**2))/rho_L )
    # c0_R = np.sqrt( gamma*(P_R-0.5*(Bx_R**2+By_R**2+Bz_R**2))/rho_R )
    c0_L = np.sqrt(
        gamma * (np.maximum(P_L - 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2), 1.0e-16)) / rho_L
    )
    c0_R = np.sqrt(
        gamma * (np.maximum(P_R - 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2), 1.0e-16)) / rho_R
    )
    ca_L = np.sqrt((Bx_L**2 + By_L**2 + Bz_L**2) / rho_L)
    ca_R = np.sqrt((Bx_R**2 + By_R**2 + Bz_R**2) / rho_R)
    cf_L = np.sqrt(c0_L**2 + ca_L**2)
    cf_R = np.sqrt(c0_R**2 + ca_R**2)
    # apply boris factor to wave speeds and momentum flux
    # alpha_L = np.minimum(1.0, cf_limit / cf_L)
    # alpha_R = np.minimum(1.0, cf_limit / cf_R)
    # alpha = np.minimum(alpha_L, alpha_R)
    # alphaSq = alpha ** 2
    ## Try 1
    # flux_Momx *= alphaSq
    # flux_Momy *= alphaSq
    # cf_L *= alpha_L
    # cf_R *= alpha_R

    C_L = cf_L + np.abs(vx_L)
    C_R = cf_R + np.abs(vx_R)
    C = np.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_R - rho_L)
    flux_Momx -= C * 0.5 * (rho_R * vx_R - rho_L * vx_L)
    flux_Momy -= C * 0.5 * (rho_R * vy_R - rho_L * vy_L)
    flux_Momz -= C * 0.5 * (rho_R * vz_R - rho_L * vz_L)
    flux_Energy -= C * 0.5 * (en_R - en_L)
    flux_By -= C * 0.5 * (By_R - By_L)
    flux_Bz -= C * 0.5 * (Bz_R - Bz_L)

    return flux_Mass, flux_Momx, flux_Momy, flux_Momz, flux_Energy, flux_By, flux_Bz


def get_flux(
    method,
    rho_L,
    rho_R,
    vx_L,
    vx_R,
    vy_L,
    vy_R,
    vz_L,
    vz_R,
    P_L,
    P_R,
    Bx_L,
    Bx_R,
    By_L,
    By_R,
    Bz_L,
    Bz_R,
    gamma,
    cf_limit,
):
    if method == "llf":
        return get_flux_llf(
            rho_L,
            rho_R,
            vx_L,
            vx_R,
            vy_L,
            vy_R,
            vz_L,
            vz_R,
            P_L,
            P_R,
            Bx_L,
            Bx_R,
            By_L,
            By_R,
            Bz_L,
            Bz_R,
            gamma,
            cf_limit,
        )
    elif method == "hlld":
        return get_flux_hlld(
            rho_L,
            rho_R,
            vx_L,
            vx_R,
            vy_L,
            vy_R,
            vz_L,
            vz_R,
            P_L,
            P_R,
            Bx_L,
            Bx_R,
            By_L,
            By_R,
            Bz_L,
            Bz_R,
            gamma,
            cf_limit,
        )
    else:
        raise ValueError(f"Unknown method {method}")


def get_dudt(rho, vx, vy, vz, P, bx, by, Bz, dx, dy, gamma, cf_limit):
    """single stage of a runge-kutta method to return dudt of all these variables"""
    Bx, By = get_avg(bx, by)

    rho_dx, rho_dy = get_gradient(rho, dx)
    vx_dx, vx_dy = get_gradient(vx, dx)
    vy_dx, vy_dy = get_gradient(vy, dx)
    vz_dx, vz_dy = get_gradient(vz, dx)
    P_dx, P_dy = get_gradient(P, dx)
    Bx_dx, Bx_dy = get_gradient(Bx, dx)
    By_dx, By_dy = get_gradient(By, dx)
    Bz_dx, Bz_dy = get_gradient(Bz, dx)

    # slope limit gradients
    if use_slope_limiting:
        rho_dx, rho_dy = slope_limit(rho, dx, rho_dx, rho_dy)
        vx_dx, vx_dy = slope_limit(vx, dx, vx_dx, vx_dy)
        vy_dx, vy_dy = slope_limit(vy, dx, vy_dx, vy_dy)
        vz_dx, vz_dy = slope_limit(vz, dx, vz_dx, vz_dy)
        P_dx, P_dy = slope_limit(P, dx, P_dx, P_dy)
        Bx_dx, Bx_dy = slope_limit(Bx, dx, Bx_dx, Bx_dy)
        By_dx, By_dy = slope_limit(By, dx, By_dx, By_dy)
        Bz_dx, Bz_dy = slope_limit(Bz, dx, Bz_dx, Bz_dy)

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR = extrapolate_in_space_to_face(
        rho, rho_dx, rho_dy, dx
    )
    vx_XL, vx_XR, vx_YL, vx_YR = extrapolate_in_space_to_face(vx, vx_dx, vx_dy, dx)
    vy_XL, vy_XR, vy_YL, vy_YR = extrapolate_in_space_to_face(vy, vy_dx, vy_dy, dx)
    vz_XL, vz_XR, vz_YL, vz_YR = extrapolate_in_space_to_face(vz, vz_dx, vz_dy, dx)
    P_XL, P_XR, P_YL, P_YR = extrapolate_in_space_to_face(P, P_dx, P_dy, dx)
    Bx_XL, Bx_XR, Bx_YL, Bx_YR = extrapolate_in_space_to_face(Bx, Bx_dx, Bx_dy, dx)
    By_XL, By_XR, By_YL, By_YR = extrapolate_in_space_to_face(By, By_dx, By_dy, dx)
    Bz_XL, Bz_XR, Bz_YL, Bz_YR = extrapolate_in_space_to_face(Bz, Bz_dx, Bz_dy, dx)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    (
        flux_Mass_X,
        flux_Momx_X,
        flux_Momy_X,
        flux_Momz_X,
        flux_Energy_X,
        flux_By_X,
        flux_Bz_X,
    ) = get_flux(
        riemann_solver,
        rho_XR,
        rho_XL,
        vx_XR,
        vx_XL,
        vy_XR,
        vy_XL,
        vz_XR,
        vz_XL,
        P_XR,
        P_XL,
        Bx_XR,
        Bx_XL,
        By_XR,
        By_XL,
        Bz_XR,
        Bz_XL,
        gamma,
        cf_limit,
    )
    (
        flux_Mass_Y,
        flux_Momy_Y,
        flux_Momz_Y,
        flux_Momx_Y,
        flux_Energy_Y,
        flux_Bz_Y,
        flux_Bx_Y,
    ) = get_flux(
        riemann_solver,
        rho_YR,
        rho_YL,
        vy_YR,
        vy_YL,
        vz_YR,
        vz_YL,
        vx_YR,
        vx_YL,
        P_YR,
        P_YL,
        By_YR,
        By_YL,
        Bz_YR,
        Bz_YL,
        Bx_YR,
        Bx_YL,
        gamma,
        cf_limit,
    )
    dudt_Mass = dudt_fluxes(flux_Mass_X, flux_Mass_Y, dx)
    dudt_Momx = dudt_fluxes(flux_Momx_X, flux_Momx_Y, dx)
    dudt_Momy = dudt_fluxes(flux_Momy_X, flux_Momy_Y, dx)
    dudt_Momz = dudt_fluxes(flux_Momz_X, flux_Momz_Y, dx)
    dudt_Energy = dudt_fluxes(flux_Energy_X, flux_Energy_Y, dx)
    dudt_Bz = dudt_fluxes(flux_Bz_X, flux_Bz_Y, dx)
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


def main():
    """MHD Simulation"""

    # Check for command line arguments
    if len(sys.argv) != 3:
        print("Usage: python mhd-boris.py <problem_id> <cf_limit>")
        return

    # Parse command line argument
    # problem id & fast speed limit for boris integrator (e.g. try 1.0, 1.5, 2.0)
    prob_id = int(sys.argv[1])
    cf_limit = float(sys.argv[2])

    t = 0.0
    global \
        N, \
        boxsize, \
        gamma, \
        courant_fac, \
        t_end, \
        t_out, \
        use_slope_limiting, \
        riemann_solver, \
        plot_in_real_time

    # Mesh
    dx = boxsize / N
    vol = dx**2
    xlin = np.linspace(0.5 * dx, boxsize - 0.5 * dx, N)
    X, Y = np.meshgrid(xlin, xlin, indexing="ij")
    # xlin_node = np.linspace(dx, boxsize, N)
    # Xn, Yn = np.meshgrid( xlin_node, xlin_node, indexing="ij" )

    # Generate Initial Conditions:
    if prob_id == 1:
        # Orszag-Tang vortex problem
        riemann_solver = "hlld"
        rho = (gamma**2 / (4.0 * np.pi)) * np.ones(X.shape)
        vx = -np.sin(2.0 * np.pi * Y)
        vy = np.sin(2.0 * np.pi * X)
        P = (gamma / (4.0 * np.pi)) * np.ones(X.shape)  # init. gas pressure
        # (Az is at top-right node of each cell)
        Az = np.cos(4.0 * np.pi * X) / (4.0 * np.pi * np.sqrt(4.0 * np.pi)) + np.cos(
            2.0 * np.pi * Y
        ) / (2.0 * np.pi * np.sqrt(4.0 * np.pi))
        vz = np.zeros(X.shape)
        Bz = np.zeros(X.shape)
    elif prob_id == 2:
        # Circularly polarized Alfven wave
        rho = np.ones(X.shape)
        P = 0.1 * np.ones(X.shape)  # init. gas pressure
        amp = 0.1
        t_end = 4.0

        alpha = np.pi / 4.0
        Xpar = (np.cos(alpha) * X + np.sin(alpha) * Y) * np.sqrt(2.0)
        v_perp = 0.1 * np.sin(2.0 * np.pi * Xpar)
        v_par = np.zeros(X.shape)
        # b_perp = 0.1 * np.sin(2.0*np.pi*Xpar)
        vx = v_par * np.cos(alpha) - v_perp * np.sin(alpha)
        vy = v_par * np.sin(alpha) + v_perp * np.cos(alpha)
        Az = 0.1 / (2.0 * np.pi) * np.cos(2.0 * np.pi * Xpar)
        vz = 0.1 * np.cos(2.0 * np.pi * Xpar)
        Bz = 0.1 * np.cos(2.0 * np.pi * Xpar)

        # simplified ICs
        # TODO: switch to using the above ones instead XXX
        vx = np.zeros(X.shape)
        vy = amp * np.sin(2.0 * np.pi * X)
        vz = amp * np.cos(2.0 * np.pi * X)
        Az = amp / (2.0 * np.pi) * np.cos(2.0 * np.pi * X)
        Bz = amp * np.cos(2.0 * np.pi * X)

    elif prob_id == 3:
        # Magnetic Field Loop Test
        rho = np.ones(X.shape)
        P = np.ones(X.shape)
        vx = np.ones(X.shape) * np.sin(np.pi / 3.0)
        vy = np.ones(X.shape) * np.cos(np.pi / 3.0)
        vz = np.zeros(X.shape)
        Bz = np.zeros(X.shape)
        anorm = 1.0e-3
        r0 = 0.3
        r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
        Az = np.maximum(anorm * (r0 - r), 0)
        # courant_fac = courant_fac / 10.0
    else:
        print("Problem ID not recognized")
        return

    bx, by = get_curl(Az, dx)
    Bx, By = get_avg(bx, by)

    # add magnetic pressure to get the total pressure
    P = P + 0.5 * (Bx**2 + By**2 + Bz**2)

    # Get conserved variables
    Mass, Momx, Momy, Momz, Energy = get_conserved(
        rho, vx, vy, vz, P, Bx, By, Bz, gamma, vol
    )

    # keep track of timesteps
    dt_sav = []

    # prep figure
    plt.figure(figsize=(4, 4), dpi=80)
    out_count = 1

    # Simulation Main Loop
    while t < t_end:
        # get Primitive variables
        Bx, By = get_avg(bx, by)
        rho, vx, vy, vz, P = get_primitive(
            Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol, cf_limit
        )

        # get time step (CFL) = dx / max signal speed
        # c0 = np.sqrt( gamma*(P-0.5*(Bx**2+By**2+Bz**2))/rho)
        c0 = np.sqrt(
            gamma * (np.maximum(P - 0.5 * (Bx**2 + By**2 + Bz**2), 1.0e-16)) / rho
        )
        ca = np.sqrt((Bx**2 + By**2 + Bz**2) / rho)
        cf = np.sqrt(c0**2 + ca**2)
        max_cf = np.max(cf)
        alpha = np.minimum(1.0, cf_limit / np.sqrt(c0**2 + ca**2))
        # Try 1
        # cf *= alpha
        dt = courant_fac * np.min(dx / (cf + np.sqrt(vx**2 + vy**2 + vz**2)))
        u_max = np.max(np.sqrt(vx**2 + vy**2 + vz**2))

        plot_this_turn = False
        if t + dt > out_count * t_out:
            # dt = out_count*t_out - t
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
        ) = get_dudt(rho, vx, vy, vz, P, bx, by, Bz, dx, dx, gamma, cf_limit)

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

        # second stage
        Bx, By = get_avg(bx, by)
        rho, vx, vy, vz, P = get_primitive(
            Mass_1, Momx_1, Momy_1, Momz_1, Energy_1, Bx, By, Bz_1, gamma, vol, cf_limit
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
        ) = get_dudt(rho, vx, vy, vz, P, bx_1, by_1, Bz, dx, dx, gamma, cf_limit)

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
        dt_sav.append(dt)

        # check div B
        divB = get_div(bx, by, dx)
        print(
            "t=",
            f"{t:.4f}",
            " max_cf =",
            f"{max_cf:.4f}",
            " u_max=",
            f"{u_max:.4f}",
            " alpha=",
            f"{np.min(alpha):.4f}",
            " mean(|divB|)=",
            f"{np.mean(np.abs(divB)):.4f}",
            " mean(|Bz|)=",
            f"{np.mean(np.abs(Bz)):.4f}",
        )

        # plot in real time
        if (plot_in_real_time and plot_this_turn) or (t >= t_end):
            plt.cla()
            # plt.imshow(rho.T, cmap='jet')
            # plt.clim(0.06, 0.5)
            if prob_id == 1:
                plt.imshow(np.sqrt(bx**2 + by**2).T, cmap="jet")
                plt.clim(0.0, 0.8)
            elif prob_id == 2:
                plt.imshow(Bz.T, cmap="jet")
                plt.clim(-0.1, 0.1)
            elif prob_id == 3:
                plt.imshow(np.sqrt(bx**2 + by**2).T, cmap="jet")
                plt.clim(0.0, 0.0011)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            plt.pause(0.001)
            out_count += 1

    print("done!")

    prefix = "output/p" + str(prob_id) + "_"

    # Save figure
    plt.savefig(prefix + "P_B_" + str(cf_limit) + ".png", dpi=240)
    # plt.show()

    # Save rho, P_B, v, vA, cf, and dt_sav
    Bx, By = get_avg(bx, by)
    rho, vx, vy, vz, P = get_primitive(
        Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol, cf_limit
    )
    P_B = 0.5 * np.sqrt(Bx**2 + By**2 + Bz**2)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    # c0 = np.sqrt( gamma*(P-0.5*(Bx**2+By**2+Bz**2))/rho )
    c0 = np.sqrt(gamma * (np.maximum(P - 0.5 * (Bx**2 + By**2 + Bz**2), 1.0e-16)) / rho)
    ca = np.sqrt((Bx**2 + By**2 + Bz**2) / rho)
    cf = np.sqrt(c0**2 + ca**2)
    np.save(prefix + "data_rho_" + str(cf_limit) + ".npy", rho.T)
    np.save(prefix + "data_P_B_" + str(cf_limit) + ".npy", P_B.T)
    np.save(prefix + "data_v_" + str(cf_limit) + ".npy", v.T)
    np.save(prefix + "data_ca_" + str(cf_limit) + ".npy", ca.T)
    np.save(prefix + "data_cf_" + str(cf_limit) + ".npy", cf.T)
    np.save(prefix + "data_dt_" + str(cf_limit) + ".npy", dt_sav)

    return


if __name__ == "__main__":
    main()
