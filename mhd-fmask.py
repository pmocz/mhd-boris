import matplotlib.pyplot as plt
import numpy as np
import sys

"""
2.5D Constrained Transport Magnetohydrodynamics
Philip Mocz, @PMocz
Adam Reyes
(2024)

Simulate the Orszag-Tang vortex MHD problem (+more)
with Force Masking

The original problem has rho_min ~ 0.05, cf_max ~ 0.05, v_max ~ 1.6
"""

# Simulation parameters (global)
N = 128  # resolution
use_slope_limiting = True

# directions for np.roll()
#  -1    right/up
#   1    left/down


def mask_function(rho, rho_limit):
    """
    Masking function to apply to the density field
    rho       is matrix of cell densities
    rho_limit is the density limit for masking
    mask      is ther masking matrix
    """
    mask = np.abs(rho) > rho_limit
    # mask = 0.5 + 0.5*np.tanh((np.log10(rho) - np.log10(rho_limit))/0.2)

    return mask


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


def get_primitive(Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol):
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


# local Lax-Friedrichs/Rusanov
def get_flux(
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
    rho_limit,
):
    """
    Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule
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
    # cf_L = np.sqrt(c0_L**2 + ca_L**2)
    # cf_R = np.sqrt(c0_R**2 + ca_R**2)

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

    mask = mask_function(rho_star, rho_limit)
    ca_L *= np.sqrt(mask)
    ca_R *= np.sqrt(mask)
    cf_L = np.sqrt(c0_L**2 + ca_L**2)
    cf_R = np.sqrt(c0_R**2 + ca_R**2)

    P_star = (gamma - 1.0) * (
        en_star
        - 0.5 * (momx_star**2 + momy_star**2 + momz_star**2) / rho_star
        - 0.5 * (Bx_star**2 + By_star**2 + Bz_star**2)
    ) + 0.5 * (Bx_star**2 + By_star**2 + Bz_star**2)

    P_star_masked = (gamma - 1.0) * (
        en_star
        - 0.5 * (momx_star**2 + momy_star**2 + momz_star**2) / rho_star
        - 0.5 * (Bx_star**2 + By_star**2 + Bz_star**2)
    ) + 0.5 * (Bx_star**2 + By_star**2 + Bz_star**2) * mask

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star_masked - Bx_star * Bx_star * mask
    flux_Momy = momx_star * momy_star / rho_star - Bx_star * By_star * mask
    flux_Momz = momx_star * momz_star / rho_star - Bx_star * Bz_star * mask
    flux_Energy = (en_star + P_star_masked) * momx_star / rho_star - Bx_star * (
        Bx_star * momx_star + By_star * momy_star + Bz_star * momz_star
    ) / rho_star * mask
    flux_By = (By_star * momx_star - Bx_star * momy_star) / rho_star
    flux_Bz = (Bz_star * momx_star - Bx_star * momz_star) / rho_star

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


def get_dudt(rho, vx, vy, vz, P, bx, by, Bz, dx, gamma, rho_limit):
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

    # compute fluxes
    (
        flux_Mass_X,
        flux_Momx_X,
        flux_Momy_X,
        flux_Momz_X,
        flux_Energy_X,
        flux_By_X,
        flux_Bz_X,
    ) = get_flux(
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
        rho_limit,
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
        rho_limit,
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


def main():
    """MHD Simulation"""

    # Check for command line arguments
    if len(sys.argv) != 3:
        print("Usage: python mhd-boris.py <problem_id> <rho_limit>")
        return

    # Parse command line argument
    # problem id & density limit for force masking (e.g. 0.05, 0.1, 0.2)
    prob_id = int(sys.argv[1])
    rho_limit = float(sys.argv[2])

    global N, use_slope_limiting
    t = 0.0
    courant_fac = 0.4
    t_end = 0.5
    t_out = 0.01  # draw frequency
    plot_in_real_time = True

    # Mesh
    boxsize = 1.0
    dx = boxsize / N
    vol = dx**2
    xlin = np.linspace(0.5 * dx, boxsize - 0.5 * dx, N)
    X, Y = np.meshgrid(xlin, xlin, indexing="ij")
    # xlin_node = np.linspace(dx, boxsize, N)
    # Xn, Yn = np.meshgrid( xlin_node, xlin_node, indexing="ij" )

    # Generate Initial Conditions:
    gamma = 5.0 / 3.0  # ideal gas gamma
    if prob_id == 1:
        # Orszag-Tang vortex problem
        t_end = 0.5
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
        P = 0.2 * 0.1 * np.ones(X.shape)  # init. gas pressure
        amp = 0.1
        t_end = 5.0

        # angled ICs
        # alpha = np.pi / 4.0
        # Xpar = (np.cos(alpha) * X + np.sin(alpha) * Y) * np.sqrt(2.0)
        # v_perp = 0.1 * np.sin(2.0 * np.pi * Xpar)
        # v_par = np.zeros(X.shape)
        # b_perp = 0.1 * np.sin(2.0*np.pi*Xpar)
        # vx = v_par * np.cos(alpha) - v_perp * np.sin(alpha)
        # vy = v_par * np.sin(alpha) + v_perp * np.cos(alpha)
        # Az = amp / (2.0 * np.pi) * np.cos(2.0 * np.pi * Xpar)
        # vz = amp * np.cos(2.0 * np.pi * Xpar)
        # Bz = amp * np.cos(2.0 * np.pi * Xpar)

        # simplified ICs
        vx = np.zeros(X.shape)
        vy = amp * np.sin(2.0 * np.pi * X)
        vz = amp * np.cos(2.0 * np.pi * X)
        Az = amp / (2.0 * np.pi) * np.cos(2.0 * np.pi * X)
        Bz = amp * np.cos(2.0 * np.pi * X)

    elif prob_id == 3:
        # Magnetic Field Loop Test
        t_end = 1.0
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
    else:
        print("Problem ID not recognized")
        return

    bx, by = get_curl(Az, dx)
    if prob_id == 2:
        bx = np.ones(X.shape)
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
            Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol
        )

        # get time step (CFL) = dx / max signal speed
        # c0 = np.sqrt( gamma*(P-0.5*(Bx**2+By**2+Bz**2))/rho)
        c0 = np.sqrt(
            gamma * (np.maximum(P - 0.5 * (Bx**2 + By**2 + Bz**2), 1.0e-16)) / rho
        )
        ca = np.sqrt((Bx**2 + By**2 + Bz**2) / rho)
        mask = mask_function(rho, rho_limit)
        ca *= np.sqrt(mask)
        cf = np.sqrt(c0**2 + ca**2)
        c0_max = np.max(c0)
        cf_max = np.max(cf)
        # Adjust timestep
        dt = courant_fac * np.min(dx / (cf + np.sqrt(vx**2 + vy**2 + vz**2)))
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
        ) = get_dudt(rho, vx, vy, vz, P, bx, by, Bz, dx, gamma, rho_limit)

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
            Mass_1, Momx_1, Momy_1, Momz_1, Energy_1, Bx, By, Bz_1, gamma, vol
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
        ) = get_dudt(rho, vx, vy, vz, P, bx_1, by_1, Bz, dx, gamma, rho_limit)

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
            " c0_max =",
            f"{c0_max:.4f}",
            " cf_max =",
            f"{cf_max:.4f}",
            " v_max=",
            f"{v_max:.4f}",
            " rho_min=",
            f"{np.min(Mass) / vol:.4f}",
            " |divB|=",
            f"{np.mean(np.abs(divB)):.4f}",
            " |Bz|=",
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
                # plt.imshow(Bz.T, cmap="jet")
                # plt.clim(-0.1, 0.1)
                plt.plot(Bz[:, N // 2])
                plt.ylim(-2.0 * amp, 2.0 * amp)
            elif prob_id == 3:
                plt.imshow(np.sqrt(bx**2 + by**2).T, cmap="jet")
                plt.clim(0.0, 0.0011)
            ax = plt.gca()
            if prob_id != 2:
                ax.invert_yaxis()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_aspect("equal")
            plt.pause(0.001)
            out_count += 1

    print("done!")

    prefix = "output/mask_p" + str(prob_id) + "_"

    # Save figure
    plt.savefig(prefix + "P_B_" + str(rho_limit) + ".png", dpi=240)
    # plt.show()

    # Save rho, P_B, v, vA, cf, and dt_sav
    Bx, By = get_avg(bx, by)
    rho, vx, vy, vz, P = get_primitive(
        Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol
    )
    P_B = 0.5 * np.sqrt(Bx**2 + By**2 + Bz**2)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    # c0 = np.sqrt( gamma*(P-0.5*(Bx**2+By**2+Bz**2))/rho )
    c0 = np.sqrt(gamma * (np.maximum(P - 0.5 * (Bx**2 + By**2 + Bz**2), 1.0e-16)) / rho)
    ca = np.sqrt((Bx**2 + By**2 + Bz**2) / rho)
    cf = np.sqrt(c0**2 + ca**2)
    np.save(prefix + "data_rho_" + str(rho_limit) + ".npy", rho.T)
    np.save(prefix + "data_P_B_" + str(rho_limit) + ".npy", P_B.T)
    np.save(prefix + "data_Bz_" + str(rho_limit) + ".npy", Bz.T)
    np.save(prefix + "data_v_" + str(rho_limit) + ".npy", v.T)
    np.save(prefix + "data_ca_" + str(rho_limit) + ".npy", ca.T)
    np.save(prefix + "data_cf_" + str(rho_limit) + ".npy", cf.T)
    np.save(prefix + "data_dt_" + str(rho_limit) + ".npy", dt_sav)

    return


if __name__ == "__main__":
    main()
