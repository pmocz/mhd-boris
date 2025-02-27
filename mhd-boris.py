import matplotlib.pyplot as plt
import numpy as np
import sys

"""
2.5D Constrained Transport Magnetohydrodynamics
Philip Mocz (2024), @PMocz

Simulate the Orszag-Tang vortex MHD problem
with a Boris-like Integrator to control timesteps!

The original problem has cf_max ~ 1.9, u_max ~ 1.6
"""

# directions for np.roll()
R = -1  # right/up
L = 1  # left/down


def get_curl(Az, dx):
    """
    Calculate the discrete curl
    Az       is matrix of nodal z-component of magnetic potential
    dx       is the cell size
    bx       is matrix of cell face x-component magnetic-field
    by       is matrix of cell face y-component magnetic-field
    """

    bx = (Az - np.roll(Az, L, axis=1)) / dx  # = d Az / d y
    by = -(Az - np.roll(Az, L, axis=0)) / dx  # =-d Az / d x

    return bx, by


def get_div(bx, by, dx):
    """
    Calculate the discrete curl of each cell
    dx       is the cell size
    bx       is matrix of cell face x-component magnetic-field
    by       is matrix of cell face y-component magnetic-field
    """

    divB = (bx - np.roll(bx, L, axis=0) + by - np.roll(by, L, axis=1)) / dx

    return divB


def get_B_avg(bx, by):
    """
    Calculate the volume-averaged magnetic field
    bx       is matrix of cell face x-component magnetic-field
    by       is matrix of cell face y-component magnetic-field
    Bx       is matrix of cell Bx
    By       is matrix of cell By
    """

    Bx = 0.5 * (bx + np.roll(bx, L, axis=0))
    By = 0.5 * (by + np.roll(by, L, axis=1))

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
    ) * (gamma - 1) + 0.5 * (Bx**2 + By**2 + Bz**2)

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

    f_dx = (np.roll(f, R, axis=0) - np.roll(f, L, axis=0)) / (2.0 * dx)
    f_dy = (np.roll(f, R, axis=1) - np.roll(f, L, axis=1)) / (2.0 * dx)

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
                1.0, ((f - np.roll(f, L, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0))
            ),
        )
        * f_dx
    )
    f_dx = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, (-(f - np.roll(f, R, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0))
            ),
        )
        * f_dx
    )
    f_dy = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, ((f - np.roll(f, L, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0))
            ),
        )
        * f_dy
    )
    f_dy = (
        np.maximum(
            0.0,
            np.minimum(
                1.0, (-(f - np.roll(f, R, axis=1)) / dx) / (f_dy + 1.0e-8 * (f_dy == 0))
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
    f_XL = np.roll(f_XL, R, axis=0)
    f_XR = f + f_dx * dx / 2.0

    f_YL = f - f_dy * dx / 2.0
    f_YL = np.roll(f_YL, R, axis=1)
    f_YR = f + f_dy * dx / 2.0

    return f_XL, f_XR, f_YL, f_YR


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
    F += dt * dx * np.roll(flux_F_X, L, axis=0)
    F += -dt * dx * flux_F_Y
    F += dt * dx * np.roll(flux_F_Y, L, axis=1)

    return F


def constrained_transport(bx, by, flux_By_X, flux_Bx_Y, dx, dt):
    """
    Apply fluxes to face-centered magnetic fields in a constrained transport manner
    bx        is matrix of cell face x-component magnetic-field
    by        is matrix of cell face y-component magnetic-field
    flux_By_X is a matrix of the x-dir fluxes of By
    flux_Bx_Y is a matrix of the y-dir fluxes of Bx
    dx        is the cell size
    dt        is the timestep
    """

    # update solution
    # Ez at top right node of cell = avg of 4 fluxes
    Ez = 0.25 * (
        -flux_By_X
        - np.roll(flux_By_X, R, axis=1)
        + flux_Bx_Y
        + np.roll(flux_Bx_Y, R, axis=0)
    )
    dbx, dby = get_curl(-Ez, dx)

    bx += dt * dbx
    by += dt * dby

    return bx, by


def getFlux(
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
        (P_L - 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2)) / (gamma - 1)
        + 0.5 * rho_L * (vx_L**2 + vy_L**2 + vz_L**2)
        + 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2)
    )
    en_R = (
        (P_R - 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2)) / (gamma - 1)
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

    P_star = (gamma - 1) * (
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
    flux_Mass -= C * 0.5 * (rho_L - rho_R)
    flux_Momx -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Momz -= C * 0.5 * (rho_L * vz_L - rho_R * vz_R)
    flux_Energy -= C * 0.5 * (en_L - en_R)
    flux_By -= C * 0.5 * (By_L - By_R)
    flux_Bz -= C * 0.5 * (Bz_L - Bz_R)

    return flux_Mass, flux_Momx, flux_Momy, flux_Momz, flux_Energy, flux_By, flux_Bz


def main():
    """Finite Volume simulation"""

    # Check for command line arguments
    if len(sys.argv) != 3:
        print("Usage: python mhd-boris.py <problem_id> <cf_limit>")
        return

    # Parse command line argument
    # for the boris integrator (try 1.0, 1.5, 2.0)
    prob_id = int(sys.argv[1])
    cf_limit = float(sys.argv[2])

    # Simulation parameters
    N = 128  # resolution
    boxsize = 1.0
    gamma = 5.0 / 3.0  # ideal gas gamma
    courant_fac = 0.4
    t = 0.0
    tEnd = 0.5
    tOut = 0.01  # draw frequency
    useSlopeLimiting = True
    plotRealTime = True  # switch on for plotting as the simulation goes along

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

        # bx, by = get_curl(Az, dx)
        # plt.imshow(vx.T, cmap='jet')
        # plt.show()
        # XXX

        # simplified ICs XXX
        vx = np.zeros(X.shape)
        vy = 0.1 * np.sin(2.0 * np.pi * X)
        vz = 0.1 * np.cos(2.0 * np.pi * X)
        Ax = np.zeros(X.shape)
        Ay = 0.1 / (2.0 * np.pi) * np.sin(2.0 * np.pi * X)
        Az = 0.1 / (2.0 * np.pi) * np.cos(2.0 * np.pi * X)
        Bz = 0.1 * np.cos(2.0 * np.pi * X)

        courant_fac = courant_fac / 10.0
        tEnd = 4.0

    elif prob_id == 3:
        # Magnetic Field Loop Test
        rho = np.ones(X.shape)
        P = np.ones(X.shape)
        vx = np.ones(X.shape) * np.sin(np.pi / 3.0)
        vy = np.ones(X.shape) * np.cos(np.pi / 3.0)
        vz = np.zeros(X.shape)
        Bz = np.ones(X.shape)
        anorm = 1.0e-3
        r0 = 0.3
        r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
        Az = np.maximum(anorm * (r0 - r), 0)
        # courant_fac = courant_fac / 10.0
    else:
        print("Problem ID not recognized")
        return

    bx, by = get_curl(Az, dx)
    Bx, By = get_B_avg(bx, by)

    # add magnetic pressure to get the total pressure
    P = P + 0.5 * (Bx**2 + By**2 + Bz**2)

    # Get conserved variables
    Mass, Momx, Momy, Momz, Energy = get_conserved(
        rho, vx, vy, vz, P, Bx, By, Bz, gamma, vol
    )

    # keep track of timesteps
    dt_sav = []

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    outputCount = 1

    # Simulation Main Loop
    while t < tEnd:
        # get Primitive variables
        Bx, By = get_B_avg(bx, by)
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
        # cf *= alpha
        dt = courant_fac * np.min(dx / (cf + np.sqrt(vx**2 + vy**2 + vz**2)))
        u_max = np.max(np.sqrt(vx**2 + vy**2 + vz**2))

        plotThisTurn = False
        if t + dt > outputCount * tOut:
            # dt = outputCount*tOut - t
            plotThisTurn = True

        # calculate gradients
        rho_dx, rho_dy = get_gradient(rho, dx)
        vx_dx, vx_dy = get_gradient(vx, dx)
        vy_dx, vy_dy = get_gradient(vy, dx)
        vz_dx, vz_dy = get_gradient(vz, dx)
        P_dx, P_dy = get_gradient(P, dx)
        Bx_dx, Bx_dy = get_gradient(Bx, dx)
        By_dx, By_dy = get_gradient(By, dx)
        Bz_dx, Bz_dy = get_gradient(Bz, dx)

        # slope limit gradients
        if useSlopeLimiting:
            rho_dx, rho_dy = slope_limit(rho, dx, rho_dx, rho_dy)
            vx_dx, vx_dy = slope_limit(vx, dx, vx_dx, vx_dy)
            vy_dx, vy_dy = slope_limit(vy, dx, vy_dx, vy_dy)
            vz_dx, vz_dy = slope_limit(vz, dx, vz_dx, vz_dy)
            P_dx, P_dy = slope_limit(P, dx, P_dx, P_dy)
            Bx_dx, Bx_dy = slope_limit(Bx, dx, Bx_dx, Bx_dy)
            By_dx, By_dy = slope_limit(By, dx, By_dx, By_dy)
            Bz_dx, Bz_dy = slope_limit(Bz, dx, Bz_dx, Bz_dy)

        # extrapolate rho half-step in time
        rho_prime = rho - 0.5 * dt * (
            vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy
        )
        # extrapolate velocity half-step in time
        vx_prime = vx - 0.5 * dt * (
            vx * vx_dx
            + vy * vx_dy
            + (1.0 / rho) * P_dx
            - (Bx / rho) * (2.0 * Bx_dx + By_dy)
            - (By / rho) * Bx_dy
        )
        vy_prime = vy - 0.5 * dt * (
            vx * vy_dx
            + vy * vy_dy
            + (1.0 / rho) * P_dy
            - (Bx / rho) * By_dx
            - (By / rho) * (Bx_dx + 2.0 * By_dy)
        )
        vz_prime = vz - 0.5 * dt * (
            vx * vz_dx + vy * vz_dy - (Bx / rho) * Bz_dx - (By / rho) * Bz_dy
        )
        # extrapolate pressure half-step in time
        P_prime = P - 0.5 * dt * (
            (gamma * (P - 0.5 * (Bx**2 + By**2 + Bz**2)) + By**2 + Bz**2) * vx_dx
            - Bx * By * vy_dx
            - Bx * Bz * vz_dx
            + vx * P_dx
            + (gamma - 2.0) * (Bx * vx + By * vy + Bz * vz) * Bx_dx
            - By * Bx * vx_dy
            + (gamma * (P - 0.5 * (Bx**2 + By**2 + Bz**2)) + Bx**2 + Bz**2) * vy_dy
            - By * Bz * vz_dy
            + vy * P_dy
            + (gamma - 2.0) * (Bx * vx + By * vy + Bz * vz) * By_dy
        )

        # extrapolate magnetic field half-step in time
        Bx_prime = Bx - 0.5 * dt * (Bx * vy_dy - By * vx_dy - vx * By_dy + vy * Bx_dy)
        By_prime = By - 0.5 * dt * (-Bx * vy_dx + By * vx_dx + vx * By_dx - vy * Bx_dx)
        Bz_prime = Bz - 0.5 * dt * (
            -Bx * vz_dx
            - By * vz_dy
            + Bz * vx_dx
            + Bz * vy_dy
            + vx * Bz_dx
            + vy * Bz_dy
            - vz * Bx_dx
            - vz * By_dy
        )

        # extrapolate in space to face centers
        rho_XL, rho_XR, rho_YL, rho_YR = extrapolate_in_space_to_face(
            rho_prime, rho_dx, rho_dy, dx
        )
        vx_XL, vx_XR, vx_YL, vx_YR = extrapolate_in_space_to_face(
            vx_prime, vx_dx, vx_dy, dx
        )
        vy_XL, vy_XR, vy_YL, vy_YR = extrapolate_in_space_to_face(
            vy_prime, vy_dx, vy_dy, dx
        )
        vz_XL, vz_XR, vz_YL, vz_YR = extrapolate_in_space_to_face(
            vz_prime, vz_dx, vz_dy, dx
        )
        P_XL, P_XR, P_YL, P_YR = extrapolate_in_space_to_face(P_prime, P_dx, P_dy, dx)
        Bx_XL, Bx_XR, Bx_YL, Bx_YR = extrapolate_in_space_to_face(
            Bx_prime, Bx_dx, Bx_dy, dx
        )
        By_XL, By_XR, By_YL, By_YR = extrapolate_in_space_to_face(
            By_prime, By_dx, By_dy, dx
        )
        Bz_XL, Bz_XR, Bz_YL, Bz_YR = extrapolate_in_space_to_face(
            Bz_prime, Bz_dx, Bz_dy, dx
        )

        # compute fluxes (local Lax-Friedrichs/Rusanov)
        (
            flux_Mass_X,
            flux_Momx_X,
            flux_Momy_X,
            flux_Momz_X,
            flux_Energy_X,
            flux_By_X,
            flux_Bz_X,
        ) = getFlux(
            rho_XL,
            rho_XR,
            vx_XL,
            vx_XR,
            vy_XL,
            vy_XR,
            vz_XR,
            vz_XL,
            P_XL,
            P_XR,
            Bx_XL,
            Bx_XR,
            By_XL,
            By_XR,
            Bz_XL,
            Bz_XR,
            gamma,
            cf_limit,
        )
        (
            flux_Mass_Y,
            flux_Momy_Y,
            flux_Momx_Y,
            flux_Momz_Y,
            flux_Energy_Y,
            flux_Bx_Y,
            flux_Bz_Y,
        ) = getFlux(
            rho_YL,
            rho_YR,
            vy_YL,
            vy_YR,
            vx_YL,
            vx_YR,
            vz_YL,
            vz_YR,
            P_YL,
            P_YR,
            By_YL,
            By_YR,
            Bx_YL,
            Bx_YR,
            Bz_YL,
            Bz_YR,
            gamma,
            cf_limit,
        )

        # update solution
        Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
        Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
        Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
        Momz = apply_fluxes(Momz, flux_Momz_X, flux_Momz_Y, dx, dt)
        Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)
        Bz = apply_fluxes(Bz, flux_Bz_X, flux_Bz_Y, dx / vol, dt)
        bx, by = constrained_transport(bx, by, flux_By_X, flux_Bx_Y, dx, dt)

        # update time
        t += dt

        # save timestep
        dt_sav.append(dt)

        # check div B
        divB = get_div(bx, by, dx)
        print(
            "t =",
            f"{t:.4f}",
            "max_cf=",
            f"{max_cf:.4f}",
            "u_max=",
            f"{u_max:.4f}",
            "alpha=",
            f"{np.min(alpha):.4f}",
            ", mean |divB| = ",
            f"{np.mean(np.abs(divB)):.4f}",
        )

        # plot in real time
        if (plotRealTime and plotThisTurn) or (t >= tEnd):
            plt.cla()
            # plt.imshow(rho.T, cmap='jet')
            # plt.clim(0.06, 0.5)
            if prob_id == 1:
                plt.imshow(np.sqrt(bx**2 + by**2).T, cmap="jet")
                plt.clim(0.0, 0.8)
            elif prob_id == 2:
                plt.imshow(Bz.T, cmap="jet")
                # XXXplt.clim(0.0, 0.8)
            elif prob_id == 3:
                plt.imshow(np.sqrt(bx**2 + by**2).T, cmap="jet")
                # XXXplt.clim(0.0, 0.8)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            plt.pause(0.001)
            outputCount += 1

    print("done!")

    prefix = "output/p" + str(prob_id) + "_"

    # Save figure
    plt.savefig(prefix + "P_B_" + str(cf_limit) + ".png", dpi=240)
    # plt.show()

    # Save rho, P_B, v, vA, cf, and dt_sav
    Bx, By = get_B_avg(bx, by)
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
