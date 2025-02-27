import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

"""
A simple 3D MHD code in JAX

Philip Mocz
Flatiron Institute

2025
"""


@jax.jit
def get_curl(Ax, Ay, Az, dx):
    """
    Calculate the discrete curl from edges to faces
    """

    bx = (Az - jnp.roll(Az, 1, axis=1)) / dx - (Ay - jnp.roll(Ay, 1, axis=2)) / dx
    by = (Ax - jnp.roll(Ax, 1, axis=2)) / dx - (Az - jnp.roll(Az, 1, axis=0)) / dx
    bz = (Ay - jnp.roll(Ay, 1, axis=0)) / dx - (Ax - jnp.roll(Ax, 1, axis=1)) / dx

    return bx, by, bz


@jax.jit
def get_div(bx, by, bz, dx):
    """
    Calculate the discrete divergence from faces to cell-centers
    """

    divB = (
        bx
        - jnp.roll(bx, 1, axis=0)
        + by
        - jnp.roll(by, 1, axis=1)
        + bz
        - jnp.roll(bz, 1, axis=2)
    ) / dx

    return divB


@jax.jit
def get_avg(bx, by, bz):
    """
    Calculate the field average from faces to cell-centers
    """

    Bx = 0.5 * (bx + jnp.roll(bx, 1, axis=0))
    By = 0.5 * (by + jnp.roll(by, 1, axis=1))
    Bz = 0.5 * (bz + jnp.roll(bz, 1, axis=2))

    return Bx, By, Bz


@jax.jit
def get_conserved(rho, vx, vy, vz, P, bx, by, bz, gamma, vol):
    """
    Calculate the conserved variables from the primitive variables
    """

    Bx, By, Bz = get_avg(bx, by, bz)

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


@jax.jit
def get_primitive(Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol):
    """
    Calculate the primitive variables from the conservative variables
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


@jax.jit
def get_gradient(f, dx):
    """
    Calculate the gradients of a field with 2nd-order central difference
    """

    f_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    f_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dx)
    f_dz = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2.0 * dx)

    return f_dx, f_dy, f_dz


@jax.jit
def slope_limiter(f, dx, f_dx, f_dy, f_dz):
    """
    Apply slope limiter to slopes (minmod)
    """

    eps = 1.0e-12

    # Keep a copy of the original slopes
    orig_f_dx = f_dx
    orig_f_dy = f_dy
    orig_f_dz = f_dz

    # Function to adjust the denominator safely
    def adjust_denominator(denom):
        denom_safe = jnp.where(
            denom > 0, denom + eps, jnp.where(denom < 0, denom - eps, eps)
        )
        return denom_safe

    # For x-direction
    denom = adjust_denominator(orig_f_dx)
    num = (f - jnp.roll(f, 1, axis=0)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dx = limiter * f_dx

    num = -(f - jnp.roll(f, -1, axis=0)) / dx
    ratio = num / denom  # Use the same adjusted denominator
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dx = limiter * f_dx

    # For y-direction
    denom = adjust_denominator(orig_f_dy)
    num = (f - jnp.roll(f, 1, axis=1)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dy = limiter * f_dy

    num = -(f - jnp.roll(f, -1, axis=1)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dy = limiter * f_dy

    # For z-direction
    denom = adjust_denominator(orig_f_dz)
    num = (f - jnp.roll(f, 1, axis=2)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dz = limiter * f_dz

    num = -(f - jnp.roll(f, -1, axis=2)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dz = limiter * f_dz

    return f_dx, f_dy, f_dz


@jax.jit
def extrapolate_to_face(f, f_dx, f_dy, f_dz, dx):
    """
    Extrapolate a cell-centered field to face-centered values
    """

    f_XL = f - f_dx * dx / 2.0
    f_XL = jnp.roll(f_XL, -1, axis=0)
    f_XR = f + f_dx * dx / 2.0

    f_YL = f - f_dy * dx / 2.0
    f_YL = jnp.roll(f_YL, -1, axis=1)
    f_YR = f + f_dy * dx / 2.0

    f_ZL = f - f_dz * dx / 2.0
    f_ZL = jnp.roll(f_ZL, -1, axis=2)
    f_ZR = f + f_dz * dx / 2.0

    return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR


@jax.jit
def apply_fluxes(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dt):
    """
    Apply fluxes to conserved variables
    """

    F += -dt * dx**2 * flux_F_X
    F += dt * dx**2 * jnp.roll(flux_F_X, 1, axis=0)
    F += -dt * dx**2 * flux_F_Y
    F += dt * dx**2 * jnp.roll(flux_F_Y, 1, axis=1)
    F += -dt * dx**2 * flux_F_Z
    F += dt * dx**2 * jnp.roll(flux_F_Z, 1, axis=2)

    return F


@jax.jit
def constrained_transport(
    bx, by, bz, flux_By_X, flux_Bx_Y, flux_Bz_Y, flux_By_Z, flux_Bx_Z, flux_Bz_X, dx, dt
):
    """
    Apply fluxes to face-centered magnetic fields in a divergence-preserving manner
    """

    # compute electric fields at nodes
    Ex = 0.25 * (
        flux_Bz_Y
        + jnp.roll(flux_Bz_Y, -1, axis=2)
        - flux_By_Z
        - jnp.roll(flux_By_Z, -1, axis=1)
    )
    Ey = 0.25 * (
        flux_Bx_Z
        + jnp.roll(flux_Bx_Z, -1, axis=0)
        - flux_Bz_X
        - jnp.roll(flux_Bz_X, -1, axis=2)
    )
    Ez = 0.25 * (
        flux_By_X
        + jnp.roll(flux_By_X, -1, axis=1)
        - flux_Bx_Y
        - jnp.roll(flux_Bx_Y, -1, axis=0)
    )

    # compute db components
    dbx, dby, dbz = get_curl(Ex, Ey, Ez, dx)

    # update magnetic fields
    bx += dt * dbx
    by += dt * dby
    bz += dt * dbz

    return bx, by, bz


@jax.jit
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
):
    """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule
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

    # Find wave speeds
    c0_L = jnp.sqrt(gamma * (P_L - 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2)) / rho_L)
    c0_R = jnp.sqrt(gamma * (P_R - 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2)) / rho_R)
    ca_L = jnp.sqrt((Bx_L**2 + By_L**2 + Bz_L**2) / rho_L)
    ca_R = jnp.sqrt((Bx_R**2 + By_R**2 + Bz_R**2) / rho_R)
    cf_L = jnp.sqrt(
        0.5 * (c0_L**2 + ca_L**2) + 0.5 * jnp.sqrt((c0_L**2 + ca_L**2) ** 2)
    )
    cf_R = jnp.sqrt(
        0.5 * (c0_R**2 + ca_R**2) + 0.5 * jnp.sqrt((c0_R**2 + ca_R**2) ** 2)
    )
    C_L = cf_L + jnp.abs(vx_L)
    C_R = cf_R + jnp.abs(vx_R)
    C = jnp.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_R - rho_L)
    flux_Momx -= C * 0.5 * (rho_R * vx_R - rho_L * vx_L)
    flux_Momy -= C * 0.5 * (rho_R * vy_R - rho_L * vy_L)
    flux_Momz -= C * 0.5 * (rho_R * vz_R - rho_L * vz_L)
    flux_Energy -= C * 0.5 * (en_R - en_L)
    flux_By -= C * 0.5 * (By_R - By_L)
    flux_Bz -= C * 0.5 * (Bz_R - Bz_L)

    # (note: flux_Bx = 0)

    return flux_Mass, flux_Momx, flux_Momy, flux_Momz, flux_Energy, flux_By, flux_Bz


@jax.jit
def update_sim(values):
    """
    Update the solution state by one timestep according to the ideal MHD equations
    """

    rho, bx, by, bz, Mass, Momx, Momy, Momz, Energy, dx, dt, gamma = values

    vol = dx**3

    # get cell-centered magnetic field values
    Bx, By, Bz = get_avg(bx, by, bz)

    # get primitive variables
    rho, vx, vy, vz, P = get_primitive(
        Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, gamma, vol
    )

    # get time step (CFL) = dx / max signal speed
    c0 = jnp.sqrt(gamma * (P - 0.5 * (Bx**2 + By**2 + Bz**2)) / rho)
    ca = jnp.sqrt((Bx**2 + By**2 + Bz**2) / rho)
    cf = jnp.sqrt(0.5 * (c0**2 + ca**2) + 0.5 * jnp.sqrt((c0**2 + ca**2) ** 2))
    courant_fac = 0.4
    dt = courant_fac * jnp.min(dx / (cf + jnp.sqrt(vx**2 + vy**2 + vz**2)))

    # calculate gradients
    rho_dx, rho_dy, rho_dz = get_gradient(rho, dx)
    vx_dx, vx_dy, vx_dz = get_gradient(vx, dx)
    vy_dx, vy_dy, vy_dz = get_gradient(vy, dx)
    vz_dx, vz_dy, vz_dz = get_gradient(vz, dx)
    P_dx, P_dy, P_dz = get_gradient(P, dx)
    Bx_dx, Bx_dy, Bx_dz = get_gradient(Bx, dx)
    By_dx, By_dy, By_dz = get_gradient(By, dx)
    Bz_dx, Bz_dy, Bz_dz = get_gradient(Bz, dx)

    # slope limit gradients
    rho_dx, rho_dy, rho_dz = slope_limiter(rho, dx, rho_dx, rho_dy, rho_dz)
    vx_dx, vx_dy, vx_dz = slope_limiter(vx, dx, vx_dx, vx_dy, vx_dz)
    vy_dx, vy_dy, vy_dz = slope_limiter(vy, dx, vy_dx, vy_dy, vy_dz)
    vz_dx, vz_dy, vz_dz = slope_limiter(vz, dx, vz_dx, vz_dy, vz_dz)
    P_dx, P_dy, P_dz = slope_limiter(P, dx, P_dx, P_dy, P_dz)
    Bx_dx, Bx_dy, Bx_dz = slope_limiter(Bx, dx, Bx_dx, Bx_dy, Bx_dz)
    By_dx, By_dy, By_dz = slope_limiter(By, dx, By_dx, By_dy, By_dz)
    Bz_dx, Bz_dy, Bz_dz = slope_limiter(Bz, dx, Bz_dx, Bz_dy, Bz_dz)

    # extrapolate rho half-step in time
    rho_prime = rho - 0.5 * dt * (
        vx * rho_dx
        + rho * vx_dx
        + vy * rho_dy
        + rho * vy_dy
        + vz * rho_dz
        + rho * vz_dz
    )

    # extrapolate velocity half-step in time
    vx_prime = vx - 0.5 * dt * (
        vx * vx_dx
        + vy * vx_dy
        + vz * vx_dz
        + (1.0 / rho) * P_dx
        - (Bx / rho) * (2.0 * Bx_dx + By_dy + Bz_dz)
        - (By / rho) * Bx_dy
        - (Bz / rho) * Bx_dz
    )
    vy_prime = vy - 0.5 * dt * (
        vx * vy_dx
        + vy * vy_dy
        + vz * vy_dz
        + (1.0 / rho) * P_dy
        - (Bx / rho) * By_dx
        - (By / rho) * (Bx_dx + 2.0 * By_dy + Bz_dz)
        - (Bz / rho) * By_dz
    )
    vz_prime = vz - 0.5 * dt * (
        vx * vz_dx
        + vy * vz_dy
        + vz * vz_dz
        + (1.0 / rho) * P_dz
        - (Bx / rho) * Bz_dx
        - (By / rho) * Bz_dy
        - (Bz / rho) * (Bx_dx + By_dy + 2.0 * Bz_dz)
    )

    # extrapolate pressure half-step in time
    vx_dx_term = (gamma * (P - 0.5 * (Bx**2 + By**2 + Bz**2)) + By**2 + Bz**2) * vx_dx
    vy_dx_term = -Bx * By * vy_dx
    vz_dx_term = -Bx * Bz * vz_dx
    P_dx_term = vx * P_dx
    Bx_dx_term = (gamma - 2.0) * (Bx * vx + By * vy + Bz * vz) * Bx_dx
    vx_dy_term = -By * Bx * vx_dy
    vy_dy_term = (gamma * (P - 0.5 * (Bx**2 + By**2 + Bz**2)) + Bx**2 + Bz**2) * vy_dy
    vz_dy_term = -By * Bz * vz_dy
    P_dy_term = vy * P_dy
    Bx_dy_term = (gamma - 2.0) * (Bx * vx + By * vy + Bz * vz) * By_dy
    vx_dz_term = -Bz * Bx * vx_dz
    vy_dz_term = -Bz * By * vy_dz
    vz_dz_term = (gamma * (P - 0.5 * (Bx**2 + By**2 + Bz**2)) + Bx**2 + By**2) * vz_dz
    P_dz_term = vz * P_dz
    Bx_dz_term = (gamma - 2.0) * (Bx * vx + By * vy + Bz * vz) * Bz_dz
    P_prime = P - 0.5 * dt * (
        vx_dx_term
        + vy_dx_term
        + vz_dx_term
        + P_dx_term
        + Bx_dx_term
        + vx_dy_term
        + vy_dy_term
        + vz_dy_term
        + P_dy_term
        + Bx_dy_term
        + vx_dz_term
        + vy_dz_term
        + vz_dz_term
        + P_dz_term
        + Bx_dz_term
    )

    # extrapolate magnetic field half-step in time
    Bx_prime = Bx - 0.5 * dt * (
        Bx * vy_dy
        + Bx * vz_dz
        - By * vx_dy
        - Bz * vx_dz
        - vx * By_dy
        - vx * Bz_dz
        + vy * Bx_dy
        + vz * Bx_dz
    )
    By_prime = By - 0.5 * dt * (
        -Bx * vy_dx
        + By * vx_dx
        + By * vz_dz
        - Bz * vy_dz
        + vx * By_dx
        - vy * Bx_dx
        - vy * Bz_dz
        + vz * By_dz
    )
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
    rho_XL, rho_XR, rho_YL, rho_YR, rho_ZL, rho_ZR = extrapolate_to_face(
        rho_prime, rho_dx, rho_dy, rho_dz, dx
    )
    vx_XL, vx_XR, vx_YL, vx_YR, vx_ZL, vx_ZR = extrapolate_to_face(
        vx_prime, vx_dx, vx_dy, vx_dz, dx
    )
    vy_XL, vy_XR, vy_YL, vy_YR, vy_ZL, vy_ZR = extrapolate_to_face(
        vy_prime, vy_dx, vy_dy, vy_dz, dx
    )
    vz_XL, vz_XR, vz_YL, vz_YR, vz_ZL, vz_ZR = extrapolate_to_face(
        vz_prime, vz_dx, vz_dy, vz_dz, dx
    )
    P_XL, P_XR, P_YL, P_YR, P_ZL, P_ZR = extrapolate_to_face(
        P_prime, P_dx, P_dy, P_dz, dx
    )
    Bx_XL, Bx_XR, Bx_YL, Bx_YR, Bx_ZL, Bx_ZR = extrapolate_to_face(
        Bx_prime, Bx_dx, Bx_dy, Bx_dz, dx
    )
    By_XL, By_XR, By_YL, By_YR, By_ZL, By_ZR = extrapolate_to_face(
        By_prime, By_dx, By_dy, By_dz, dx
    )
    Bz_XL, Bz_XR, Bz_YL, Bz_YR, Bz_ZL, Bz_ZR = extrapolate_to_face(
        Bz_prime, Bz_dx, Bz_dy, Bz_dz, dx
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
    )
    (
        flux_Mass_Z,
        flux_Momz_Z,
        flux_Momx_Z,
        flux_Momy_Z,
        flux_Energy_Z,
        flux_Bx_Z,
        flux_By_Z,
    ) = get_flux(
        rho_ZR,
        rho_ZL,
        vz_ZR,
        vz_ZL,
        vx_ZR,
        vx_ZL,
        vy_ZR,
        vy_ZL,
        P_ZR,
        P_ZL,
        Bz_ZR,
        Bz_ZL,
        Bx_ZR,
        Bx_ZL,
        By_ZR,
        By_ZL,
        gamma,
    )

    # update solution
    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, flux_Mass_Z, dx, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, flux_Momx_Z, dx, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, flux_Momy_Z, dx, dt)
    Momz = apply_fluxes(Momz, flux_Momz_X, flux_Momz_Y, flux_Momz_Z, dx, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y, flux_Energy_Z, dx, dt)
    bx, by, bz = constrained_transport(
        bx,
        by,
        bz,
        flux_By_X,
        flux_Bx_Y,
        flux_Bz_Y,
        flux_By_Z,
        flux_Bx_Z,
        flux_Bz_X,
        dx,
        dt,
    )

    # check div B
    # divB = get_div(bx, by, bz, dx)

    return rho, bx, by, bz, Mass, Momx, Momy, Momz, Energy, dx, dt, gamma


@jax.jit
def compute_divergence(P_dx, P_dy, P_dz, dx):
    P_dx_dx, _, _ = get_gradient(P_dx, dx)
    _, P_dy_dy, _ = get_gradient(P_dy, dx)
    _, _, P_dz_dz = get_gradient(P_dz, dx)
    div_P_grad = P_dx_dx + P_dy_dy + P_dz_dz
    return div_P_grad


def main():
    """MHD Simulation"""

    # jax.config.update("jax_enable_x64", True)
    # print(jax.default_backend())
    # print(jax.devices())

    # Simulation parameters
    N = 32  # 64  # cells per unit length
    gamma = 5.0 / 3.0  # ideal gas gamma
    t_end = 4.0
    dt = 0

    # Geometry parameters
    dx = 1.0 / N

    # Mesh
    x_lin = jnp.linspace(0.5 * dx, 1.0 - 0.5 * dx, N)
    y_lin = jnp.linspace(0.5 * dx, 1.0 - 0.5 * dx, N)
    z_lin = jnp.linspace(0.5 * dx, 1.0 - 0.5 * dx, N)
    X, Y, Z = jnp.meshgrid(x_lin, y_lin, z_lin, indexing="ij")

    # Generate Initial Conditions
    rho = jnp.ones(X.shape)
    P = 0.1 * jnp.ones(X.shape)  # init. gas pressure

    alpha = jnp.pi / 4.0
    Xpar = (jnp.cos(alpha) * X + jnp.sin(alpha) * Y) * jnp.sqrt(2)
    v_perp = 0.1 * jnp.sin(2.0 * jnp.pi * Xpar)
    v_par = jnp.zeros(X.shape)
    # b_perp = 0.1 * jnp.sin(2.0*np.pi*Xpar)
    vx = v_par * jnp.cos(alpha) - v_perp * jnp.sin(alpha)
    vy = v_par * jnp.sin(alpha) + v_perp * jnp.cos(alpha)
    Az = 0.1 / (2.0 * jnp.pi) * jnp.cos(2.0 * jnp.pi * Xpar)
    vz = 0.1 * jnp.cos(2.0 * jnp.pi * Xpar)
    # Bz = 0.1 * jnp.cos(2.0*jnp.pi*Xpar)
    # XXX
    Ax = 0.1 / (2.0 * jnp.pi) * jnp.sin(2.0 * jnp.pi * Xpar)
    Ay = 0.1 / (2.0 * jnp.pi) * jnp.sin(2.0 * jnp.pi * Xpar)

    # simplified ICs XXX
    vx = jnp.zeros(X.shape)
    vy = 0.1 * jnp.sin(2.0 * jnp.pi * X)
    vz = 0.1 * jnp.cos(2.0 * jnp.pi * X)
    Ax = jnp.zeros(X.shape)
    Ay = 0.1 / (2.0 * jnp.pi) * jnp.sin(2.0 * jnp.pi * X)
    Az = 0.1 / (2.0 * jnp.pi) * jnp.cos(2.0 * jnp.pi * X)

    bx, by, bz = get_curl(Ax, Ay, Az, dx)

    Bx, By, Bz = get_avg(bx, by, bz)
    # add magnetic pressure to get the total pressure
    P = P + 0.5 * (Bx**2 + By**2 + Bz**2)

    # Do the simulation
    Mass, Momx, Momy, Momz, Energy = get_conserved(
        rho, vx, vy, vz, P, bx, by, bz, gamma, dx**3
    )
    t = 0
    while t < t_end:
        values = (
            rho,
            bx,
            by,
            bz,
            Mass,
            Momx,
            Momy,
            Momz,
            Energy,
            dx,
            dt,
            gamma,
        )
        rho, bx, by, bz, Mass, Momx, Momy, Momz, Energy, dx, dt, gamma = update_sim(
            values
        )

        # update time
        t += dt
        print("dt=", dt)

        # plot bz
        bz_plot = jax.lax.slice(bz, (0, 0, N // 2), (N, N, N // 2 + 1)).reshape((N, N))
        plt.imshow(bz_plot.T, cmap="jet")
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect("equal")
        plt.pause(0.001)


if __name__ == "__main__":
    main()
