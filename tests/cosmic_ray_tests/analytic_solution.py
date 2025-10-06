import numpy as np
from scipy.integrate import quad
from scipy.optimize import root, root_scalar

def get_cosmic_ray_analytic_solution(
        thermal_pressure_left: float = 17.172,
        thermal_pressure_right: float = 0.05,

        density_left: float = 1.0,
        density_right: float = 1.0 / 8.0,

        thermal_to_cosmic_ray_pressure_ratio_left: float = 2.0,
        thermal_to_cosmic_ray_pressure_ratio_right: float = 1.0,

        injection_efficiency: float = 0.5,

        t_end: float = 0.35,

        left_boundary: float = 0.0,
        right_boundary: float = 10.0,

        initial_shock_pos: float = 5.0,

        num_rarefaction_evals: int = 51,
        
    ) -> tuple:


    # Initial guess for root finding
    xr0 = 0.3
    xs0 = 3.8

    gth = 5.0 / 3.0
    eth1 = thermal_pressure_right / (gth - 1.0)

    gCR = 4.0 / 3.0
    gCRi = 4.0 / 3.0
    PCR5 = thermal_to_cosmic_ray_pressure_ratio_left * thermal_pressure_left
    PCR1 = thermal_to_cosmic_ray_pressure_ratio_right * thermal_pressure_right
    eCR1 = PCR1 / (gCR - 1.0)
    Xinj = injection_efficiency / (1.0 - injection_efficiency)

    P5 = PCR5 + thermal_pressure_left
    P1 = PCR1 + thermal_pressure_right
    eps1 = eCR1 + eth1

    g5 = (gCR * PCR5 + gth * thermal_pressure_left) / P5 if P5 != 0 else (gCR + gth) / 2.0
    c5 = np.sqrt(g5 * P5 / density_left)

    def P3(xr):
        return PCR5 * xr**gCR + thermal_pressure_left * xr**gth

    def PCR2(xs):
        return PCR1 * xs**gCR

    def eCR2(xs):
        return PCR2(xs) / (gCR - 1.0)

    def ethad(xs):
        return eth1 * xs**gth

    fac = 1.0 / ((1.0 - injection_efficiency) * (gth - 1.0) / (gCRi - 1.0) + injection_efficiency)

    def eps2(xr, xs):
        return fac * (P3(xr) / (gCRi - 1.0) + Xinj * ethad(xs) - (gCR - 1.0) / (gCRi - 1.0) * eCR2(xs)) \
            + eCR2(xs) - Xinj * ethad(xs)

    ACR5 = PCR5 * density_left**(-gCR)
    Ath5 = thermal_pressure_left * density_left**(-gth)

    # Integrand function for the NIntegrate part
    def integrand(r, gCR_val, ACR5_val, gth_val, Ath5_val):
        term1 = gCR_val * ACR5_val * r**(gCR_val - 3.0)
        term2 = gth_val * Ath5_val * r**(gth_val - 3.0)
        value_inside_sqrt = term1 + term2
        if value_inside_sqrt < 0:
            value_inside_sqrt = 0
        return np.sqrt(value_inside_sqrt)

    def Int_rho(rho):
        result, _ = quad(integrand, 0, rho, args=(gCR, ACR5, gth, Ath5))
        return result

    def equations(vars):
        xr, xs = vars
        if xr <= 0 or xs <= 1:
            return [1e10, 1e10]

        p3_val = P3(xr)

        rho3_calc = xr * density_left

        V_density_left = Int_rho(density_left)
        V_rho3 = Int_rho(rho3_calc)
        diff_V = V_density_left - V_rho3

        # Calculate f1
        f1_val = (p3_val - P1) * (xs - 1.0) - density_right * xs * (diff_V)**2

        # Calculate f2
        eps2_val = eps2(xr, xs)
        f2_val = eps2_val - xs * eps1 - 0.5 * (p3_val + P1) * (xs - 1.0)

        return [f1_val, f2_val]

    # Initial guess vector
    initial_guess = [xr0, xs0]

    # Solve the system
    sol = root(equations, initial_guess, method='lm', tol=1e-7) # Adjust tol as needed

    if not sol.success:
        print("Warning: Root finding did not converge.")
        print(f"Message: {sol.message}")

    xr_sol, xs_sol = sol.x

    rho3 = xr_sol * density_left
    rho2 = xs_sol * density_right
    p3_sol = P3(xr_sol)


    pressure_diff_term = p3_sol - P1
    if pressure_diff_term < 0:
        v3 = np.nan
    else:
        v3 = np.sqrt(pressure_diff_term * (xs_sol - 1.0) / (xs_sol * density_right))

    if rho2 == density_right:
        vs = v3
    else:
        vs = rho2 * v3 / (rho2 - density_right)


    c_rho3 = np.sqrt(gCR * ACR5 * rho3**(gCR - 1.0) + gth * Ath5 * rho3**(gth - 1.0))

    V_density_left_vt = Int_rho(density_left)
    V_rho3_vt = Int_rho(rho3)
    diff_V_vt = V_density_left_vt - V_rho3_vt

    vt = -diff_V_vt + c_rho3

    PCR3 = PCR5 * (rho3 / density_left)**gCR
    Pth3 = p3_sol - PCR3
    Pth2 = 1.0 / (1.0 + injection_efficiency) * (p3_sol + Xinj * (gCRi - 1.0) / (gth - 1.0) * thermal_pressure_right * xs_sol**gth - PCR1 * xs_sol**gCR)
    Pinj = Xinj * (gCRi - 1.0) / (gth - 1.0) * (Pth2 - thermal_pressure_right * xs_sol**gth)

    x_rarefaction_left = initial_shock_pos - c5 * t_end
    x_rarefaction_right = initial_shock_pos - vt * t_end
    x_contact_discontinuity = initial_shock_pos + v3 * t_end
    x_shock = initial_shock_pos + vs * t_end

    rho_results = np.full((num_rarefaction_evals,), np.nan)

    t = t_end
    x = -c5 * t + (-vt + c5) * t * np.linspace(0, num_rarefaction_evals, num_rarefaction_evals + 1) / num_rarefaction_evals

    for j in range(1, num_rarefaction_evals + 1):
        x_j = x[j]

        def rf(rho):
            if rho <= 0 or rho > density_left:
                return 1e10

            c_rho = np.sqrt(gCR * ACR5 * rho**(gCR - 1.0) + gth * Ath5 * rho**(gth - 1.0))

            rf_val = Int_rho(rho) - Int_rho(density_left) + x_j / t + c_rho

            return rf_val

        min_rho_bound = rho3 * (1 - 1e-6) # Slightly less than rho3
        max_rho_bound = density_left * (1 + 1e-6) # Slightly more than density_left (but clamp later)

        # Ensure bounds are valid
        if min_rho_bound <= 0: min_rho_bound = 1e-9

        sol_rf = root_scalar(rf, bracket=(min_rho_bound, max_rho_bound), method='brentq', xtol=1e-7)

        if sol_rf.converged:
            rho_rf_root = sol_rf.root
        else:
            print(f"Warning: root_scalar failed to converge for t={t:.2f}, j={j}. Flag: {sol_rf.flag}")
            rho_rf_root = np.nan

        j_idx = j - 1
        if 0 <= j_idx < rho_results.shape[0]:
            # Clamp density to physical bounds [0, density_left]
            if rho_rf_root < 0: rho_rf_root = 0.0
            if rho_rf_root > density_left: rho_rf_root = density_left # Should ideally not happen if physics/solver ok
            rho_results[j_idx] = rho_rf_root


    rhorf = rho_results

    rho_full = np.concatenate((
        np.array([density_left, density_left]),
        rho_results,
        np.array([rho3, rho3]),
        np.array([rho2, rho2]),
        np.array([density_right, density_right]),
    ))

    x_end = -c5 * t_end + (-vt + c5) * t_end * np.linspace(0, num_rarefaction_evals, num_rarefaction_evals + 1) / num_rarefaction_evals

    ACR5  = PCR5 * density_left**(-gCR) 
    Ath5  = thermal_pressure_left * density_left**(-gth) 
    Pthrf = Ath5 * rhorf**gth
    PCRrf = ACR5 * rhorf**gCR
    vrf   = x_end[1:] / t + np.sqrt(gCR * ACR5 * rhorf**(gCR-1) + gth * Ath5 * rhorf**(gth-1))

    velocity_full = np.concatenate((
        np.array([0.0, 0.0]),
        vrf,
        np.array([v3, v3]),
        np.array([v3, v3]),
        np.array([0.0, 0.0]),
    ))

    thermal_pressure_full = np.concatenate((
        np.array([thermal_pressure_left, thermal_pressure_left]),
        Pthrf,
        np.array([Pth3, Pth3]),
        np.array([Pth2, Pth2]),
        np.array([thermal_pressure_right, thermal_pressure_right]),
    ))

    cosmic_ray_pressure_full = np.concatenate((
        np.array([PCR5, PCR5]),
        PCRrf,
        np.array([PCR3, PCR3]),
        np.array([PCR2(xs_sol) + Pinj, PCR2(xs_sol) + Pinj]),
        np.array([PCR1, PCR1]),
    ))

    total_pressure_full = thermal_pressure_full + cosmic_ray_pressure_full

    x_full = np.concatenate((
        np.array([left_boundary, x_rarefaction_left]),
        x_end[1:] + initial_shock_pos,
        np.array([x_rarefaction_right, x_contact_discontinuity]),
        np.array([x_contact_discontinuity, x_shock]),
        np.array([x_shock, right_boundary]),
    ))

    return (
        x_full,
        rho_full,
        velocity_full,
        thermal_pressure_full,
        cosmic_ray_pressure_full,
        total_pressure_full,
    )


if __name__ == "__main__":
    x_full, rho_full, velocity_full, thermal_pressure_full, cosmic_ray_pressure_full, total_pressure_full = get_cosmic_ray_analytic_solution()
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(x_full, rho_full)
    axs[0].set_title("Density")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("Density")
    axs[1].plot(x_full, velocity_full)
    axs[1].set_title("Velocity")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Velocity")
    axs[2].plot(x_full, total_pressure_full, label="Total Pressure")
    axs[2].plot(x_full, thermal_pressure_full, label="Thermal Pressure")
    axs[2].plot(x_full, cosmic_ray_pressure_full, label="Cosmic Ray Pressure")
    axs[2].legend()
    axs[2].set_title("Pressure")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("Pressure")
    plt.tight_layout()
    plt.savefig("cosmic_ray_analytic_solution.png")