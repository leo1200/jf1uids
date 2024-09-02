

from jf1uids.CFL import cfl_time_step
from jf1uids.muscl_scheme import evolve_state

def time_integration(primitive_state, dt_max, C_cfl, dx, gamma, t_end):
    progress = 0
    t = 0
    while t < t_end:
        dt = cfl_time_step(primitive_state, dx, dt_max, gamma, C_cfl)
        primitive_state = evolve_state(primitive_state, dx, dt, gamma)
        t += dt
        progress_new = int(t / t_end * 100)
        if progress_new > progress:
            print(f"Progress: {progress_new}%")
            progress = progress_new
    
    return primitive_state