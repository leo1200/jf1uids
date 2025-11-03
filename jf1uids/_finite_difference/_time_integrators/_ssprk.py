from functools import partial
import jax
import jax.numpy as jnp
from typing import Union

from jf1uids._finite_difference._interface_fluxes._weno import (
    _weno_flux_x,
    _weno_flux_y,
    _weno_flux_z,
)

from jf1uids.variable_registry.registered_variables import RegisteredVariables

    
@partial(jax.jit, static_argnames=["registered_variables"])
def _ssprk4(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    registered_variables: RegisteredVariables,
):
    """
    SSPRK matching Fortran ssprk.txt.
    
    Key: weno returns dF which is flux at interfaces.
    For cell i, update is: -dtdl * (dF[i] - dF[i-1])
    where dF[i] is flux at interface i (between cells i and i+1).
    
    Using periodic rolls:
    - roll(dF, -1)[i] = dF[i+1]
    - dF[i] = dF[i]  
    - roll(dF, 1)[i] = dF[i-1]
    
    So: dF[i] - roll(dF, 1)[i] = dF[i] - dF[i-1] ✓
    """
    
    def compute_rhs(state, k2_coeff):
        """Compute RHS matching Fortran."""
        dtdl = k2_coeff * dt / grid_spacing
        
        # Get interface fluxes
        dF_x = _weno_flux_x(state, gamma, registered_variables)
        dF_y = _weno_flux_y(state, gamma, registered_variables)
        dF_z = _weno_flux_z(state, gamma, registered_variables)
        
        # Fortran: qone(m,i) = -dtdl*(dF(m,i)-dF(m,i-1))
        # dF[i] - roll(dF,1)[i] = dF[i] - dF[i-1]
        rhs = -dtdl * (
            (dF_x - jnp.roll(dF_x, 1, axis=1))
            + (dF_y - jnp.roll(dF_y, 1, axis=2))
            + (dF_z - jnp.roll(dF_z, 1, axis=3))
        )
        
        return rhs
    
    q0 = conserved_state
    
    # Stage 1: q1 = q0 + rhs0
    k1_1 = 1.0
    k2_1 = 0.39175222700392
    k3_1 = 0.0
    
    rhs_0 = compute_rhs(q0, k2_1)
    q1 = k1_1 * q0 + rhs_0
    
    # Stage 2: q2 = k1*q0 + k3*q1 + rhs1
    k1_2 = 0.44437049406734
    k2_2 = 0.36841059262959
    k3_2 = 0.55562950593266
    
    rhs_1 = compute_rhs(q1, k2_2)
    q2 = k1_2 * q0 + k3_2 * q1 + rhs_1
    
    # Stage 3: q3 = k1*q0 + k3*q2 + rhs2
    k1_3 = 0.62010185138540
    k2_3 = 0.25189177424738
    k3_3 = 0.37989814861460
    
    rhs_2 = compute_rhs(q2, k2_3)
    q3 = k1_3 * q0 + k3_3 * q2 + rhs_2
    
    # Stage 4: q4 = k1*q0 + k3*q3 + rhs3
    k1_4 = 0.17807995410773
    k2_4 = 0.54497475021237
    k3_4 = 0.82192004589227
    
    rhs_3 = compute_rhs(q3, k2_4)
    q4 = k1_4 * q0 + k3_4 * q3 + rhs_3
    
    # Stage 5: special initialization then update
    k1_5 = -2.081261929715610e-02
    k2_5 = 0.22600748319395
    k3_5 = 5.03580947213895e-01
    k4_5 = 0.51723167208978
    k5_5 = -6.518979800418380e-12
    
    # Fortran: q5 = q0 + k4/k1*q2 + k5/k1*q3 (before RHS)
    q5_init = q0 + (k4_5 / k1_5) * q2 + (k5_5 / k1_5) * q3
    
    rhs_4 = compute_rhs(q4, k2_5)
    
    # Fortran: q5 = k1*q5_init + k3*q4 + rhs4
    q5 = k1_5 * q5_init + k3_5 * q4 + rhs_4
    
    return q5