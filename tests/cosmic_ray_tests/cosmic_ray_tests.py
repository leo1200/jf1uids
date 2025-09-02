# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

from shock_tube import shock_tube_test
from sedov_crs import sedov_cr_test
from philipp_sedov import philipp_sedov

# shock tube with and without
# diffusive shock acceleration
shock_tube_test()

# Sedov blast wave with
# cosmic rays and diffusive
# shock acceleration
sedov_cr_test()

# Philipps Sedov tests
philipp_sedov()