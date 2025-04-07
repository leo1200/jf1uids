# For proper cooling something like Grackle
# might be used. For now we are interested in
# the most simple cooling model.

# dE/dt + ... = Phi(T, rho)
# Phi(T, rho) = n_H * Gamma(T) - n_H ^ 2 * Lambda(T)

# for a simple cooling term (Lambda) see 5.3 in
# https://arxiv.org/pdf/2111.03399

# also see
# https://academic.oup.com/mnras/article/502/3/3179/6081066

# and
# https://iopscience.iop.org/article/10.1088/0067-0049/181/2/391

# where this source is handled with
# Brents method? (or Joung & Mac Low 2006)
# see also
# https://iopscience.iop.org/article/10.3847/1538-4357/abc011
