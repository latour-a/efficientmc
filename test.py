import numpy as np
import efficientmc as mc

# Graine fixée afin d'avoir des résultats reproductibles.
np.random.seed(0)

# Monte Carlo de base :
gen = mc.generators.GaussianGenerator(500000, np.array([[1.]]),
                                      ["BlackScholes"], np.random.randn)
market = mc.pricemodels.BlackScholesModel("BlackScholes", 100., 0., 0.2, gen)
itm = mc.assets.EuropeanCall("itm", market, strike=90., maturity=1.)
atm = mc.assets.EuropeanCall("atm", market, strike=100., maturity=1.)
otm = mc.assets.EuropeanCall("otm", market, strike=110., maturity=1.)
cf, volumes, prices = mc.runmc(otm, atm, itm)
mtm = mc.getmtm(cf)
print("Classique - MtM")
for k, v in mtm.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))
delta = mc.getdelta(volumes, prices)
print("Classique - Delta")
for k, v in delta.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))

# Monte Carlo antithétique :
gen = mc.generators.GaussianGenerator(500000, np.array([[1.]]), ["BlackScholes"],
                                      mc.generators.antithetic_randn)
market = mc.pricemodels.BlackScholesModel("BlackScholes", 100., 0., 0.2, gen)
itm = mc.assets.EuropeanCall("itm", market, strike=90., maturity=1.)
atm = mc.assets.EuropeanCall("atm", market, strike=100., maturity=1.)
otm = mc.assets.EuropeanCall("otm", market, strike=110., maturity=1.)
cf, volumes, prices = mc.runmc(otm, atm, itm)
mtm = mc.getmtm(cf)
print("\nAntithétique - Mtm:")
for k, v in mtm.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))
delta = mc.getdelta(volumes, prices)
print("Antithétique - Delta")
for k, v in delta.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))

# Quasi Monte Carlo avec suite de Sobol
"""
On réduit le nombre de simulations nsims car les nombres générés ont des
propriétés spéciales (équirépartition), cela accélère la convergence du calcul
et donc le temps de calcul
On obtient des valeurs similaires au Monte Carlo classique avec moins de simulations
"""
gen = mc.generators.GaussianGenerator(50000, np.array([[1.]]), ["BlackScholes"],
                                      mc.generators.sobol)
market = mc.pricemodels.BlackScholesModel("BlackScholes", 100., 0., 0.2, gen)
itm = mc.assets.EuropeanCall("itm", market, strike=90., maturity=1.)
atm = mc.assets.EuropeanCall("atm", market, strike=100., maturity=1.)
otm = mc.assets.EuropeanCall("otm", market, strike=110., maturity=1.)
cf, volumes, prices = mc.runmc(otm, atm, itm)
mtm = mc.getmtm(cf)
print("\nQuasi Monte Carlo Sobol :")
for k, v in mtm.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))
delta = mc.getdelta(volumes, prices)
print("Quasi Monte Carlo Sobol-Delta")
for k, v in delta.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))

# Quasi Monte Carlo avec suite de Van Der Corput
gen = mc.generators.GaussianGenerator(5000, np.array([[1.]]), ["BlackScholes"],
                                      mc.generators.van_der_corput_dimension)
market = mc.pricemodels.BlackScholesModel("BlackScholes", 100., 0., 0.2, gen)
itm = mc.assets.EuropeanCall("itm", market, strike=90., maturity=1.)
atm = mc.assets.EuropeanCall("atm", market, strike=100., maturity=1.)
otm = mc.assets.EuropeanCall("otm", market, strike=110., maturity=1.)
cf, volumes, prices = mc.runmc(otm, atm, itm)
mtm = mc.getmtm(cf)
print("\nQuasi Monte Carlo Van Der Corput :")
for k, v in mtm.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))
delta = mc.getdelta(volumes, prices)
print("Quasi Monte Carlo Van Der Corput-Delta")
for k, v in delta.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))
    
# Quasi Monte Carlo avec suite de Halton:
gen = mc.generators.GaussianGenerator(500, np.array([[1.]]), ["BlackScholes"],
                                      mc.generators.halton)
market = mc.pricemodels.BlackScholesModel("BlackScholes", 100., 0., 0.2, gen)
itm = mc.assets.EuropeanCall("itm", market, strike=90., maturity=1.)
atm = mc.assets.EuropeanCall("atm", market, strike=100., maturity=1.)
otm = mc.assets.EuropeanCall("otm", market, strike=110., maturity=1.)
cf, volumes, prices = mc.runmc(otm, atm, itm)
mtm = mc.getmtm(cf)
print("\nQuasi Monte Carlo Halton - Mtm:")
for k, v in mtm.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))
delta = mc.getdelta(volumes, prices)
print("Quasi Monte Carlo Halton - Delta")
for k, v in delta.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))

# Quasi Monte Carlo avec suite de Halton version 2:
gen = mc.generators.GaussianGenerator(50000, np.array([[1.]]), ["BlackScholes"],
                                      mc.generators.halton2)
market = mc.pricemodels.BlackScholesModel("BlackScholes", 100., 0., 0.2, gen)
itm = mc.assets.EuropeanCall("itm", market, strike=90., maturity=1.)
atm = mc.assets.EuropeanCall("atm", market, strike=100., maturity=1.)
otm = mc.assets.EuropeanCall("otm", market, strike=110., maturity=1.)
cf, volumes, prices = mc.runmc(otm, atm, itm)
mtm = mc.getmtm(cf)
print("\nQuasi Monte Carlo Halton 2 - Mtm:")
for k, v in mtm.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))
delta = mc.getdelta(volumes, prices)
print("Quasi Monte Carlo Halton 2- Delta")
for k, v in delta.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))
