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
sims = mc.runmc(otm, atm, itm)
res = mc.postpro(sims)
for k, v in res.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))

# Monte Carlo antithétique :
gen = mc.generators.GaussianGenerator(500000, np.array([[1.]]), ["BlackScholes"],
                                      mc.generators.antithetic_randn)
market = mc.pricemodels.BlackScholesModel("BlackScholes", 100., 0., 0.2, gen)
itm = mc.assets.EuropeanCall("itm", market, strike=90., maturity=1.)
atm = mc.assets.EuropeanCall("atm", market, strike=100., maturity=1.)
otm = mc.assets.EuropeanCall("otm", market, strike=110., maturity=1.)
sims = mc.runmc(otm, atm, itm)
res = mc.postpro(sims)
for k, v in res.items():
    print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))
