import numpy as np
import efficientmc as mc
from functools import partial

def runtests(allgenerators, partialmarket, partialassets):
    """
    Calcule la MtM et le delta de tous les actifs de `partialassets` en
    leur associant le marché `partialmarket` et en utilisant successivement
    les générateurs Monte Carlo de `allgenerators`.

    Paramètres :
    ------------
    allgenerators
        Liste de générateurs Monte Carlo.
    partialmarket
        Constructeur permettant d'instancier un marché dont tous les
        paramètres ont été fixés hormis le générateur Monte Carlo.
    allassets
        Liste de constructeurs permettant d'instancier des actifs dont
        tous les paramètres ont été fixés hormis le marché.
    """
    for key, gen in allgenerators.items():
        market = partialmarket(gen)
        allassets = [asset(market=market) for asset in partialassets]
        cf, volumes, prices = mc.runmc(*allassets)
        mtm = mc.getmtm(cf)
        print("%s - MtM" % key)
        for k, v in mtm.items():
            print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))
        delta = mc.getdelta(volumes, prices)
        print("%s - Delta" % key)
        for k, v in delta.items():
            print("%s: %.5f [%.5f, %.5f]" % (k, v.mean, v.iclow, v.icup))
        print("")

if __name__ == '__main__':
    # Graine fixée afin d'avoir des résultats reproductibles.
    np.random.seed(0)

    # Calls européens, modèle de Black-Scholes :
    ALLGENERATORS = {"Classique": mc.generators.GaussianGenerator(500000, np.array([[1.]]), ["BlackScholes"], np.random.randn),
                     "Antithétique": mc.generators.GaussianGenerator(500000, np.array([[1.]]), ["BlackScholes"], mc.generators.antithetic_randn),
                     "Sobol": mc.generators.GaussianGenerator(50000, np.array([[1.]]), ["BlackScholes"], mc.generators.sobol),
                     "Van Der Corput": mc.generators.GaussianGenerator(5000, np.array([[1.]]), ["BlackScholes"], mc.generators.van_der_corput_dimension),
                     "Halton": mc.generators.GaussianGenerator(500, np.array([[1.]]), ["BlackScholes"], mc.generators.halton),
                     "Halton 2": mc.generators.GaussianGenerator(50000, np.array([[1.]]), ["BlackScholes"], mc.generators.halton2)}
    PARTIALMARKET = partial(mc.pricemodels.BlackScholesModel, "BlackScholes", 100., 0., 0.2)
    PARTIALASSETS = [partial(mc.assets.EuropeanCall, name="itm", strike=90., maturity=1.),
                     partial(mc.assets.EuropeanCall, name="atm", strike=100., maturity=1.),
                     partial(mc.assets.EuropeanCall, name="otm", strike=110., maturity=1.),]
    runtests(ALLGENERATORS, PARTIALMARKET, PARTIALASSETS)
