import numpy as np
import efficientmc as mc
from functools import partial

def runtests(allgenerators, partialmarkets, partialassets):
    """
    Calcule la MtM et le delta de tous les actifs de `partialassets`
    en leur associant les marchés `partialmarkets` et en utilisant
    successivement les générateurs Monte Carlo de `allgenerators`.

    Paramètres :
    ------------
    allgenerators
        Liste de générateurs Monte Carlo.
    partialmarkets
        Liste de constructeur permettant d'instancier des marchés dont
        tous les paramètres ont été fixés hormis le générateur Monte
        Carlo.
    allassets
        Liste de constructeurs permettant d'instancier des actifs dont
        tous les paramètres ont été fixés hormis le marché.
    """
    for key, gen in allgenerators.items():
        allmarkets = {key: mkt(gen) for key, mkt in partialmarkets.items()}
        allassets = [asset(**allmarkets) for asset in partialassets]
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
                     "Halton 2": mc.generators.GaussianGenerator(50000, np.array([[1.]]), ["BlackScholes"], mc.generators.halton2),
                     "Halton F": mc.generators.GaussianGenerator(5000, np.array([[1.]]), ["BlackScholes"], mc.generators.haltonF)}
    PARTIALMARKETS = {"market": partial(mc.pricemodels.BlackScholesModel, "BlackScholes", 100., 0., 0.2)}
    PARTIALASSETS = [partial(mc.assets.EuropeanCall, name="itm", strike=90., maturity=1.),
                     partial(mc.assets.EuropeanCall, name="atm", strike=100., maturity=1.),
                     partial(mc.assets.EuropeanCall, name="otm", strike=110., maturity=1.),]
    runtests(ALLGENERATORS, PARTIALMARKETS, PARTIALASSETS)

    # Spreads européens, modèle de Black-Scholes :
    ALLGENERATORS = {"Classique": mc.generators.GaussianGenerator(500000, np.array([[1.0, 0.5], [0.5, 1.0]]), ["BlackScholes1", "BlackScholes2"], np.random.randn),
                     "Antithétique": mc.generators.GaussianGenerator(500000, np.array([[1.0, 0.5], [0.5, 1.0]]), ["BlackScholes1", "BlackScholes2"], mc.generators.antithetic_randn),
                     "Sobol": mc.generators.GaussianGenerator(50000, np.array([[1.0, 0.5], [0.5, 1.0]]), ["BlackScholes1", "BlackScholes2"], mc.generators.sobol),
                     "Van Der Corput": mc.generators.GaussianGenerator(5000, np.array([[1.0, 0.5], [0.5, 1.0]]), ["BlackScholes1", "BlackScholes2"], mc.generators.van_der_corput_dimension),
                     "Halton": mc.generators.GaussianGenerator(500, np.array([[1.0, 0.5], [0.5, 1.0]]), ["BlackScholes1", "BlackScholes2"], mc.generators.halton),
                     "Halton 2": mc.generators.GaussianGenerator(50000, np.array([[1.0, 0.5], [0.5, 1.0]]), ["BlackScholes1", "BlackScholes2"], mc.generators.halton2),
                     "Halton F": mc.generators.GaussianGenerator(5000, np.array([[1.0, 0.5], [0.5, 1.0]]), ["BlackScholes1", "BlackScholes2"], mc.generators.haltonF)}
    PARTIALMARKETS = {"market1": partial(mc.pricemodels.BlackScholesModel, "BlackScholes1", 100., 0., 0.2),
                      "market2": partial(mc.pricemodels.BlackScholesModel, "BlackScholes2", 100., 0., 0.2)}
    PARTIALASSETS = [partial(mc.assets.EuropeanSpread, name="spread", maturity=1.)]
    runtests(ALLGENERATORS, PARTIALMARKETS, PARTIALASSETS)
    
    # Basket option, modèle de Black-Scholes :
    ALLGENERATORS = {"Classique": mc.generators.GaussianGenerator(500000, np.array([[1.0, 0.5], [0.5, 1.0]]), ["MultiAssetsBlackScholes"], np.random.randn),
                     "Antithétique": mc.generators.GaussianGenerator(500000, np.array([[1.0, 0.5], [0.5, 1.0]]), ["MultiAssetsBlackScholes"], mc.generators.antithetic_randn),
                     "Sobol": mc.generators.GaussianGenerator(50000, np.array([[1.0, 0.5], [0.5, 1.0]]), ["MultiAssetsBlackScholes"], mc.generators.sobol),
                     "Van Der Corput": mc.generators.GaussianGenerator(5000, np.array([[1.0, 0.5], [0.5, 1.0]]), ["MultiAssetsBlackScholes"], mc.generators.van_der_corput_dimension),
                     "Halton": mc.generators.GaussianGenerator(500, np.array([[1.0, 0.5], [0.5, 1.0]]), ["MultiAssetsBlackScholes"], mc.generators.halton),
                     "Halton 2": mc.generators.GaussianGenerator(50000, np.array([[1.0, 0.5], [0.5, 1.0]]), ["MultiAssetsBlackScholes"], mc.generators.halton2),
                     "Halton F": mc.generators.GaussianGenerator(5000, np.array([[1.0, 0.5], [0.5, 1.0]]), ["MultiAssetsBlackScholes"], mc.generators.haltonF)}
    PARTIALMARKETS = {"markets": partial(mc.pricemodels.MultiAssetsBlackScholesModel, "MultiAssetsBlackScholes", 2, np.array([200., 190.]), 0., np.array([0.2, 0.2])) }
    PARTIALASSETS = [partial(mc.assets.BasketOption, name="basket", typeO="call",numbersU=2, numbersA=np.array([2., 2.]), maturity=1.,strike=110.)]
    runtests(ALLGENERATORS, PARTIALMARKETS, PARTIALASSETS)
