import numpy as np
from efficientmc.pricemodels.pricemodel import PriceModel

class BlackScholes(PriceModel):
    "Modèle de prix de Black-Scholes."

    def __init__(self, initprice, rate, sigma):
        """
        Initialise une nouvelle instance de `BlackScholes`.

        Paramètres
        ----------
        initprice : double
            Valeur initiale (en t=0) de l'actif modélisé.
        rate : double
            Taux court constant.
        sigma : double positif
            Volatilité du modèle.
        """
        self.initprice = initprice
        self.rate = rate
        self.sigma = sigma

    def simulate(self, prevdate, prevprices, date):
        #TODO: la gestion de la discrétisation en temps est à améliorer
        # (notamment pour gérer le CIR du Heston).
        #TODO: faire un décorateur pour ces vérifs là.
        if prevdate < date:
            dt = date - prevdate
            #TODO: on ne peut pas mettre les bruits là-dedans (cas des variables
            # antithétiques...).
            noises = np.random.randn(*(prevprices.shape))
            prices = prevprices * np.exp((self.rate - 0.5 * self.sigma**2) * dt\
                                         + self.sigma * np.sqrt(dt) * noises)
            return prices
        elif prevdate == date:
            return prevprices
        else:
            raise NotImplementedError("brownian bridge not implemented.")

    def getdf(self, start, end):
        return np.exp(-self.rate * (end - start))
