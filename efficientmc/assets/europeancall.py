import numpy as np
import scipy.stats as sps
from efficientmc.assets.asset import Asset
from efficientmc.assets.utils import get_BlackScholes_d1

class EuropeanCall(Asset):
    "Call européen, option de payoff :math:`(S_T - K)^+`."

    def __init__(self, strike, maturity):
        """
        Initialise une nouvelle instance de `EuropeanCall`.

        Paramètres
        ----------
        strike : double
            Strike de l'option.
        maturity
            Maturité de l'option (i.e. : la date d'exercice est `maturity`).
        """
        self.strike = strike
        self.maturity = maturity

    def computepayoff(self, date, price, model):
        if date <= self.maturity:
            df = model.getdf(date, self.maturity)
            newprices = model.simulate(date, price, self.maturity)
            return df * np.maximum(newprices - self.strike, 0.)
        else:
            return 0.

    def _get_BlackScholes_price(self, date, price, model):
        """
        Calcule le prix de l'option dans le cadre du modèle de Black-Scholes.

        Paramètres
        -----------
        date
            Date à laquelle calculer le prix de l'option.
        price : double ou np.array
            Valeur du sous-jacent à la date `date`.
        model
            Modèle de prix utilisé pour le sous-jacent.
        """
        if date < self.maturity:
            d1 = get_BlackScholes_d1(date, self.maturity, price, self.strike,
                                     model)
            d2 = d1 - model.sigma * np.sqrt(self.maturity - date)
            df = model.getdf(date, self.maturity)
            return price * sps.norm.cdf(d1) - self.strike * df * sps.norm.cdf(d2)
        elif date == self.maturity:
            return self.computepayoff(date, price, model)
        else:
            return 0.

    def _get_BlackScholes_delta(self, date, price, model):
        """
        Calcule le delta de l'option dans le cadre du modèle de Black-Scholes.

        Paramètres
        -----------
        date
            Date à laquelle calculer le delta.
        price : double ou np.array
            Valeur du sous-jacent à la date `date`.
        model
            Modèle de prix utilisé pour le sous-jacent.
        """
        if date < self.maturity:
            d1 = get_BlackScholes_d1(date, self.maturity, price, self.strike,
                                     model)
            return sps.norm.cdf(d1)
        elif date == self.maturity:
            return np.where(price > self.strike, 1., 0.)
        else:
            return 0.
