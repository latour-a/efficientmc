import numpy as np
import scipy.stats as sps
from abc import ABCMeta, abstractmethod

class Asset(metaclass=ABCMeta):
    "Actif financier à valoriser et à couvrir."

    def getprice(self, date, price, model, analytical=False, nsims=None):
        """
        Calcule le prix de l'actif.

        Paramètres
        -----------
        date
            Date à laquelle calculer le prix de l'actif.
        price
            Valeur des sous-jacents à la date `date`.
        model
            Modèle de prix utilisé pour les sous-jacents.
        analytical : booléen, `False` par défaut
            Utilise une formule fermée pour effectuer le calcul.
        nsims : entier ou `None`
            Nombre de simulations à utiliser dans le cadre d'un calcul
            Monte Carlo.
        """
        if analytical:
            modelname = type(model).__name__
            try:
                func = getattr(self, "_get_" + modelname + "_price")
            except AttributeError:
                raise NotImplementedError("no analytical method implemented "\
                                          "for asset %s with model %s." % \
                                          (type(self).__name__, modelname))
            return func(date, price, model)
        else: # Monte Carlo
            if not np.array(price).shape and nsims > 0:
                price = price + np.zeros((nsims))
            values = self.computepayoff(date, price, model)
            return np.mean(values)

    def getdelta(self, date, price, model, analytical=False, simulator=None):
        """
        Calcule le delta de l'actif.

        Paramètres
        -----------
        date
            Date à laquelle calculer le delta de l'actif.
        price
            Valeur des sous-jacents à la date `date`.
        model
            Modèle de prix utilisé pour les sous-jacents.
        analytical : booléen, `False` par défaut
            Utilise une formule fermée pour effectuer le calcul.
        nsims : entier ou `None`
            Nombre de simulations à utiliser dans le cadre d'un calcul
            Monte Carlo.
        """
        if analytical:
            modelname = type(model).__name__
            try:
                func = getattr(self, "_get_" + modelname + "_delta")
            except AttributeError:
                raise NotImplementedError("no analytical method implemented "\
                                          "for asset %s with model %s." % \
                                          (type(self).__name__, modelname))
            return func(date, price, model)
        else: # Monte Carlo
            raise NotImplementedError

    @abstractmethod
    def computepayoff(self, date, prices, model):
        """
        Calcule le payoff de l'actif à la date `date` pour les prix `prices`
        dans le modèle `model`.

        Paramètres
        ----------
        date : date
            Date à laquelle est effectué le calcul du payoff.
        prices : double ou numpy.array
            Prix à la date `date`.
        model
            Modèle de prix utilisé pour simuler les prix `prices`.
        """
        pass

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

    def _get_BlackScholes_d1(self, date, price, model):
        """
        Calcule le terme :math:`d_1` intervenant dans la formule de
        Black-Scholes.

        Paramètres
        -----------
        date
            Date à laquelle calculer :math:`d_1`.
        price : double ou np.array
            Valeur du sous-jacent à la date `date`.
        model
            Modèle de prix utilisé pour le sous-jacent.
        """
        ttm = self.maturity - date
        d1 = (np.log(price / self.strike) + (0.5 * model.sigma**2 + model.rate) * ttm) \
             / (model.sigma * np.sqrt(ttm))
        return d1

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
            d1 = self._get_BlackScholes_d1(date, price, model)
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
            d1 = self._get_BlackScholes_d1(date, price, model)
            return sps.norm.cdf(d1)
        elif date == self.maturity:
            return np.where(price > self.strike, 1., 0.)
        else:
            return 0.
