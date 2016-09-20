import numpy as np
import scipy.stats as sps
from abc import ABCMeta, abstractmethod

class Asset(metaclass=ABCMeta):
    """
    TODO
    """

    def getprice(self, date, price, model, analytical=False, nsims=None):
        """
        TODO
        """
        if analytical:
            modelname = type(model).__name__
            try:
                func = getattr(self, "_get_" + modelname + "_price")
            except AttributeError:
                raise NotImplementedError("no analytical method implemented "\
                                          "for option %s with model %s." % \
                                          (type(self).__name__, modelname))
            return func(date, price, model)
        else: # Monte Carlo
            #TODO: le nsims n'est pas clair.
            #TODO: gérer la pondération (dans le cas de l'importance sampling).
            #TODO: tel quel, absurde de pouvoir donner un tableau (de manière
            # sous-jacente, on suppose quand même qu'ils partent tous d'une
            # même valeur initiale).
            if not np.array(price).shape and nsims > 0:
                price = price + np.zeros((nsims))
            values = self.computepayoff(date, price, model)
            return np.mean(values)

    def getdelta(self, date, price, model, analytical=False, simulator=None):
        """
        TODO
        """
        if analytical:
            raise NotImplementedError("TODO")
        else: # Monte Carlo
            if not simulator:
                simulator = None # Simulateur par défaut.
            #TODO: tenir compte des taux.
            raise NotImplementedError("TODO")

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
    """
    TODO
    """

    def __init__(self, strike, maturity):
        """
        Initialise une nouvelle instance de `EuropeanCall`.

        Paramètres
        ----------
        strike : double
            Strike de l'option.
        maturity : double
            Maturité de l'option (i.e. : la date d'exercice est `maturity`).
        """
        self.strike = strike
        self.maturity = maturity

    def computepayoff(self, date, price, model):
        #TODO: on est en train de mélanger Monte Carlo et calcul du payoff.
        #TODO: faire un décorateur pour la vérification des dates...?
        if date <= self.maturity:
            df = model.getdf(date, self.maturity)
            newprices = model.simulate(date, price, self.maturity)
            return df * np.maximum(newprices - self.strike, 0.)
        else:
            return 0.

    def _get_BlackScholes_d1(self, date, price, model):
        """
        TODO
        """
        #TODO: en l'occurrence, on pourrait faire une sous-classe qui hérite
        # de EuropeanCall, qui contient les bouts de code spécifiques à
        # Black-Scholes, et qui est instanciée à la volée par price : permet
        # de davantage séparer les calculs pour les différents modèles.
        ttm = self.maturity - date
        d1 = (np.log(price / self.strike) + (0.5 * model.sigma**2 + model.rate) * ttm) \
             / (model.sigma * np.sqrt(ttm))
        return d1

    def _get_BlackScholes_price(self, date, price, model):
        """
        TODO
        """
        if date < self.maturity:
            d1 = self._get_BlackScholes_d1(date, price, model)
            d2 = d1 - model.sigma * np.sqrt(self.maturity - date)
            df = model.getdf(date, self.maturity)
            return price * sps.norm.cdf(d1) - self.strike * df * sps.norm.cdf(d2)
        elif date == self.maturity:
            #TODO: bancal.
            return self.computepayoff(date, price, model)
        else:
            return 0.

    def _get_BlackScholes_delta(self, date, price):
        """
        TODO
        """
        if date < self.maturity:
            d1 = self._get_BlackScholes_d1(date, price, model)
            return sps.norm.cdf(d1)
        elif date == self.maturity:
            return np.where(price > self.strike, 1., 0.)
        else:
            return 0.
