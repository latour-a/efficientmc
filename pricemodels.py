import numpy as np
from abc import ABCMeta, abstractmethod

class PriceModel(metaclass=ABCMeta):
    "Modèle de prix (classe abstraite)."

    @abstractmethod
    def simulate(self, prevdate, prevprices date):
        """
        Simule des prix dans le modèle.

        Paramètres
        ----------
        prevdate : date
            Date correspondant à la précédente simulation.
        prevprices : double ou numpy.array
            Prix simulés pour `prevdate`.
        date : date
            Date à laquelle simuler les prix.
        """
        #TODO: pourquoi est-ce que ce n'est pas une méthode à part qui mange
        # un modèle de prix ? En effet, on va vouloir un simulateur qui
        # implèmente un cache... mais c'est quelque chose de tout à fait
        # différent du modèle en lui-même.
        pass

    @abstractmethod
    def getdf(self, start, end):
        "Calcule le facteur d'actualisation entre `start` et `end`."
        #TODO: ne fonctionne pas en pratique (il faut gérer les différentes devises).
        pass

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
