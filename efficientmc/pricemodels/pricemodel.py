import numpy as np
from abc import ABCMeta, abstractmethod

class PriceModel(metaclass=ABCMeta):
    "Modèle de prix (classe abstraite)."

    @abstractmethod
    def simulate(self, prevdate, prevprices, date):
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
