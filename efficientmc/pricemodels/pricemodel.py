import numpy as np
from abc import ABCMeta, abstractmethod

class PriceModel(metaclass=ABCMeta):
    "Modèle de prix (classe abstraite)."

    @abstractmethod
    def simulate(self, date, prevdate, prevprices, **kwargs):
        """
        Simule des prix dans le modèle.

        Paramètres
        ----------
        date : date
            Date à laquelle simuler les prix.
        prevdate : date
            Date correspondant à la précédente simulation.
        prevprices : double ou numpy.array
            Prix simulés pour `prevdate`.
        kwargs
            Arguments optionnels.
        """
        pass

    @abstractmethod
    def getdf(self, start, end):
        "Calcule le facteur d'actualisation entre `start` et `end`."
        #TODO: ne fonctionne pas en pratique (il faut gérer les différentes devises).
        pass
