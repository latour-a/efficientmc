import numpy as np
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
            raise NotImplementedError

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

    @abstractmethod
    def getdates(self, begin):
        """
        Renvoie les dates supérieures ou égales à `begin` pour lesquelles
        l'actif verse (ou est susceptible de verser) un cashflow.

        Paramètres
        ----------
        begin : date
            Limite inférieure de la liste de dates à renvoyer.
        """
        pass
