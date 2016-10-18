import numpy as np
from efficientmc.utils import timecached, DateCache

class EuropeanCall:
    "Call européen : option de payoff :math:`(S_T - K)^+`."

    def __init__(self, name, market, strike, maturity):
        """
        Initialise une nouvelle instance de la classe `EuropeanCall`, de
        payoff :math:`(S_T - K)^+`.

        Paramètres :
        ------------
        name
            Identifiant associé à l'instance.
        market
            Le marché sur lequel porte l'option (:math:`(S_t)`) ; le
            sous-jacent est le produit spot.
        strike : double
            Le strike de l'option (:math:`K`).
        maturity : date
            La  maturité de l'option (:math:`T`).
        """
        self.name = name
        self.market = market
        self.strike = strike
        self.maturity = maturity
        self.cache = DateCache()

    def getdates(self):
        """
        Renvoie l'ensemble des dates pour lesquelles l'objet nécessite
        un calcul spécifique.
        """
        return [self.maturity]

    def getmarkets(self):
        """
        Renvoie l'ensemble des marchés auxquels est exposé l'actif.
        """
        return [self.market]

    @timecached
    def getcf(self, date):
        "Renvoie les cash-flows générés par l'option à la date `date`."
        if date == self.maturity:
            #FIXME: l'option peut aussi porter sur un forward, introduire
            # plutôt la notion de produit.
            prices = self.market.getspot(date)
            return np.maximum(prices - self.strike, 0.)
        else:
            return 0.

    @timecached
    def get_discounted_cf(self, date):
        """
        Renvoie les cash-flows actualisés générés par l'option à la
        date `date`.
        """
        if date == self.maturity:
           return self.market.getdf(date) * self.getcf(date)
        else:
            return 0.
