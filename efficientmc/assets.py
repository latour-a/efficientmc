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
        "Renvoie l'ensemble des marchés auxquels est exposé l'actif."
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

    @timecached
    def getvolume(self, date, market):
        """
        Renvoie les volumes exercés au titre de l'option sur le marché
        `market` à la date `date`.
        """
        if market == self.market and date == self.maturity:
            #FIXME: même remarque que pour `getcf`.
            prices = self.market.getspot(date)
            return np.where(prices > self.strike, 1., 0.)
        else:
            return 0.

class EuropeanSpread:
    "Spread européen : option de payoff :math:`(S^1_T - S^2_T)^+`."

    def __init__(self, name, market1, market2, maturity):
        """
        Initialise une nouvelle instance de la classe `EuropeanSpread`, de
        payoff :math:`(S^1_T - S^2_T)^+`.

        Paramètres :
        ------------
        name
            Identifiant associé à l'instance.
        market1
            Le premier marché sur lequel porte l'option (:math:`(S^1_t)`) ;
            le sous-jacent est le produit spot.
        market2
            Le second marché sur lequel porte l'option (:math:`(S^2_t)`) ;
            le sous-jacent est le produit spot.
        maturity : date
            La  maturité de l'option (:math:`T`).
        """
        self.name = name
        self.market1 = market1
        self.market2 = market2
        self.maturity = maturity
        self.cache = DateCache()

    def getdates(self):
        """
        Renvoie l'ensemble des dates pour lesquelles l'objet nécessite
        un calcul spécifique.
        """
        return [self.maturity]

    def getmarkets(self):
        "Renvoie l'ensemble des marchés auxquels est exposé l'actif."
        return [self.market1, self.market2]

    @timecached
    def getcf(self, date):
        "Renvoie les cash-flows générés par l'option à la date `date`."
        if date == self.maturity:
            #FIXME: l'option peut aussi porter sur un forward, introduire
            # plutôt la notion de produit.
            #FIXME: les deux marchés peuvent ne pas être dans la même
            # monnaie.
            prices1 = self.market1.getspot(date)
            prices2 = self.market2.getspot(date)
            return np.maximum(prices1 - prices2, 0.)
        else:
            return 0.

    @timecached
    def get_discounted_cf(self, date):
        """
        Renvoie les cash-flows actualisés générés par l'option à la
        date `date`.
        """
        if date == self.maturity:
            # Par convention, les cash-flows sont exprimés dans la devise
            # de `self.market1`.
           return self.market1.getdf(date) * self.getcf(date)
        else:
            return 0.

    @timecached
    def getvolume(self, date, market):
        """
        Renvoie les volumes exercés au titre de l'option sur le marché
        `market` à la date `date`.
        """
        if date == self.maturity and market in self.getmarkets():
            #FIXME: même remarque que pour `getcf`.
            prices1 = self.market1.getspot(date)
            prices2 = self.market2.getspot(date)
            if market == self.market1:
                return np.where(prices1 > prices2, 1., 0.)
            elif market == self.market2:
                return np.where(prices1 > prices2, -1., 0.)
        else:
            return 0.

class BasketOption:
    "Basket option: math:`\sum_{i=1}^{numbersU}w_i*S_{t}^i`."
    
    def __init__(self,name,markets,typeO,numbersU,numbersA,maturity,strike):
        
        """
        Paramètres :
        ------------
        name
            Identifiant associé à l'instance.
        markets
            Les marchés sur lequel porte l'option (:math:`(S_t^i)`) ; le
            sous-jacent est le produit spot.
        typeO : 
            Le type de l'option ou l'option appliqué au panier
        numbersU: int
            Le nombre d'actifs choisis par l'utilisateur
        numberA: double
            Le nombre de chaque actif dans le panier (\sum_{i=1}^{numbersU}w_i=1)
        sum: double
            (\sum_{i=1}^{numbersU}w_i*S_{t}^i)
        maturity : date 
            La  maturité de l'option (:math:`T`).
        strike : double (optionnel si le type de l'option demandée demande un strike comme un call)
            Le strike de l'option (:math:`K`).
        """
        self.name = name
        self.markets = markets
        self.typeO=typeO
        self.numbersU=numbersU
        self.numbersA=numbersA
        self.maturity = maturity
        self.strike = strike
        self.cache = DateCache()
    
    def getdates(self):
        """
        Renvoie l'ensemble des dates pour lesquelles l'objet nécessite
        un calcul spécifique.
        """
        return [self.maturity]
    
    def getmarkets(self):
        return [self.markets]
    
    def getweight(self,index):
        compt=0
        totalweight=0
        weight=np.empty((len(self.numbersA)))
        for i in range(0,len(self.numbersA),1):
            totalweight+=self.numbersA[i]
     
        for i in range(0,len(self.numbersA)-1,1):
            weight[i]=self.numbersA[i]/totalweight
        return weight[index]
    
    def getsum(self,date):
        prices=0
        for i in range(0,len(self.numbersA),1):
            prices+= self.markets.getspot(date,i)*self.getweight(i)
        return prices
    
    @timecached
    def getcf(self,date):
        "Renvoie les cash-flows générés par l'option à la date `date`."
        if date == self.maturity:
            #FIXME: l'option peut aussi porter sur un forward, introduire
            # plutôt la notion de produit.
            if self.typeO=="call":
                return np.maximum(self.getsum(date) - self.strike, 0.)
        else:
            return 0
        
    @timecached
    def get_discounted_cf(self, date):
        """
        Renvoie les cash-flows actualisés générés par l'option à la
        date `date`.
        """
        if date == self.maturity:
           return self.markets.getdf(date) * self.getcf(date)
        else:
            return 0.

    @timecached
    def getvolume(self, date, market):
        """
        Renvoie les volumes exercés au titre de l'option sur le marché
        `market` à la date `date`.
        """
        if market == self.markets and date == self.maturity:
            #FIXME: même remarque que pour `getcf`.
            prices = self.getsum(date)
            return np.where(prices > self.strike, 1., 0.)
        else:
            return 0.
        
    
        
