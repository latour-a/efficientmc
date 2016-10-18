import numpy as np
from efficientmc.utils import timecached, DateCache

class BlackScholesModel:
    "Modèle de Black-Scholes : :math:`\frac{dS_t}{S_t} = r dt + \sigma dW_t`."

    def __init__(self, name, initvalue, rate, sigma, randomgen):
        """
        Initialise une nouvelle instance de la classe `BlackScholesModel`,
        de dynamique :
            :math:`\frac{dS_t}{S_t} = r dt + \sigma dW_t`

        Paramètres :
        ------------
        name
            Identifiant associé à l'instance.
        initvalue : flottant
            Valeur initiale de l'actif (:math:`S_0`).
        rate : flottant
            Taux court (:math:`r`).
        sigma : flottant, positif
            Volatilité (:math:`\sigma`).
        randomgen
            Générateur aléatoire permettant de simuler des bruits
            gaussiens.
        """
        self.name = name
        self.initvalue = initvalue
        self.rate = rate
        self.sigma = sigma
        self.randomgen = randomgen
        self.cache = DateCache()

    def getdates(self):
        """
        Renvoie l'ensemble des dates pour lesquelles l'objet nécessite
        un calcul spécifique.
        """
        return [] # Pas besoin de schéma de discrétisation.

    def getnoisekeys(self):
        "Renvoie les identifiants des bruits associés au modèle."
        return (self.name,)

    @timecached
    def simulate(self, date):
        "Simule le modèle à la date `date`."
        try:
            prevdate, prevvalues = self.cache.getprev('simulate')
        except KeyError as e:
            prevdate = 0.
            prevvalues = 1.
        if date < prevdate:
            raise NotImplementedError("brownian bridge not implemented.")
        dt = date - prevdate
        noises = self.randomgen.getnoises(date, self.getnoisekeys()).squeeze()
        incr = np.exp((self.rate - 0.5 * self.sigma**2) * dt \
                      + self.sigma * np.sqrt(dt) * noises)
        return prevvalues * incr

    @timecached
    def getdf(self, date):
        """
        Renvoie le facteur d'actualisation à la date `date` (par rapport
        à l'origine des temps).
        """
        return np.exp(-self.rate * date)

    @timecached
    def getspot(self, date):
        "Renvoie le prix spot à la date `date`."
        return self.initvalue * self.simulate(date)

    @timecached
    def getfwd(self, date, maturity):
        """
        Renvoie le prix à la date `date` du prix forward de maturité
        `maturity`.
        """
        raise NotImplementedError("the Black-Scholes model is not a "\
                                  "forward model.")
