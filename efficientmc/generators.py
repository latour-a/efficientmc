import numpy as np
from efficientmc.utils import timecached, DateCache

class GaussianGenerator:
    "Générateur de bruits gaussiens corrélés."

    def __init__(self, nsims, corrmatrix, corrkeys, randomfunc):
        """
        Initialise une nouvelle instance de la classe `GaussianGenerator`.

        Paramètres :
        ------------
        nsims : entier positif
            Nombre de simulations à générer.
        corrmatrix : matrice carrée
            Matrice de corrélation entre les différents bruits gaussiens
            à simuler. La matrice doit être définie positive.
        corrkeys
            Liste des identifiants associés aux différentes lignes / colonnes
            de la matrice de corrélation `corrmatrix`. Les identifiants doivent
            être donnés dans l'ordre dans lequel les bruits correspondants
            apparaissent dans `corrmatrix`.
        randomfunc
            Fonction permettant de générer des bruits gaussiens indépendants
            (typiquement `np.random.randn`).
        """
        self.corrmatrix = corrmatrix
        self.corrkeys = corrkeys
        self.nsims = nsims
        self.randomfunc = randomfunc
        self.cache = DateCache()
        try:
            np.linalg.cholesky(self.corrmatrix)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("the correlation matrix is not "\
                                        "positive definite.") from None

    @property
    def nnoises(self):
        "Nombre de bruits gaussiens distincts à simuler."
        return self.corrmatrix.shape[0]

    @timecached
    def getallnoises(self, date):
        """
        Renvoie `self.nsims` réalisations de `self.nnoises` bruits
        gaussiens corrélés.
        """
        whitenoises = self.randomfunc(self.nnoises, self.nsims)
        noises = np.dot(np.linalg.cholesky(self.corrmatrix), whitenoises)
        return noises

    @timecached
    def getnoises(self, date, keys):
        """
        Renvoie un tableau de taille `(len(keys), self.nsims)` de bruits
        gaussiens corrélés correspondants aux aléas identifiés par les
        clefs `keys`.
        """
        noises = self.getallnoises(date)
        res = np.empty((len(keys), self.nsims))
        for idx, key in enumerate(keys):
            keyidx = self.corrkeys.index(key)
            res[idx, :] = noises[keyidx, :]
        return res
