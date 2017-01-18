import numpy as np
from efficientmc.utils import timecached, DateCache
import sobol_seq
from scipy.stats import norm

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

def antithetic_randn(nnoises, nsims):
    """
    Renvoie un tableau de bruits gaussiens non corrélés :math:`(G_{_i,j})`
    de taille `(nnoises, nsims)`, et tel que, pour tout :math:`i` :
        :math:`\forall 1 \leq j \leq n / 2, G_{i,n/2+j} = -G_{i,j}`

    Paramètres :
    ------------
    nnoises : entier positif
        Nombre de bruits à simuler.
    nsims : entier positif, pair
        Nombre de simulations à effectuer par prix.
    """
    if nsims % 2 != 0:
        raise ValueError("the number of simulations used with antithetic "\
                         "variables should be even.")
    half = int(0.5 * nsims)
    noises = np.empty((nnoises,nsims))
    noises[:, :half] = np.random.randn(nnoises, half)
    noises[:, half:] = -noises[:, :half]
    return noises

def vdc(n, base):
    """
    Cette fonction permet de calculer le n-ieme nombre de la base b de la 
    séquence de Van Der Corput
    """
    vdc, denom = 0,1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return norm.ppf(vdc)
def van_der_corput(nsims,b):
    """
    Cette fonction permet de générer la séquence de Van Der Corput en base b
    """
    array=np.empty(nsims)
    i=0
    for i in range(nsims):
        array[i]=vdc(i,b)
    return array

def van_der_corput_dimension(dim,nsims):
    array=np.empty((dim,nsims))
    """
    Cette fonction génère dans un tableau de taille (dim,nsims) toutes les séquences
    de la suite de Van der Corput de la base 2 à la base dim+2
    """
    for i in range(2,dim+2,1):
        array[i-2,:]=van_der_corput(nsims, i)
    return array

def sobol(nnoises,nsims):
    """
    Renvoie un tableau de valeurs générés par la suite de Sobol
    de taille (nnoises,nsims)
    """
    noises = np.empty((nsims))
    #  Utilisation de la fonction sobol_seq.i4_sobol_generate_std_normal
    # pour générer des variables quasi-aléatoires suivant une loi normale.
    noises = sobol_seq.i4_sobol_generate_std_normal(nnoises, nsims)
    return noises.reshape(nnoises, nsims)
