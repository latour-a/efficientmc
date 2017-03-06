import numpy as np
from efficientmc.utils import timecached, DateCache
import sobol_seq
import ghalton as gh
from scipy.stats import norm
from math import ceil, fmod, floor, log
from pyDOE import *

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

"Variables antithétiques"

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

"Quasi-Monte Carlo"

"Suite de Van der Corput"

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
    for i in range(nsims):
        array[i]=vdc(i+1,b)
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

"Suite de Halton"

def halton2(dim, nsims):
    """
    la fonction ne crash plus.
    Version 2 de la suite d'halton sans la librairie Python existante.
    """
    h = np.empty(nsims * dim)
    h.fill(np.nan)
    p = np.empty(nsims)
    p.fill(np.nan)
    Base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognsims = log(nsims + 1)
    for i in range(dim):
        base = Base[i]
        n = int(ceil(lognsims / log(base)))
        for t in range(n):
            p[t] = pow(base, -(t + 1) )
        for j in range(nsims):
            d = j + 1
            somme = fmod(d, base) * p[0]
            for t in range(1, n):
                d = floor(d / base)
                somme += fmod(d, base) * p[t]

            h[j*dim + i] = somme

    return norm.ppf(h.reshape(dim, nsims))

def haltonF(nnoises,nsims):
    Prime=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,\
            59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,\
             127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,\
              191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,\
               257, 263, 269, 271, 277, 281]
    halton=np.empty((nnoises,nsims))
    for i in range(0,nsims,1):
        for j in range(0,nnoises,1):
            prime=Prime[j]
            halton[j,i]=vdc(i,prime)
    return halton

"Suite de Hammersley"

def hammersley(nnoises,nsims):
    Prime=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,\
            59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,\
             127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,\
              191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,\
               257, 263, 269, 271, 277, 281]
    hammersley=np.empty((nnoises,nsims))
    for i in range(0,nsims,1):
        hammersley[0,i]=i/nsims
    for i in range(0,nsims,1):
        for j in range(1,nnoises-1,1):
            prime=Prime[j]
            hammersley[j,i]=vdc(i,prime)
    return hammersley

"Suite de Faure"

def toDigits2(n, b):
    """Convert a positive number n to its digit representation in base b."""
    digits = []
    while n > 0:
        digits.insert(0, n % b)
        n  = n // b
    digits.reverse()
    i=0
    for i in range(0,len(digits),1):
        digits[i]/=b**(i+1)
    return digits

def toDigits3(n, b,dim):
    if n!=0:
        I=int(math.log(n)/math.log(b))+1
    else:
        I=0
    z=[]
    a=toDigits2(n,b)
    if dim==1:
        return a
    else:
        for i in range(0,I-1,1):
            h=0
            for j in range(i,I-1,1):
                h+=(comb(i,j)*toDigits3(n,b,dim-1)[i])%b
            z.append(h)
        return z

def sumDigits(n,b):
        a=toDigits2(n,b)
        return np.sum(a)

def faureF(dim,nsims):
    array=np.empty((dim,nsims))
    p=2
    Prime=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,\
            59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,\
             127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,\
              191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,\
               257, 263, 269, 271, 277, 281]
    i=0
    for i in range(0,nsims,1):
        array[0,i]=sumDigits(i,2)
    if dim==1:
        "Si la dimension est égale à 1, on prend 2 comme base"
        return array
    else:
        i=0
        while p<dim:
            p=Prime[i+1]
        j=0
        s=1
        for s in range(1,dim,1):
            for j in range(0,nsims,1):
                array[s,j]=np.sum(toDigits3(j,p,s+1))
        return norm.ppf(array)

"Suite de Sobol"

def sobolF(nnoises,nsims):
    """
    Renvoie un tableau de valeurs générés par la suite de Sobol
    de taille (nnoises,nsims)
    """
    noises = np.empty((nnoises,nsims))
    #  Utilisation de la fonction sobol_seq.i4_sobol_generate_std_normal
    # pour générer des variables quasi-aléatoires suivant une loi normale.
    noises = sobol_seq.i4_sobol_generate_std_normal(nnoises, nsims)
    noises=np.transpose(noises)
    return noises

"Stratification"

def stratified_sampling1(M,nsims):
    array=np.empty(nsims)
    array=lh.sample(M,nsims)
    array=norm.ppf(array)
    return array.reshape(M,nsims)

def stratified_sampling2(nnoises,nsims):
    """On divise l'intervalle [0,1] en plusieurs stratas, m stratas"""
    noises = np.empty((nnoises,nsims))
    m=int(nnoises/nsims)
    i=0
    """Pour chaque strata, on prend l échantillons suivant la loi uniforme
    tel que nsims=m*l et enfin on prend le pdf du cdf."""
    for i in range(m+1,1):
        noises[i,:]=norm.ppf(np.random.uniform(i/m,(i+1)/m,nsims))
    return noises

def stratified_sampling3(l,nsims):
    """On divisie l'intervalle [0,1] en plusieurs stratas, m stratas"""
    """m=500"""
    m=5000
    l=nsims/m
    noises=np.empty((l,nsims))
    for i in range(0,m,1):
        noises[:,i]=norm.ppf(np.random.uniform(i/m,(i+1)/m,l))
    return noises

def stratified_sampling4(M,nsims):
    L=nsims/M
    i=0
    noises=np.empty((M,nsims))
    for i in range(M):
        noises[i,:]=norm.ppf(np.random.uniform(i/M,(i+1)/M,nsims))
    return noises

def stratified_samplingF(dim,nsims):
    "Latin Hypercube Sample, une forme efficient de stratification à plusieurs dimensions"
    lhd=np.empty((dim,nsims))
    lhd = lhs(dim, samples=nsims)
    lhd=norm.ppf(lhd)
    lhd=np.transpose(lhd)
    return lhd