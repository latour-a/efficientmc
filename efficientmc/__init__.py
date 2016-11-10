"Pricing d'actifs énergétiques par Monte Carlo."

import numpy as np
import pandas as pd
import scipy.stats as sps
import efficientmc.assets as assets
import efficientmc.pricemodels as pricemodels
import efficientmc.generators as generators
from collections import namedtuple

MCResults = namedtuple('MCResults', ('mean', 'iclow', 'icup'))

def runmc(*allassets):
    """
    Calcule par simulation Monte-Carlo les cash-flows des actifs de
    `allassets`, sous la probabilité risque-neutre.
    """
    allmarkets = set([m for asset in allassets for m in asset.getmarkets()])
    marketsdates = set([d for market in allmarkets for d in market.getdates()])
    assetsdates = set([d for asset in allassets for d in asset.getdates()])
    earnings = {asset.name: {} for asset in allassets}
    prices = {market.name: {} for market in allmarkets}
    volumes = {asset.name: {m.name: {} for m in asset.getmarkets()} \
               for asset in allassets}
    alldates = assetsdates.union(marketsdates)
    for date in alldates:
        #FIXME: utiliser une notion de `computation` pour sauvegarder et
        # aggréger les résultats au fur et à mesure.
        # Mise à jour des marchés :
        for market in allmarkets:
            market.simulate(date)
        # Mise à jour des objets :
        for asset in allassets:
            earnings[asset.name][date] = asset.get_discounted_cf(date)
            for market in asset.getmarkets():
                volumes[asset.name][market.name][date] = asset.getvolume(date,
                                                                         market)
                #FIXME: on peut avoir besoin d'autre chose que du spot.
                if date not in prices[market.name]:
                    prices[market.name][date] = market.getspot(date)
    earnings = pd.Panel(earnings).transpose(1, 0, 2)
    #FIXME: utiliser xarray.
    volumes = {key: pd.Panel(value).transpose(1, 0, 2)
               for key, value in volumes.items()}
    prices = pd.Panel(prices).transpose(1, 0, 2)
    return earnings, volumes, prices

def getmtm(cf, alpha=0.95):
    """
    Calcule la MtM (i.e. : les cash-flows moyens réalisés et un intervalle
    de confiance) de chaque actif listé dans `cf`.

    Paramètres
    ----------
    cf : pandas.Panel
        Cash-flows réalisés pour chaque simulation (`items`), chaque
        actif (`major_axis`) et chaque date (`minor_axis`).
    alpha : double compris entre 0. et 1.
        Quantile à utiliser pour le calcul des intervalles de confiances.
    """
    nsims = cf.shape[0]
    cumvalues = cf.sum(axis=2)
    mean = cumvalues.mean(axis=1)
    std = cumvalues.std(axis=1)
    res = {}
    for key, val in mean.items():
        res[key] = mkmcresults(val, std[key], nsims)
    return res

def getdelta(volumes, prices, alpha=0.95):
    """
    Calcule les deltas (avec intervalles de confiance) de chaque actif
    listé dans `cf`.

    Paramètres
    ----------
    cf : pandas.Panel
        Cash-flows réalisés pour chaque simulation (`items`), chaque
        actif (`major_axis`) et chaque date (`minor_axis`).
    volumes : dictionnaire de pandas.Panel
        Volumes exercés pour chaque actif (clef du dictionnaire), chaque
        simulation (`items`), chaque marché (`major_axis`) et chaque
        date (`minor_axis`).
    prices : pandas.Panel
        Prix réalisés pour chaque simulation (`items`), chaque marché
        (`major_axis`) et chaque date (`minor_axis`).
    alpha : double compris entre 0. et 1.
        Quantile à utiliser pour le calcul des intervalles de confiances.
    """
    nsims = prices.shape[0]
    res = {}
    for asset, volume in volumes.items():
        for market in volume.major_axis:
            #FIXME: laisser le choix du niveau d'agrégation du delta.
            price = prices.loc[:, market, :]
            initfwd = price.mean(axis=1)
            tmp = (volume.loc[:, market, :] * price).T / initfwd
            delta = (tmp * initfwd) / initfwd.sum()
            res[(asset, market)] = mkmcresults(delta.mean(), delta.std(), nsims)
    return res

def mkmcresults(mean, std, nsims, alpha=0.95):
    """
    Crée un objet `MCResults` contenant `mean` et l'intervalle de
    confiance associé.
    """
    coeff = sps.norm.ppf(0.5 + 0.5 * alpha)
    low = mean - coeff * std / np.sqrt(nsims)
    up = mean + coeff * std / np.sqrt(nsims)
    return MCResults(mean, low, up)
