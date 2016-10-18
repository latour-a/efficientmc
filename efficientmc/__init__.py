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
    res = {asset.name: {} for asset in allassets}
    allmarkets = set([m for asset in allassets for m in asset.getmarkets()])
    marketsdates = set([d for market in allmarkets for d in market.getdates()])
    assetsdates = set([d for asset in allassets for d in asset.getdates()])
    alldates = assetsdates.union(marketsdates)
    for date in alldates:
        #FIXME: utiliser une notion de `computation` pour sauvegarder et
        # aggréger les résultats au fur et à mesure.
        # Mise à jour des marchés :
        for market in allmarkets:
            market.simulate(date)
        # Mise à jour des objets :
        for asset in allassets:
            val = asset.get_discounted_cf(date)
            res[asset.name][date] = val
    return pd.Panel(res).transpose(1, 0, 2)

def postpro(sims, alpha=0.95):
    """
    Calcule la moyenne des cash-flows (et l'intervalle de confiance associé)
    pour chaque actif simulé dans `sims`.
    """
    nsims = sims.shape[0]
    cumvalues = sims.sum(axis=2)
    mean = cumvalues.mean(axis=1)
    std = cumvalues.std(axis=1)
    coeff = sps.norm.ppf(0.5 + 0.5 * alpha)
    res = {}
    for key, val in mean.items():
        low = val - coeff * std[key] / np.sqrt(nsims)
        up = val + coeff * std[key] / np.sqrt(nsims)
        res[key] = MCResults(val, low, up)
    return res
