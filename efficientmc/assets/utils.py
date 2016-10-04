import numpy as np

def get_BlackScholes_d1(date, maturity, price, strike, model):
    """
    Calcule le terme :math:`d_1` intervenant dans la formule de
    Black-Scholes.

    Paramètres
    -----------
    date
        Date à laquelle calculer :math:`d_1`.
    maturity
        Maturité de l'option.
    price : double ou np.array
        Valeur du sous-jacent à la date `date`.
    strike : double
        Strike de l'option.
    model
        Modèle de prix utilisé pour le sous-jacent.
    """
    ttm = maturity - date
    if ttm > 0.:
        num = np.log(price / strike) + (0.5 * model.sigma**2 + model.rate) * ttm
        den = model.sigma * np.sqrt(ttm)
        d1 = num / den
    else:
        d1 = -np.inf
    return d1
