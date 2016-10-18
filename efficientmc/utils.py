from functools import wraps

def timecached(func):
    """
    Décorateur permettant d'associer un cache à la fonction `func` :
    - la première fois que `func` est appelée, le résultat qu'elle renvoie
      est stocké dans le cache ;
    - par la suite, lorsque `func` est appelée de nouveau, c'est le résultat
      stocké dans le cache qui est renvoyé, sans qu'il soit nécessaire de
      recalculer `func`.
    Le cache est indexé sur les dates ; par conséquent, le premier argument
    de `func` doit représenter une date.

    Par ailleurs, `func` doit être une méthode d'un objet disposant d'un
    membre `cache` implémentant le cache en lui-même.
    """
    @wraps(func) # Pour préserver la docstring et le nom de `func`.
    def wrapper(self, date, *args):
        currdate = self.cache.currentdate
        if currdate is not None and date != currdate:
            self.cache.flush() # Nettoyage du cache
        key = self.cache.makekey(func, date, *args)
        if key not in self.cache.keys:
            res = func(self, date, *args)
            self.cache.register(date, key, res)
        return self.cache.get(key)
    return wrapper

class DateCache:
    "Cache indexé sur le temps."

    def __init__(self):
        """
        Initialise une nouvelle instance de la classe `DateCache`.
        """
        #FIXME: implémenter les méthodes nécessaires pour que le cache puisse
        # être utilisé exactement comme un dictionnaire ?
        self._prevdata = {}
        self._currdata = {}
        self.currentdate = None

    @property
    def keys(self):
        "Ensemble des clefs enregistrées dans le cache."
        return self._currdata.keys()

    def makekey(self, func, date, *args):
        """
        Renvoie la clef utilisée pour stocker le résultat du calcul
        `func(date, *args)`.
        """
        return (func.__name__,) + args

    def register(self, date, key, value):
        """
        Enregistre une valeur dans le cache.

        Paramètres :
        ------------
        date
            Date pour laquelle stocker `value`.
        key
            Clef associée à `value`.
        value
            Valeur à stocker.
        """
        if self.currentdate is None:
            self.currentdate = date
        elif self.currentdate != date:
            raise ValueError("DateCache is meant to store data "\
                             "for only one date at at time.")
        self._currdata[key] = value

    def get(self, key):
        "Renvoie la valeur associée à la clef `key`."
        return self._currdata[key]

    def getprev(self, funcname, *args):
        """
        Renvoie la dernière valeur sauvegardée dans le cache pour la
        méthode correspondant à `funcname` avec les arguments `*args`.
        """
        #FIXME: le `getprev` n'est pas très naturel...
        key = (funcname,) + args
        if key in self._prevdata.keys():
            return self._prevdata[key]
        raise KeyError("%s not in cache." % str(key))

    def flush(self):
        "Nettoie le cache."
        for key, value in self._currdata.items():
            # On écrase une éventuelle valeur existante.
            self._prevdata[key] = (self.currentdate, value)
        self._currdata = {}
        self.currentdate = None
