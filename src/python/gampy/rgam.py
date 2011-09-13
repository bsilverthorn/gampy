"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import numpy
import rpy2.robjects
import rpy2.robjects.packages
import rpy2.robjects.numpy2ri

def ilogit(x):
    return 1.0 / (1.0 + numpy.exp(-x))

class RGAM(object):
    def __init__(self, X, Y):
        # rescale the inputs
        X = numpy.array(X)

        self._inputs_mean = numpy.mean(X, axis = 0)

        X -= self._inputs_mean

        self._inputs_std = numpy.std(X, axis = 0)
        self._inputs_std[self._inputs_std <= numpy.finfo(float).eps] = 1.0

        X /= self._inputs_std

        # build our data frame
        columns = {"label": rpy2.robjects.FloatVector(Y)}
        self._feature_dimensions = set()
        self._feature_names = set()

        for d in xrange(X.shape[1]):
            column_data = X[:, d]

            if len(numpy.unique(column_data)) >= 4:
                name = "d{0}".format(d)

                columns[name] = rpy2.robjects.FloatVector(column_data)

                self._feature_names.add(name)
                self._feature_dimensions.add(d)

        data = rpy2.robjects.DataFrame(columns)

        # fit our GAM
        self._gam = rpy2.robjects.packages.importr("gam")
        self._formula = rpy2.robjects.Formula("label ~ " + " + ".join(map("s({0})".format, self._feature_names)))

        for name in data.colnames:
            self._formula.environment[name] = data.rx2(name)

        self._model = self._gam.gam(self._formula, family = rpy2.robjects.r.binomial, data = data)

    def predict_proba(self, X):
        # build our data frame
        scaled = (numpy.asarray(X) - self._inputs_mean) / self._inputs_std
        columns = {}

        for d in xrange(scaled.shape[1]):
            if d in self._feature_dimensions:
                columns["d{0}".format(d)] = rpy2.robjects.FloatVector(scaled[:, d])

        data = rpy2.robjects.DataFrame(columns)

        # generate predictions
        v = rpy2.robjects.r.predict(self._model, newdata = data)
        p = ilogit(numpy.asarray(v))

        return numpy.array([1 - p, p]).T

