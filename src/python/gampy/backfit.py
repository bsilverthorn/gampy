"""@author: Bryan Silverthorn <bcs@cargo-cult.org>"""

import plac

if __name__ == "__main__":
    from backfit import main

    plac.call(main)

import numpy
import scipy.stats
import scipy.interpolate
import scikits.learn.neighbors
import scikits.learn.linear_model

class UnivariateSplineSmoother(object):
    def __init__(self, k = 3):
        self._k = k

    def fit(self, x, y):
        x = numpy.asarray(x)
        y = numpy.asarray(y)

        assert x.ndim == 2
        assert x.shape[1] == 1

        self._spline = scipy.interpolate.UnivariateSpline(x[:, 0], y, k = self._k)

    def predict(self, x):
        x = numpy.asarray(x)

        assert x.ndim == 2
        assert x.shape[1] == 1

        return self._spline(x[:, 0])

class BivariateSplineSmoother(object):
    def fit(self, x, y):
        x = numpy.asarray(x)
        y = numpy.asarray(y)

        assert x.ndim == 2
        assert x.shape[1] == 2

        self._spline = scipy.interpolate.SmoothBivariateSpline(x[:, 0], x[:, 1], y)

    def predict(self, x):
        x = numpy.asarray(x)

        assert x.ndim == 2
        assert x.shape[1] == 2

        return self._spline(x[:, 0], x[:, 1]) # XXX broken

def logit(x):
    return 1.0 / (1.0 + numpy.exp(-x))

def inverse_logit(p):
    return numpy.log(p) - numpy.log(1.0 - p)

class GeneralizedAdditive(object):
    def __init__(self, smoothers, features = None):
        self._smoothers = smoothers
        self._features = features

        if features is not None:
            assert len(features) == len(smoothers)

    def fit(self, x, y):
        x = numpy.asarray(x)
        y = numpy.asarray(y)

        S = len(self._smoothers)
        (N,) = y.shape

        self._a = numpy.mean(y)
        self._c = numpy.zeros(S)

        for i in xrange(32): # XXX need a convergence condition
            print "starting iteration", i, "of backfitting"

            for j in xrange(S):
                residuals = numpy.array(y, dtype = float)

                for k in xrange(S):
                    if j != k and (i > 0 or k < j):
                        feature = self._get_feature(k, x)
                        residuals -= self._smoothers[k].predict(feature) - self._c[k]

                feature = self._get_feature(j, x)

                self._smoothers[j].fit(feature, residuals)

                self._c[j] = numpy.mean(self._smoothers[j].predict(feature))

    def predict(self, x):
        x = numpy.asarray(x)

        S = len(self._smoothers)
        (N, _) = x.shape

        y = numpy.ones(N) * self._a

        for i in xrange(S):
            feature = self._get_feature(i, x)

            y += self._smoothers[i].predict(feature) - self._c[i]

        return y

    def _get_feature(self, k, x):
        if self._features is None:
            return x[:, k:k + 1]
        else:
            return self._features[k](x)

class LogisticAdditive(GeneralizedAdditive):
    def fit(self, x, y):
        y = numpy.array(y, float)

        indices0 = y == 0.0
        indices1 = y == 1.0

        y[indices0] = -numpy.inf
        y[indices1] = numpy.inf
        y[~(indices0 | indices1)] = inverse_logit(y[~(indices0 | indices1)])

        print y

        GeneralizedAdditive.fit(self, x, y)

    def predict_proba(self, x):
        p = logit(self.predict(x))

        return numpy.array([1.0 - p, p])

def main():
    N = 32
    x = []
    y = []

    xs = numpy.r_[-10.0:10.0:N + 0j]
    ys0 = scipy.stats.distributions.norm.pdf(xs, loc = 2.0, scale = 1.0)
    ys1 = scipy.stats.distributions.norm.pdf(xs, loc = 2.0, scale = 1.0)

    for i in xrange(N):
        for j in xrange(N):
            x.append([xs[i], xs[j]])
            y.append(ys0[i] * ys1[j])

    x = numpy.array(x)
    y = numpy.array(y)

    gam = \
        GeneralizedAdditive(
            [
                #scikits.learn.linear_model.LinearRegression(),
                scikits.learn.neighbors.NeighborsRegressor(),
                scikits.learn.neighbors.NeighborsRegressor(),
                #BivariateSplineSmoother(),
                ],
            )

    gam.fit(x, y)

    predicted = gam.predict(x)

    print "x0,x1,y,p,residual"
    for i in xrange(y.shape[0]):
        print ",".join(map(str, [x[i, 0], x[i, 1], y[i], predicted[i], y[i] - predicted[i]]))

