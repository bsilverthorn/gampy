import plac

if __name__ == "__main__":
    from backfit import main

    plac.call(main)

import numpy
import scikits.learn.neighbors
import scikits.learn.linear_model

class GeneralizedAdditive(object):
    def __init__(self, features, smoothers):
        assert len(features) == len(smoothers)

        self._features = features
        self._smoothers = smoothers

    def fit(self, x, y):
        S = len(self._smoothers)
        (N,) = y.shape

        self._a = numpy.mean(y)
        self._c = numpy.zeros(S)

        for i in xrange(32): # XXX
            for j in xrange(S):
                residuals = numpy.copy(y)

                for k in xrange(S):
                    if j != k and (i > 0 or k < j):
                        feature = self._features[k](x)
                        residuals -= self._smoothers[k].predict(feature) - self._c[k]

                feature = self._features[j](x)

                self._smoothers[j].fit(feature, residuals)

                self._c[j] = numpy.mean(self._smoothers[j].predict(feature))

    def predict(self, x):
        S = len(self._smoothers)
        (N, _) = x.shape

        y = numpy.ones(N) * self._a

        for i in xrange(S):
            feature = self._features[i](x)
            y += self._smoothers[i].predict(feature) - self._c[i]

        return y

def main():
    x = numpy.r_[-4.0:4.0:256j][..., None]
    y = numpy.sin(x[:, 0])

    gam = \
        GeneralizedAdditive(
            [
                lambda x: x,
                #lambda x: x,
                ],
            [
                #scikits.learn.linear_model.LinearRegression(),
                scikits.learn.neighbors.NeighborsRegressor(),
                ],
            )

    gam.fit(x, y)

    predicted = gam.predict(x)

    print "x,y,p"
    for i in xrange(y.shape[0]):
        print ",".join(map(str, [x[i, 0], y[i], predicted[i]]))

