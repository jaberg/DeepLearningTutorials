"""
Learn about Theano & linear regression!

Set show_plots = True in the global variables above to see a plot appear.

Run each exercise test function individually with nosetests. For example, in bash, run

    nosetests -sd linear_regression.py:LinearRegression1D.test_fit_by_fixed_lr

to run just the first example from the commandline.
"""

import unittest
import numpy
import theano
show_plots = True
if show_plots:
    import matplotlib.pyplot as plt

numpy.random.seed(234)

def sharedX(x, name=None):
    """
    Return a theano shared variable for x, after casting it to dtype theano.config.floatX.
    """
    return theano.shared(
            numpy.asarray(x, dtype=theano.config.floatX),
            name=name)


def srandn(*size):
    return sharedX(
            numpy.random.randn(*size))


class LinearRegression1D(unittest.TestCase):
    """
    Exercises in 1-dimensional linear regression.
    """

    def setUp(self):
        n_examples = 10

        #TODO: add a test set and plot it as well
        
        self.X = srandn(n_examples)
        self.Y = sharedX(1 + 2.5 * self.X.get_value() + numpy.random.randn(n_examples))
        self.w = sharedX(0)
        self.b = sharedX(0)
        self.Yhat = self.X * self.w + self.b
        self.err = self.Yhat - self.Y
        self.mse = theano.tensor.mean(0.5 * self.err**2)
        self.gw, self.gb = theano.tensor.grad(self.mse, [self.w, self.b])

    def test_fixed_lr(self):
        """
        Fit a linear regression model by a fixed learning rate batch steepest descent
        algorithm.
        """

        lr = theano.tensor.scalar()
        fit_step = theano.function([lr], [self.mse, self.Yhat], updates={
                    self.w: self.w - lr * self.gw,
                    self.b: self.b - lr * self.gb})

        for i in xrange(100):
            mse, yhat =  fit_step(1.0)
            print i, mse

        if show_plots:
            plt.scatter(self.X.get_value(), self.Y.get_value(), label='data')
            plt.plot(self.X.get_value(), yhat, c='r', label='fit')
            plt.legend()
            plt.show()


    def test_fixed_lr_step_too_large(self):
        """
        test_fixed_lr converges somewhat slowly. You can make it converge more quickly by
        raising the learning rate (aka step size). There's a limit to how high you can make the
        step size though. What is it?  What happens when the step size is too big?

        Fill in this function body with an illustration of how things go wrong when the step
        size is too big.
        """
        raise NotImplementedError()

    def test_annealed_lr(self):
        """
        Standard (and recommended) practice is not to pick just one learning rate and run to
        convergence, but rather to anneal the learning rate with time.  Implement an annealing
        schedule for the learning rate.  Show that you can converge to the same minimum we saw
        in test_fixed_lr much more quickly (in XXX steps).
        """

        lr = theano.tensor.scalar()
        fit_step = theano.function([lr], [self.mse, self.Yhat], updates={
                    self.w: self.w - lr * self.gw,
                    self.b: self.b - lr * self.gb})

        for i in xrange(100):
            mse, yhat =  fit_step(1.0 / (.1 * i + 1))
            print i, mse

        if show_plots:
            plt.scatter(self.X.get_value(), self.Y.get_value(), label='data')
            plt.plot(self.X.get_value(), yhat, c='r', label='fit')
            plt.legend()
            plt.show()


class RegularizedLinearRegression1D(unittest.TestCase):
    """
    Exercises in regularized 1-dimensional linear regression.

    """

    def setUp(self):
        """
        """
        # Set up the basic cost function, but don't calculate the gradients.
        raise NotImplementedError()


    def test_l1_regularization(self):
        """
        Show the effect of L1 regularization in our simple 1D problem.
        """
        raise NotImplementedError()

    def test_l2_regularization(self):
        """
        Show the effect of L2 regularization in our simple 1D problem.
        """
        raise NotImplementedError()

    def test_l1_and_l2_regularization(self):
        """
        L1 and L2 regularization can be combined.

        Is there an application that you would obviously want to do this?
        """
        raise NotImplementedError()


class LinearRegressionNd(unittest.TestCase):
    """
    """
    # How to visualize result?


class Optimization(unittest.TestCase):
    """
    In this suite of exercises, take the Linear Regression example from LinearRegressionNd
    and compare some more sophisticated optimization methods than the simple one we've been
    using until now.

    How to set things up so that we can see the various effects of
    - training set size
    - unregularized vs. l1 vs. l2 regularization
    - batch vs. online
    """
    def setUp(self):
        # pack weights and bias into one vector-shaped shared variable so that we can use scipy
        # solvers on that vector.
        pass

    def test_cg(self):
        # 
        raise NotImplementedError()

    def test_lbfgs(self):
        #
        raise NotImplementedError()


class BayesianLinearRegression(unittest.TestCase):
    """
    We can switch from a point estimate of weights w to a Gaussian prior over weights w and
    still carry out a computationally efficient (i.e. tractable) inference procedure.

    :note: see Danny Tarlow's python implementation
    http://blog.smellthedata.com/2009/06/really-bayesian-logistic-regression-in.html
    """


