"""

TODO: get a training dataset.

"""
import unittest
import numpy
import theano
from theano import tensor
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


class RNN(unittest.TestCase):
    """
    First exercises in training recurrent neural networks.
    """
    def setUp(self):
        """
        TODO: add biases.
        """
        B = 20
        L = 100
        N_in = 10
        N_hid = 50
        N_out = 2

        in_hid = srandn(N_in, N_hid)
        hid_hid = srandn(N_hid, N_hid)
        hid_out = srandn(N_hid, N_out)
        hid0 = srandn(N_hid)

        X = srandn(L, B, N_in)
        X_hid = tensor.dot(X.reshape((L*B, N_in)), in_hid).reshape((L,B, N_hid))
        
        hid_seqs, updates = theano.scan(
                fn=lambda ri, rii:tensor.tanh(ri + tensor.dot(rii, hid_hid)),
                sequences=[X_hid],
                outputs_info=hid0,
                mode = theano.Mode(linker='cvm'))
        assert not updates
        del updates

        out_seqs2 = tensor.dot(hid_seqs.reshape((L*B, N_hid)), hid_out)
        out_seqs = out_seqs2.reshape((L, B, N_out))

        err = tensor.mean((X_hid[0, :, :2] - out_seqs[L-1])**2)

        params = [in_hid, hid_hid, hid0]
        gparams = tensor.grad(err, params)

        lr = tensor.scalar()

        f = theano.function([lr], err,
                updates=[(p, p - lr * gp) 
                    for (p,gp) in zip(params, gparams)],
                mode = theano.Mode(linker='cvm'))
        self.f = f

    def test_bptt(self):
        """
        Implement training by gradient descent based on backprop through time.

        Suppose we want to classify a binary sequence by its first element, how long can the
        sequence be before BPTT does not work?
        """
        self.f(0.01)

    def test_esn(self):
        """
        Instead of BPTT training, implement an echo-state network.
        """


class RNN_Opt(unittest.TestCase):
    """
    Exploring alternate optimization strategies for recurrent neural networks.
    """

    def setUp(self):
        """
        Parametrize a rnn so that the parameters are all stored in a single vector, so that it
        is scipy-optimizer-compatible.
        """

    def test_bptt_cg(self):
        """
        Modify the bptt training function you used in the previous function to use one of the
        conjugate gradient optimization routines in scipy. What happens? Does it work better
        than steepest descent?
        """

    def test_bptt_bfgs(self):
        """
        Modify the function above to use the BFGS solver instead does it help?
        """

    def test_bptt_hf(self):
        """
        Bonus: modify the function above to use the Hessian-Free solver of Martens (see Martens
        and Sutskever, 2011 for details).
        """


