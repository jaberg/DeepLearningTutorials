import unittest
import numpy
import theano
import theano.tensor as T
show_plots = True
if show_plots:
    import matplotlib.pyplot as plt

import theano.tensor.shared_randomstreams
RandomStreams = theano.tensor.shared_randomstreams.RandomStreams

numpy.random.seed(234)

def sharedX(x, name=None):
    """
    Return a theano shared variable for x, after casting it to dtype theano.config.floatX.
    """
    return theano.shared(
            numpy.asarray(x, dtype=theano.config.floatX),
            name=name)


def srandn(*size):
    return sharedX(numpy.random.randn(*size))


class IndexFromCounts(theano.Op):
    """
    A weird Op that you need for resampling with replacement.
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, counts):
        # Return an Apply of this Op (self).
        # Do error checking, and create a (typed) output variable.

        # converts variable / ndarray / list / tuple to symbolic ndarray
        counts = theano.tensor.as_tensor_variable(counts)

        # type check: this operations is for vectors only
        if counts.ndim != 1:
            raise TypeError(counts)
        
        # type check: we'll need some kind of integeter
        if str(counts.dtype)[:3] not in ('int', 'uin'):
            raise TypeError(counts)

        # Allocate an output with the same type as `counts`
        return theano.gof.Apply(self, [counts], [counts.type()])

    def perform(self, node, inputs, outcontainers):
        counts_value, = inputs
        z_container, = outcontainers
        l = []
        for i, c in enumerate(inputs[0]):
            l += [i]*c
        z_container[0] = numpy.asarray(l)
index_from_counts = IndexFromCounts()


class ParticleFilter(unittest.TestCase):
    """
    """

    def dynamics(self, x, t):
        return .5 * x + 25 * x / (1 + x**2) + 8 * T.cos(0.2 * t)

    def observation(self, x):
        return x**2 / 20

    def resample(self, particles, weights):
        counts = self.s_rng.multinomial(
                pvals=weights,
                n=self.n_particles,
                size=())
        index = index_from_counts(counts)
        return particles[index]
        
    def setUp(self):
        """
        Configure a particle-filtering problem.
        """
        n_particles = self.n_particles = 50

        # set up the stochastic dynamical system that we're going to track
        self.s_rng = RandomStreams(23424)
        v = self.s_rng.normal()  # for system
        pv = self.s_rng.normal(size=(self.n_particles,))
        w = self.s_rng.normal()  # for observation

        t = sharedX(0, name='t')

        x = sharedX(0, name='x')

        x_next = self.dynamics(x, t) + v
        y = self.observation(x_next) + w # TODO: should this be x?

        print 'n_particles:', n_particles
        print 'system:', theano.printing.pprint(x_next)
        
        # allocate particles
        particles = sharedX(numpy.zeros(self.n_particles))

        # filtering equations
        particles_next = self.dynamics(particles, t) + pv
        particles_y = particles_next**2 / 20
        weights_unnorm = T.exp(-0.5 * (particles_y - y)**2)
        weights = weights_unnorm / weights_unnorm.sum()

        self.system_f = theano.function([],
                outputs = [x_next, particles_next, weights, pv],
                updates = {
                    t: t + 1,
                    x: x_next,
                    particles: self.resample(particles_next, weights)})

    def test_basic_tracking(self):
        """
        Writeme
        """
        x, p = [], []
        n_steps = 50
        for i in xrange(n_steps):
            xi, pi, wi, pvi = self.system_f()
            x.append(xi)
            p.append(pi)
            #print xi, pi
            #print wi
            #print pvi

        x = numpy.asarray(x)
        p = numpy.asarray(p)

        if show_plots:
            plt.plot(range(n_steps), x, label='state')
            for i in range(self.n_particles):
                plt.scatter(range(n_steps), p[:,i], c='r', label=None if i else 'belief')
            plt.xlabel('time')
            plt.ylabel('position')
            plt.legend()
            plt.show()

    def test_resampling_heuristics(self):
        """
        Writeme
        """

    def test_number_of_particles(self):
        """
        Writeme
        """

    def test_learning_by_em_online(self):
        """
        """

    def test_learning_bayesian(self):
        """
        """

    def test_learning_by_bandits(self):
        """
        """

    def test_learning_by_other(self):
        """
        """


class CIndexFromCounts(theano.Op):

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, counts):
        return theano.gof.Apply(self, [counts], [counts.type()])

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, inp, out, sub):
        print "COMPILING!"
        counts, = inp   # counts: a string
        z, = out        # z: a string
        fail = sub['fail']
        return """
        npy_intp n_counts =  %(counts)s->dimensions[0];
        dtype_%(counts)s counts_sum = 0;
        dtype_%(counts)s * counts_data =
            (dtype_%(counts)s*)(%(counts)s->data);

        for (int i = 0; i < n_counts; ++i)
        {
            counts_sum += counts_data[i];
        }
        Py_XDECREF(%(z)s);
        %(z)s = (PyArrayObject*)PyArray_SimpleNew(
            1, &n_counts, PyArray_TYPE(%(counts)s));
        dtype_%(z)s * z_data = (dtype_%(z)s*)(%(z)s->data);
        for (int i = 0; i < n_counts; ++i)
        {
            for (int j = 0; j < counts_data[i]; ++j)
            {
                *z_data = i;
                ++z_data;
            }
        }
        """ % locals()
c_index_from_counts = CIndexFromCounts()

@theano.tensor.opt.register_specialize
@theano.gof.local_optimizer([index_from_counts])
def sub_cindex_from_counts(node):
    if node.op == index_from_counts:
        return [c_index_from_counts(*node.inputs)]

