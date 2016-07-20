import unittest
from nodes import *
import numpy as np
from numpy.testing import *
from simulation import *
from asserts import *


class SimulationTests(unittest.TestCase):
    def testForwardScalar(self):
        context = SimulationContext()
        in1 = Connection(1)
        in2 = Connection(2)
        sum1 = context.sum(in1, in2)
        mul1 = context.multiply(sum1, Connection(3))
        div1 = context.div(mul1, Connection(9))
        context.forward()
        self.assertEqual(div1.value, 1)

    def testBackwardScalar(self):
        context = SimulationContext()
        in1 = Connection(1)
        in2 = Connection(2)
        sum1 = context.sum(in1, in2)
        mul1 = context.multiply(sum1, Connection(4))
        div1 = context.div(mul1, Connection(8))
        div1.gradient = 5
        context.forward()
        context.backward()
        self.assertEqual(in1.gradient, 2.5)

    def testForwardSoftmax(self):
        context = SimulationContext()
        x = Connection(np.array([-2.85, 0.86, 0.28])[:, np.newaxis])
        oneHotY = Connection(np.array([0., 0., 1.])[:, np.newaxis])
        exp1 = context.exp(x)
        reduce_sum1 = context.reduce_sum(exp1, axis=0)
        mul1 = context.multiply(exp1, oneHotY)
        reduce_sum2 = context.reduce_sum(mul1, axis=0)
        div1 = context.div(reduce_sum2, reduce_sum1)
        log1 = context.log(div1)
        mul2 = context.multiply(log1, Connection(-1))
        context.forward()
        assert_array_almost_equal(mul2.value, np.array([1.04]), 3)

    def testBackwardSoftmax(self):
        context = SimulationContext()
        x = Connection(np.array([-2.85, 0.86, 0.28]))
        oneHotY = Connection(np.array([0., 0., 1.]))
        exp1 = context.exp(x)
        reduce_sum1 = context.reduce_sum(exp1, axis=0)
        mul1 = context.multiply(exp1, oneHotY)
        reduce_sum2 = context.reduce_sum(mul1, axis=0)
        div1 = context.div(reduce_sum2, reduce_sum1)
        log1 = context.log(div1)
        mul2 = context.multiply(log1, Connection(-1))
        context.forward()
        context.backward()
        numerical_gradient = [
            numericalGradient(context, x, mul2, np.array([1., 0., 0.])),
            numericalGradient(context, x, mul2, np.array([0., 1., 0.])),
            numericalGradient(context, x, mul2, np.array([0., 0., 1.]))
        ]
        assert_array_almost_equal(x.gradient, numerical_gradient)
