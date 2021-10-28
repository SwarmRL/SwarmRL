import context

from swarmrl.engine import espresso
import unittest as ut

class EspressoTest(ut.TestCase):
    def test_class(self):
        """
        Sadly, espresso systems are global so we have to do all tests on only one object
        """
