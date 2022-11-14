import numpy as np

from swarmrl.observables.concentration_field import ConcentrationField
from swarmrl.observables.multi_sensing import MultiSensing
from swarmrl.observables.vision_cone import VisionCone


class TestVisionCone:
    @classmethod
    def setup_class(cls):
        """
        Set some initial attributes.
        """
        cls.vc = VisionCone(
            vision_angle=90,
            vision_range=1,
            return_cone=False,
            vision_direction=complex(0, 1),
            source=np.array([0.0, 0.0, 0.0]),
            detect_source=False,
            box_size=np.array([1.0, 1.0, 0.0]),
        )
        cls.gradfield = ConcentrationField(
            source=np.array([0.0, 0.0, 0.0]),
            decay_fn=cls.decay_fn,
            box_size=np.array([1.0, 1.0, 0.0]),
        )
        cls.multisensing = MultiSensing(observables=[cls.vc, cls.gradfield])

    def decay_fn(self, x):
        """
        Scaling function for the task
        """
        return 10.1 * np.exp(-5 * x) - 0.1

    def test_call(self):
        """
        Tests if multisensing works.
        """
        assert len(self.multisensing.observables) == 2
