"""
Test module for loss callbacks.
"""
from swarmrl.callbacks import UniformCheckpointing


class TestLossCallbacks:
    """
    Test suite for the loss callbacks.
    """

    def test_uniform_checkpointing(self):
        """
        Test the uniform loss callback.
        """
        callback = UniformCheckpointing(save_interval=2, history=5)

        assert callback(0) == (False, None)
        assert callback(1) == (False, None)
        assert callback(2) == (True, None)
        assert callback(3) == (False, None)
        assert callback(4) == (True, None)
        assert callback(5) == (False, None)
        assert callback(6) == (True, None)
        assert callback(7) == (False, None)
        assert callback(8) == (True, None)
        assert callback(9) == (False, None)
        assert callback(10) == (True, 2)
