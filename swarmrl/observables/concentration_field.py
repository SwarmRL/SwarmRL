"""
Historical position observable computer.

Notes
-----
Observable for sensing changes in some field value, or, the gradient.
"""
import logging
from abc import ABC
from typing import List

import jax.numpy as np
import numpy as onp

from swarmrl.models.interaction_model import Colloid
from swarmrl.observables.observable import Observable

from scipy.special import kv  # modivied bessel function of second kind of real order v
from scipy.special import kvp  # derivative of modified bessel Function of second kind of real order

logger = logging.getLogger(__name__)


class ConcentrationField(Observable, ABC):
    """
    Position in box observable.

    Attributes
    ----------
    historic_positions : dict
            A dictionary of past positions of the colloid to be used in the gradient
            computation.
    """

    def __init__(self, source: np.ndarray, decay_fn: callable, box_length: np.ndarray):
        """
        Constructor for the observable.

        Parameters
        ----------
        source : np.ndarray
                Source of the field.
        decay_fn : callable
                Decay function of the field.
        box_size : np.ndarray
                Array for scaling of the distances.
        """
        self.source = source / box_length
        self.decay_fn = decay_fn
        self.historic_positions = {}
        self.box_length = box_length
        self._observable_shape = (3,)

    def initialize(self, colloids: List[Colloid]):
        """
        Initialize the observable with starting positions of the colloids.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids with which to initialize the observable.

        Returns
        -------
        Updates the class state.
        """
        for item in colloids:
            index = onp.copy(item.id)
            position = onp.copy(item.pos) / self.box_length
            self.historic_positions[str(index)] = position

    def compute_observable(self, colloid: Colloid, other_colloids: List[Colloid]):
        """
        Compute the position of the colloid.

        Parameters
        ----------
        colloid : object
                Colloid for which the observable should be computed.
        other_colloids
                Other colloids in the system.

        Returns
        -------
        delta_field : float
                Current field value minus to previous field value.
        """
        if self.historic_positions == {}:
            msg = (
                f"{type(self).__name__} requires initialization. Please set the "
                "initialize attribute of the gym to true and try again."
            )
            raise ValueError(msg)
        position = onp.copy(colloid.pos) / self.box_length
        index = onp.copy(colloid.id)
        previous_position = self.historic_positions[str(index)]

        # Update historic position.
        self.historic_positions[str(index)] = position

        current_distance = np.linalg.norm((self.source - position))
        historic_distance = np.linalg.norm(self.source - previous_position)

        # TODO: make this a real thing and not some arbitrary parameter.
        return 10000 * np.array(
            [self.decay_fn(current_distance) - self.decay_fn(historic_distance)]
        )

def calc_schmell(schmellpos, colpos, Dcoltrans=670):  # glucose in water in mum^2/s
    # result of Diffusion differential equation in 2D yielding a concentration field
    '''
    #Test script
    from scipy.special import kv  # modivied bessel function of second kind of real order v
    from scipy.special import kvp  # derivative of modified bessel Function of second kind of real order
    import numpy as np
    import matplotlib.pyplot as plt
    rodthickness = 5  # micrometer
    B = np.linspace(0.1, 10, 100)
    D = 0.0014  # translative Diffusion coefficient
    const = -2 * np.pi * D * rodthickness / 2 * B * kvp(0, B * rodthickness / 2)
    plt.plot(B, const)
    plt.show()

    # => the system is rather diffusion dominated if B is small and decay dominated if B is large B=np.sqrt(O/D)

    rodthickness = 5  # micrometer
    O=0.00000    01 #  in per second
    J=0.002
    Dcoltranss=0.0014 # micrometer ^2 / second
    B = np.sqrt(O/Dcoltranss)

    const = -2 * np.pi * Dcoltranss * rodthickness / 2 * B * kvp(0, B * rodthickness / 2)
    print(const)
    A= J/const
    print(A)
    r=np.linspace(2, 200, 100)
    l = B * r
    schmellmagnitude = A*kv(0,l)
    plt.plot(r,schmellmagnitude)
    plt.show()
    '''

    Deltadistance = np.linalg.norm(np.array(schmellpos) - np.array(colpos), axis=-1)
    Deltadistance = np.where(Deltadistance == 0, 5, Deltadistance)
    # prevent zero division
    direction = (schmellpos - colpos) / np.stack([Deltadistance, Deltadistance], axis=-1)
    '''
    #uncomment this for gaus curves
    schmellmagnitude += np.exp(-0.5 * Deltadistance ** 2 / rodthickness ** 2)
    schmellgradient += Deltadistance * np.exp(-0.5 * Deltadistance ** 2 / rodthickness ** 2) * direction
    '''
    O = 0.4  # in per second
    J = 1800  # in chemics per second whatever units the chemics are in. These values are chosen according to the Amplitude \approx 1
    # const = -2 * np.pi * Dcoltrans * rodthickness / 2 * np.sqrt(O / Dcoltrans) * kvp(0, np.sqrt(
    #   O / Dcoltrans) * rodthickness / 2) #reuse already calculated value
    const = 4182.925639571625
    # J=const*A the const sums over the chemics that flow throw an imaginary boundary at radius rodthickness/2
    # A is the Amplitude of the potential i.e. the chemical density (google "first ficks law" for theoretical info)
    A = J / const

    l = np.sqrt(O / Dcoltrans) * Deltadistance
    schmellmagnitude = A * kv(0, l)
    schmellgradient = - np.stack([A * kvp(0, l), A * kvp(0, l)], axis=-1) * direction
    return schmellmagnitude, schmellgradient
