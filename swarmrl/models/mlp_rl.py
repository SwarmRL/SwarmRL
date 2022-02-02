"""
Module to implement a simple multi-layer perceptron for the colloids.
"""
from swarmrl.models.interaction_model import InteractionModel
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task
from swarmrl.losses.policy_gradient_loss import Loss
from swarmrl.models.interaction_model import Action
import torch
import numpy as np


class MLPRL(InteractionModel):
    """
    Class for the simple MLP RL implementation.

    The multi-layer perceptron learner is a simple global network model wherein all
    particles are passed into the network and forces computed for each individually.

    Attributes
    ----------
    actor : Network
                A sequential torch model to use as an actor.
    critic : Network
                A sequential torch model to use as a critic.
    task : callable
                Callable function from which a reward can be computed.
    """

    def __init__(
        self,
        actor: Network,
        critic: Network,
        task: Task,
        loss: Loss,
        observable: Observable,
    ):
        """
        Constructor for the MLP RL.

        Parameters
        ----------
        actor : torch.nn.Sequential
                A sequential torch model to use as an actor.
        critic : torch.nn.Sequential
                A sequential torch model to use as a critic.
        task : Task
                Callable from which a reward can be computed.
        loss : Loss
                A loss model to use in the A-C loss computation.
        actions : dict
                A dictionary of possible actions. Key names should describe the action
                and the value should be a data class. See the actions module for
                more information.
        observable : Observable
                Observable class with which to compute the input vector.
        """
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.task = task
        self.loss = loss
        self.observable = observable

        # Properties stored during the episode.
        self.action_probabilities = []
        self.observables = []

        # Actions on particles.
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.new_direction = None

        translate = Action(force=10.0)
        rotate_clockwise = Action(torque=np.array([0.0, 0.0, 1.0]))
        rotate_counter_clockwise = Action(torque=np.array([0.0, 0.0, -1.0]))
        do_nothing = Action()

        self.actions = {
            "RotateClockwise": rotate_clockwise,
            "Translate": translate,
            "RotateCounterClockwise": rotate_counter_clockwise,
            "DoNothing": do_nothing
        }

    def calc_action(self, colloid, other_colloids) -> Action:
        """
        Return the selected action on the particles.

        Parameters
        ----------
        colloid
        other_colloids

        Returns
        -------

        """
        action = self.compute_state(colloid, other_colloids)

        return self.actions[list(self.actions)[action]]

    def compute_feature_vector(self, colloid: object, other_colloids: list):
        """
        Compute the feature vector.

        Parameters
        ----------
        colloid : object
                Colloid of interest
        other_colloids : list
                List of all other colloids.

        Returns
        -------

        """
        return self.observable.compute_observable(colloid, other_colloids)

    def compute_state(self, colloid, other_colloids) -> int:
        """
        Compute the state of the active learning algorithm.

        If the model is not an active learner this method is ignored.

        Notes
        -----
        1.) Compute current state
        2.) Store necessary properties
        3.) Compute and set new forces / torques.
        """
        # Collect current state information.
        scaling = torch.nn.Softmax()
        state = self.compute_feature_vector(colloid, other_colloids)
        action_logits = self.actor(state)
        selector = torch.distributions.Categorical(abs(action_logits))
        action = selector.sample()
        action_probabilities = scaling(action_logits)

        # Update the stored data.
        self.action_probabilities.append(
            list(action_probabilities.detach().numpy())[action]
        )
        self.observables.append(state.numpy())

        return action

    def update_critic(self, loss: torch.Tensor):
        """
        Perform an update on the critic network.

        Parameters
        ----------
        loss : torch.Tensor
                A loss vector to pass to the model.

        Returns
        -------
        Runs back-propagation on the critic model.
        """
        self.critic.update_model(loss)

    def update_actor(self, loss: torch.Tensor):
        """
        Perform an update on the actor network.

        Parameters
        ----------
        loss : torch.Tensor
                A loss vector to pass to the model.
        Returns
        -------
        Runs back-propagation on the actor model.
        """
        self.actor.update_model(loss)

    def update_rl(self):
        """
        Update the RL algorithm.
        """
        # Compute real rewards.
        rewards = self.task.compute_reward(torch.tensor(self.observables))
        n_particles = rewards.shape[0]
        time_steps = rewards.shape[1]

        # Compute predicted rewards.
        predicted_rewards = self.critic(torch.tensor(self.observables))
        predicted_rewards = torch.reshape(
            predicted_rewards, (n_particles, time_steps)
        )

        action_probabilities = torch.reshape(
            torch.tensor(self.action_probabilities), (n_particles, time_steps)

        )
        # Compute loss.
        actor_loss, critic_loss = self.loss.compute_loss(
            action_probabilities,
            predicted_rewards,
            rewards,
        )
        # Perform back-propagation.
        self.update_actor(actor_loss)
        self.update_critic(critic_loss)

        # Clear out the parameters in the class.
        self.observables = []
        self.action_probabilities = []
