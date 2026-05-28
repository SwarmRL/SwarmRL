from typing import Any

import numpy as np
from loguru import logger
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from swarmrl.engine.engine import Engine
from swarmrl.trainers.trainer import Trainer


class UniversalTrainer(Trainer):
    """
    Universal SwarmRL Trainer.
    Seamlessly supports On- and Off-policy agents, as well as
    episodic and continuous training routines.
    """

    @staticmethod
    def _is_off_policy_agent(agent) -> bool:
        """Duck-typed off-policy detection."""
        return getattr(agent, "replay_buffer", None) is not None

    @staticmethod
    def _get_agent_reward(agent) -> float:
        """Extract the latest reward from either off-policy or on-policy agents."""
        if hasattr(agent, "_last_reward"):
            return agent._last_reward
        if hasattr(agent, "trajectory") and agent.trajectory.rewards:
            return agent.trajectory.rewards[-1]
        return 0.0

    def perform_rl_training(
        self,
        n_episodes: int,
        episode_length: int,
        load_bar: bool = True,
        # for continuous training:
        system_runner: Engine = None,
        # for episodic training (including resets):
        get_engine: callable = None,
        system: Any = None,
        reset_frequency: int = 1,
        save_episodic_data: bool = False,
    ):
        """
        Perform the RL training loop.
        Adapts to training routine and agent requirements.

        This universal training loop supports both continuous (no environment resets)
        and episodic (with periodic environment resets) training regimes.
        - For Continuous Training: Provide an initialized `system_runner`.
        - For Episodic Training: Provide `get_engine` and `system`.

        Parameters
        ----------
        n_episodes : int
            Number of episodes to use in the training.
        episode_length : int
            Number of time steps in one episode.
        load_bar : bool (default=True)
            If true, show a progress bar.
        system_runner : Engine, optional
            Engine used to perform steps for each agent (Continuous Training).
            If provided, the environment will not be reset between episodes.
        get_engine : callable, optional
            Function to get the engine for the simulation (Episodic Training).
            Used to dynamically re-instantiate the environment.
        system : espressomd.System, optional
            The core physics system object passed to `get_engine` during resets.
        reset_frequency : int, default=1
            After how many episodes the simulation is reset. Only applies if
            `get_engine` is provided.
        save_episodic_data : bool (default=False)
            If true, save the episode data incrementally. The `get_engine` function
            must take `(system, str(cycle_index))` as arguments to pass the
            cycle_index to the EspressoMD engine as 'h5_group_tag'. If false,
            the data of the last episode is overwritten by the new data. See
            the implementation in
            CI/espresso_tests/integration_tests/test_rl_trainers.py
        """

        # Setup initial engine
        if system_runner is not None:
            self.engine = system_runner
            is_episodic = False
        elif get_engine is not None and system is not None:
            # Deliberately None so Episode 0 initializes it, as in Episodic Trainer
            self.engine = None
            is_episodic = True
        else:
            raise ValueError(
                "You must provide either `system_runner` (for continuous) "
                "or `get_engine` AND `system` (for episodic)."
            )

        force_fn = self.initialize_training()
        if not is_episodic:
            for agent in self.agents.values():
                agent.reset_agent(self.engine.colloids)
        rewards_history = []
        cycle_index = 0
        killed = False

        progress = Progress(
            "Episode: {task.fields[Episode]}",
            BarColumn(),
            "Reward: {task.fields[current_reward]} |"
            " Running: {task.fields[running_reward]}",
            TimeRemainingColumn(),
        )

        with progress:
            task = progress.add_task(
                "RL Training",
                total=n_episodes,
                Episode=0,
                current_reward=0.0,
                running_reward=0.0,
                visible=load_bar,
            )

            for episode in range(n_episodes):
                # 1. Environment reset logic
                # In episode 0, episode % reset_frequency == 0 -> Builds first engine.
                if is_episodic and (episode % reset_frequency == 0 or killed):
                    logger.info(f"Resetting system at episode {episode}")
                    if self.engine is not None:
                        self.engine.finalize()
                    self.engine = None

                    if save_episodic_data:
                        try:
                            self.engine = get_engine(system, f"{cycle_index}")
                            cycle_index += 1
                        except TypeError:
                            raise ValueError(
                                "The system runner does not support episodic data "
                                "saving. Your get_engine function should take a system "
                                "and a str(cycle_index) as arguments. The cycle_index "
                                "is passed to the EspressoMD engine as "
                                "'h5_group_tag'."
                            )
                    else:
                        self.engine = get_engine(system)

                    # Reset agents after environment was initialized/reset
                    for agent in self.agents.values():
                        agent.reset_agent(self.engine.colloids)

                current_reward = 0.0
                killed = False

                # 2. Integrate time_slice-wise
                for time_slice in range(episode_length):
                    # TODO: Can we improve this performance-wise,  (less steps)
                    # if we add an if statement checking for is_episodic?
                    # Integrate until the next time_slice
                    self.engine.integrate(1, force_fn)

                    step_reward = 0.0
                    for agent in self.agents.values():
                        step_reward += self._get_agent_reward(agent)

                        # Off-Policy Training
                        if self._is_off_policy_agent(agent):
                            agent.update_agent()

                        if agent.kill_switch:
                            killed = True

                    current_reward += step_reward
                    if killed:
                        logger.info("Simulation killed by task.")
                        break

                # 3. On-policy agents update once per episode.
                for agent in self.agents.values():
                    # Duck-Typing: PPO/ActorCritic has no replay buffer -> learns now.
                    if not self._is_off_policy_agent(agent):
                        agent.update_agent()

                # 4. Logging, history, checkpointing
                rewards_history.append(current_reward)
                self.maybe_save_checkpoint(
                    np.array(rewards_history), episode, current_reward
                )

                run_rew = np.round(np.mean(rewards_history[-10:]), 2)
                progress.update(
                    task,
                    advance=1,
                    Episode=episode + 1,
                    current_reward=np.round(current_reward, 2),
                    running_reward=run_rew,
                )

                # Check early stopping
                break_training, stop_after_episode = self.check_for_stop_criterion()
                if break_training:
                    if episode < stop_after_episode:
                        logger.info(
                            "Stopping criterion reached, but running out training"
                            f" until {stop_after_episode}"
                        )
                    else:
                        logger.info(
                            f"Stopping training after episode {stop_after_episode}"
                        )
                        break
                    break

            if self.engine is not None:
                self.engine.finalize()

            self.finalize_agents()

        return np.array(rewards_history)
