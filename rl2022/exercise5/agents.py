from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import List, Dict, DefaultDict

from gym.spaces import Space
from gym.spaces.utils import flatdim


class MultiAgent(ABC):
    """Base class for multi-agent reinforcement learning

    **DO NOT CHANGE THIS BASE CLASS**

    """

    def __init__(
        self,
        num_agents: int,
        action_spaces: List[Space],
        gamma: float,
        **kwargs
    ):
        """Constructor of base agent for Q-Learning

        Initializes basic variables of MARL agents
        namely epsilon, learning rate and discount rate.

        :param num_agents (int): number of agents
        :param action_spaces (List[Space]): action spaces of the environment for each agent
        :param gamma (float): discount factor (gamma)

        :attr n_acts (List[int]): number of actions for each agent
        """

        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.n_acts = [flatdim(action_space) for action_space in action_spaces]

        self.gamma: float = gamma

    @abstractmethod
    def act(self) -> List[int]:
        """Chooses an action for all agents for stateless task

        :return (List[int]): index of selected action for each agent
        """
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def learn(self):
        ...


class IndependentQLearningAgents(MultiAgent):
    """Agent using the Independent Q-Learning algorithm

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of IndependentQLearningAgents

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping actions ACTs
            to respective Q-values for all agents

        Initializes some variables of the Independent Q-Learning agents, namely the epsilon, discount rate
        and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for i in range(self.num_agents)]


    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :return (List[int]): index of selected action for each agent
        """
        selected_actions = []
        for i in range(self.num_agents):
            actions = self.n_acts[i]
            q_table = self.q_tables[i]
            r = random.random()
        
            if r <= self.epsilon:
                selected_action = random.randint(0, actions-1) #select random action
            else:
                action_values = [q_table[action] for action in range(actions)]
                max_value = max(action_values)
                #select random action with maximum value
                selected_action = random.choice([i for i, action_value in enumerate(action_values) if max_value == action_value])
            #add selected action to list of selected actions
            selected_actions.append(selected_action)
        return selected_actions

    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current actions of each agent
        """
        updated_values = []
        for i in range(self.num_agents):
            best_q_val = 0 if dones[i] else max([self.q_table[i][action] for action in range(self.n_acts[i])])
            self.q_tables[i][actions[i]] = self.q_tables[i][actions[i]] + self.learning_rate * (rewards[i] + self.gamma * best_q_val - self.q_tables[i][actions[i]])
            updated_values = self.q_tables[i][actions[i]]
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        decay, max_deduct = 0.15, 0.6
        self.epsilon = 1.0(min(1.0, timestep/decay *max_timestep)) *max_deduct


class JointActionLearning(MultiAgent):
    """
    Agents using the Joint Action Learning algorithm with Opponent Modelling

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**
    """

    def __init__(self, learning_rate: float =0.5, epsilon: float =1.0, **kwargs):
        """Constructor of JointActionLearning

        :param learning_rate (float): learning rate for Q-learning updates
        :param epsilon (float): epsilon value for all agents

        :attr q_tables (List[DefaultDict]): tables for Q-values mapping joint actions ACTs
            to respective Q-values for all agents
        :attr models (List[DefaultDict]): each agent holding model of other agent
            mapping other agent actions to their counts

        Initializes some variables of the Joint Action Learning agents, namely the epsilon, discount rate and learning rate
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_acts = [flatdim(action_space) for action_space in self.action_spaces]

        # initialise Q-tables for all agents
        self.q_tables: List[DefaultDict] = [defaultdict(lambda: 0) for _ in range(self.num_agents)]

        # initialise models for each agent mapping state to other agent actions to count of other agent action
        # in state
        self.models = [defaultdict(lambda: 0) for _ in range(self.num_agents)] 

    def act(self) -> List[int]:
        """Implement the epsilon-greedy action selection here for stateless task

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :return (List[int]): index of selected action for each agent
        """
        joint_action = []
        for i in range(self.num_agents):
            if random.random() < self.epsilon:
                selected_action = random.randint(0, self.n_acts[i] - 1)
            else:
                actions_evs = 0
                max_ev = max(actions_evs)
                selected_action = random.choice([i for i, action_ev in enumerate(actions_evs) if action_ev == max_ev])
            joint_action.append(selected_action)
        return joint_action

    def learn(
        self, actions: List[int], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """Updates the Q-tables and models based on agents' experience

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        :param action (List[int]): index of applied action of each agent
        :param rewards (List[float]): received reward for each agent
        :param dones (List[bool]): flag indicating whether a terminal state has been reached for each agent
        :return (List[float]): updated Q-values for current observation-action pair of each agent
        """
        updated_values = []
        
        return updated_values

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        raise NotImplementedError("Needed for Q5")
