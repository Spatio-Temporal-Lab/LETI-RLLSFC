from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.ppo_updater import PPOUpdater
from src.rl.replay_buffer import ReplayBuffer, Transition


class TraversalActorNetwork(nn.Module):
    """
    Policy network (Actor) for generating probability distribution (unnormalized logits) for each action based on input state.

    Architecture description:
        Input: state vector (state)
        Output: action logits for computing softmax probability distribution.

    Args:
        state_dim : int
            State feature dimension.
        action_dim : int
            Number of available actions.
        hidden_dims : List[int], optional
            List of hidden layer neuron counts (default: [256, 256]).
        dropout_rate : float, optional
            Dropout ratio (default: 0.1).
    """

    DEFAULT_DROPOUT_RATE = 0.1
    MASKED_LOGIT_PENALTY = 1e9

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        dropout_rate: float = None
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        dropout_rate = dropout_rate if dropout_rate is not None else self.DEFAULT_DROPOUT_RATE
        
        layers: List[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation, output logits for each action.

        Args:
            state : torch.Tensor
                State input tensor with shape (batch_size, state_dim).

        Returns:
            torch.Tensor
                Action logits with shape (batch_size, action_dim).
        """
        return self.network(state)

    def select_action(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor]:
        """
        Sample a valid action from the policy distribution.

        Logic description:
            1. Compute logits for input state;
            2. Mask invalid actions based on action mask;
            3. Apply softmax to remaining actions to get probability distribution;
            4. Sample action from distribution.

        Args:
            state : torch.Tensor
                Current state tensor with shape (1, state_dim).
            action_mask : torch.Tensor
                Action mask (1 indicates available action, 0 indicates disabled action).

        Returns:
            Tuple[int, torch.Tensor]
                - Sampled action index;
                - Corresponding log probability.
        """
        logits = self.forward(state)
        masked_logits = logits + (action_mask - 1.0) * self.MASKED_LOGIT_PENALTY
        probabilities = F.softmax(masked_logits, dim=-1)

        distribution = torch.distributions.Categorical(probabilities)
        if deterministic:
            action = torch.argmax(probabilities, dim=-1)
        else:
            action = distribution.sample()

        return action.item(), distribution.log_prob(action)


class TraversalCriticNetwork(nn.Module):
    """
    Value network (Critic) for estimating expected return of input state (i.e., V(s)).

    Args:
        state_dim : int
            State feature dimension.
        hidden_dims : List[int], optional
            List of hidden layer neuron counts (default: [256, 256]).
    """

    def __init__(self, state_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]

        layers: List[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation, output value estimate for current state.

        Args:
            state : torch.Tensor
                State input tensor with shape (batch_size, state_dim).

        Returns:
            torch.Tensor
                State value estimate with shape (batch_size, 1).
        """
        return self.network(state)


class TraversalPolicyAgent:
    """
    Actor–Critic agent responsible for learning traversal policies.

    PPO (Proximal Policy Optimization) based agent for learning optimal action policies in spatial traversal tasks.

    Module structure:
        - Actor: Policy network for outputting action distribution;
        - Critic: Value network for evaluating state value;
        - Optimizer: Independent Adam optimizers for actor/critic.
        - ReplayBuffer: Experience replay buffer.
        - PPOUpdater: PPO update logic encapsulation (supports GAE).

    Args:
        state_dim : int
            State feature dimension.
        action_dim : int
            Action space dimension.
        lr_actor : float, default = 3e-4
            Learning rate for policy network (Actor).
        lr_critic : float, default = 3e-4
            Learning rate for value network (Critic).
        eps_clip: float, default = 0.2
            PPO clipping threshold.
        k_epochs: int, default = 4
            Number of times to repeat learning on each round of interaction data.
        gamma : float, default = 0.99
            Discount factor.
        entropy_coef: float, default = 0.01
            Entropy coefficient for encouraging exploration.
        gae_lambda: float, default = 0.95
            GAE lambda parameter for balancing bias and variance.
        gradient_clip_norm: float, default = 1.0
            Gradient clipping threshold.
        device : str, default = "cpu"
            Computing device ("cpu" or "cuda").
        hidden_dims : List[int], optional
            Hidden layer structure (default: [256, 256]).
        dropout_rate : float, optional
            Dropout ratio (default: 0.1).
    """

    LR_SCHEDULER_T_MAX = 100
    LR_SCHEDULER_ETA_MIN = 1e-6

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            eps_clip: float = 0.2,
            k_epochs: int = 4,
            gamma: float = 0.99,
            entropy_coef: float = 0.01,
            gae_lambda: float = 0.95,
            gradient_clip_norm: float = 1.0,
            device: str = "cpu",
            hidden_dims: List[int] = None,
            dropout_rate: float = None,
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim

        hidden_dims = hidden_dims or [256, 256]
        self.actor = TraversalActorNetwork(
            state_dim, action_dim, hidden_dims, dropout_rate
        ).to(self.device)
        self.critic = TraversalCriticNetwork(state_dim, hidden_dims).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=self.LR_SCHEDULER_T_MAX, eta_min=self.LR_SCHEDULER_ETA_MIN
        )
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=self.LR_SCHEDULER_T_MAX, eta_min=self.LR_SCHEDULER_ETA_MIN
        )

        self.replay_buffer = ReplayBuffer()

        self.ppo_updater = PPOUpdater(
            actor=self.actor,
            critic=self.critic,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            gamma=gamma,
            eps_clip=eps_clip,
            k_epochs=k_epochs,
            entropy_coef=entropy_coef,
            device=self.device,
            gae_lambda=gae_lambda,
            gradient_clip_norm=gradient_clip_norm
        )

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action based on current state and return action, log probability, and state value.

        Args:
            state : np.ndarray
                Current environment state vector.
            action_mask : np.ndarray
                Valid action mask (invalid action positions are 0).

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor]
                - action: Selected action index;
                - log_prob: Log probability corresponding to the action;
                - value: Value estimate for current state.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value = self.critic(state_tensor)
            action, log_prob = self.actor.select_action(
                state_tensor,
                mask_tensor,
                deterministic=deterministic,
            )
        return action, log_prob, value

    def store_transition(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            log_prob: torch.Tensor,
            value: torch.Tensor,
            mask: np.ndarray,
            done: bool,
    ) -> None:
        """
        Store interaction experience (state, action, reward, etc.) in buffer for subsequent updates.

        Args:
            state : np.ndarray
                Current state vector.
            action : int
                Currently executed action index.
            reward : float
                Reward obtained after executing action.
            log_prob : torch.Tensor
                Log probability of the action for policy gradient computation.
            value : torch.Tensor
                Value estimate for current state.
            mask : np.ndarray
                Action mask.
            done : bool
                Whether it is a terminal state.
        """
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            log_prob=log_prob,
            value=value,
            mask=mask,
            done=done
        )
        self.replay_buffer.store(transition)

    def update(self) -> Dict[str, float]:
        """
        Update Actor-Critic networks using stored interaction sequences.
        
        Returns:
            Dictionary containing loss information
        """
        if self.replay_buffer.is_empty():
            return {}

        loss_info = self.ppo_updater.update(self.replay_buffer.get_all())

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        self.replay_buffer.clear()

        return loss_info

    def save(self, filepath: str) -> None:
        """Save model state, optimizers, and schedulers."""
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "actor_sched": self.actor_scheduler.state_dict(),
            "critic_sched": self.critic_scheduler.state_dict()
        }, filepath)

    def load(self, filepath: str) -> None:
        """Load model and optimizer parameters from specified file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        self.actor_scheduler.load_state_dict(checkpoint["actor_sched"])
        self.critic_scheduler.load_state_dict(checkpoint["critic_sched"])
