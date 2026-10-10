"""PPO update utilities."""
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.rl.replay_buffer import Transition


class PPOUpdater:
    """Encapsulate PPO update logic with GAE support."""

    ADVANTAGE_EPSILON = 1e-8

    def __init__(
        self,
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        gamma: float,
        eps_clip: float,
        k_epochs: int,
        entropy_coef: float,
        device: torch.device,
        gae_lambda: float = 0.95,
        gradient_clip_norm: float = 1.0,
    ):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.device = device
        self.gae_lambda = gae_lambda
        self.gradient_clip_norm = gradient_clip_norm

    def compute_gae_advantages(
        self,
        rewards: List[float],
        values: torch.Tensor,
        dones: List[bool],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized policy advantages and unnormalized critic returns.

        Args:
            rewards: Per-step rewards
            values: Value estimates recorded when the actions were taken
            dones: Terminal flags aligned with rewards
        """
        raw_advantages: List[float] = []
        gae = 0.0
        values_list = values.detach().cpu().reshape(-1).numpy().tolist()

        for t in reversed(range(len(rewards))):
            next_value = values_list[t + 1] if t + 1 < len(rewards) else 0.0

            delta = rewards[t] + self.gamma * next_value * (1.0 - float(dones[t])) - values_list[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            raw_advantages.insert(0, gae)

        raw_advantages_tensor = torch.tensor(raw_advantages, dtype=torch.float32, device=self.device)
        returns = raw_advantages_tensor + values
        normalized_advantages = (
            (raw_advantages_tensor - raw_advantages_tensor.mean())
            / (raw_advantages_tensor.std(unbiased=False) + self.ADVANTAGE_EPSILON)
        )
        return normalized_advantages, returns

    def update(self, transitions: List[Transition]) -> Dict[str, float]:
        """Run one PPO update over the collected transitions."""
        if not transitions:
            return {}

        old_states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        old_actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        old_log_probs = torch.stack([t.log_prob for t in transitions]).reshape(-1).detach().to(self.device)
        old_values = torch.stack([t.value for t in transitions]).reshape(-1).detach().to(self.device)
        candidate_actions = [
            torch.as_tensor(t.candidate_actions, dtype=torch.long, device=self.device)
            for t in transitions
        ]
        if any(candidates.numel() == 0 for candidates in candidate_actions):
            raise RuntimeError("PPO transition has no candidate actions")
        max_candidates = max(candidates.numel() for candidates in candidate_actions)
        padded_candidates = torch.zeros(
            (len(candidate_actions), max_candidates), dtype=torch.long, device=self.device
        )
        candidate_mask = torch.zeros(
            (len(candidate_actions), max_candidates), dtype=torch.bool, device=self.device
        )
        local_actions = torch.empty(len(candidate_actions), dtype=torch.long, device=self.device)
        for idx, candidates in enumerate(candidate_actions):
            width = candidates.numel()
            padded_candidates[idx, :width] = candidates
            candidate_mask[idx, :width] = True
            selected = (candidates == old_actions[idx]).nonzero(as_tuple=False)
            if selected.numel() == 0:
                raise RuntimeError("Stored action is absent from its candidate set")
            local_actions[idx] = selected[0, 0]

        rewards = [t.reward for t in transitions]
        dones = [t.done for t in transitions]
        advantages, returns = self.compute_gae_advantages(rewards, old_values, dones)

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(self.k_epochs):
            candidate_logits = self.actor.score_candidates(old_states, padded_candidates)
            candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1e9)
            dist = Categorical(logits=candidate_logits)
            new_log_probs = dist.log_prob(local_actions)
            entropy = dist.entropy().mean()

            new_values = self.critic(old_states).reshape(-1)

            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            critic_loss = F.mse_loss(new_values, returns)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_norm)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip_norm)
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return {
            "loss": (total_actor_loss + total_critic_loss) / self.k_epochs,
            "actor_loss": total_actor_loss / self.k_epochs,
            "critic_loss": total_critic_loss / self.k_epochs,
        }
