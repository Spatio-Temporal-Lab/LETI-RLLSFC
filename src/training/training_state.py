"""Training state management."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingState:
    """Training process state tracking.
    
    Centrally manages all state variables during training to avoid scattered management in the main class.
    """
    
    episode_rewards: List[float] = field(default_factory=list)
    """Total reward for each episode"""
    
    episode_lengths: List[int] = field(default_factory=list)
    """Number of steps for each episode"""
    
    train_improvement_history: List[float] = field(default_factory=list)
    """Historical improvement rate records for training set at each evaluation"""
    
    val_improvement_history: List[float] = field(default_factory=list)
    """Historical improvement rate records for validation set at each evaluation"""
    
    test_improvement_history: List[float] = field(default_factory=list)
    """Historical improvement rate records for test set at each evaluation"""
    
    loss_history: List[float] = field(default_factory=list)
    """Loss function value at each episode update"""
    
    improvement_episodes: List[int] = field(default_factory=list)
    """Episode number corresponding to each evaluation occurrence"""
    
    best_improvement: float = float('-inf')
    """Highest improvement rate percentage achieved during training"""
    
    patience_counter: int = 0
    """Current patience consumption counter"""
    
    negative_streak_counter: int = 0
    """Maximum consecutive period of negative returns"""
    
    early_stop_episode: Optional[int] = None
    """Episode number when early stopping is triggered; None otherwise"""
    
    def record_episode(self, reward: float, length: int, loss: Optional[float] = None) -> None:
        """Record the result of a single episode.
        
        Args:
            reward: Total episode reward
            length: Episode step count
            loss: Loss value (optional)
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if loss is not None:
            self.loss_history.append(loss)
    
    def record_evaluation(self, episode: int, 
                          train_improvement: float,
                          val_improvement: float,
                          test_improvement: float) -> None:
        """Record evaluation results.
        
        Args:
            episode: Current episode number
            train_improvement: Training set improvement rate
            val_improvement: Validation set improvement rate
            test_improvement: Test set improvement rate
        """
        self.improvement_episodes.append(episode)
        
        self.train_improvement_history.append(train_improvement if train_improvement is not None else 0.0)
        self.val_improvement_history.append(val_improvement if val_improvement is not None else 0.0)
        self.test_improvement_history.append(test_improvement if test_improvement is not None else 0.0)
    
    def update_best_improvement(self, improvement: float) -> bool:
        """Update the best improvement rate.
        
        Args:
            improvement: Current improvement rate
            
        Returns:
            Whether the best record was refreshed
        """
        if improvement > self.best_improvement:
            self.best_improvement = improvement
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False
    
    def update_negative_streak(self, improvement: float) -> None:
        """Update the consecutive negative returns counter.
        
        Args:
            improvement: Current improvement rate
        """
        if improvement < 0:
            self.negative_streak_counter += 1
        else:
            self.negative_streak_counter = 0
    
    def get_recent_avg_reward(self, window: int = 10) -> float:
        """Get the average reward of the most recent N episodes.
        
        Args:
            window: Window size
            
        Returns:
            Average reward
        """
        if not self.episode_rewards:
            return 0.0
        recent = self.episode_rewards[-window:]
        return sum(recent) / len(recent)
