"""Unified configuration management module

This module provides project configuration management with modular design, dividing configuration into multiple sub-modules:
- IndexConfig: Quadtree index configuration
- DataConfig: Data loading configuration
- RewardConfig: Reward function configuration
- TrainConfig: Training process configuration
- NetworkConfig: Neural network configuration
- PathConfig: Path configuration

Usage example:
    # Load configuration from YAML file
    config = TShapeConfig.from_yaml('default.yaml')
    
    # Access configuration
    max_level = config.index.max_level
    num_trajs = config.data.num_trajectories
    
    # Save configuration
    config.save_yaml('configs/my_config.yaml')
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import json
import os
import torch
import yaml


def _default_dataset_profiles() -> Dict[str, "DatasetProfileConfig"]:
    tdrive_path = os.environ.get("DATASET_TDRIVE_PATH")
    chengdu_path = os.environ.get("DATASET_CHENGDU_PATH")

    return {
        "tdrive": DatasetProfileConfig(
            description="Beijing TDrive trajectory dataset",
            trajectory_path=tdrive_path,
            query_root="resource/queries/tdrive",
            min_x=115.29,
            min_y=39.00,
            max_x=117.83,
            max_y=41.50,
        ),
        "chengdu": DatasetProfileConfig(
            description="Chengdu CDTaxi trajectory dataset",
            trajectory_path=chengdu_path,
            query_root="resource/queries/chengdu",
            min_x=104.04,
            min_y=30.65,
            max_x=104.13,
            max_y=30.73,
        ),
    }


def _dataset_env_var(dataset_name: str) -> Optional[str]:
    env_map = {
        "tdrive": "DATASET_TDRIVE_PATH",
        "chengdu": "DATASET_CHENGDU_PATH",
        "cdtaxi": "DATASET_CHENGDU_PATH",
    }
    return env_map.get(dataset_name)


@dataclass
class ExperimentConfig:
    """Experiment configuration

    Attributes:
        name: Experiment name (used to organize output directory)
        description: Experiment description
    """
    name: str = "default"
    description: str = ""

    def get_output_dir(self, paths: Optional["PathConfig"] = None) -> Path:
        """Get experiment output directory: outputs/experiments/{name}/"""
        if paths is None:
            return Path("outputs/experiments") / self.name
        return paths.get_experiment_dir(self.name)

    def get_checkpoints_dir(self, paths: Optional["PathConfig"] = None) -> Path:
        """Get model checkpoint directory"""
        return self.get_output_dir(paths) / "checkpoints"

    def get_results_dir(self, paths: Optional["PathConfig"] = None) -> Path:
        """Get experiment result directory (order, evaluation, etc.)"""
        return self.get_output_dir(paths) / "results"

    def get_logs_dir(self, paths: Optional["PathConfig"] = None) -> Path:
        """Get experiment log directory"""
        return self.get_output_dir(paths) / "logs"

    def get_figures_dir(self, paths: Optional["PathConfig"] = None) -> Path:
        """Get experiment chart directory"""
        return self.get_output_dir(paths) / "figures"

    def get_similarity_dir(self, paths: Optional["PathConfig"] = None) -> Path:
        """Get experiment private similarity matrix directory."""
        return self.get_output_dir(paths) / "similarity"


@dataclass
class PathConfig:
    """Path configuration
    
    Attributes:
        outputs_dir: Output file base directory (supports relative and absolute paths)
        resource_dir: Resource file base directory
        
    Note:
        - Relative paths will be resolved relative to project root directory
        - Can be overridden by environment variables:
          * PROJECT_ROOT: Project root directory
          * OUTPUT_DIR: Output base directory
    """
    outputs_dir: str = "outputs"
    resource_dir: str = "resource"

    def _resolve_path(self, path: str) -> Path:
        """Resolve path, supports relative and absolute paths"""
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.project_root / candidate

    @property
    def project_root(self) -> Path:
        """Get project root directory"""
        env_root = os.environ.get("PROJECT_ROOT")
        if env_root:
            return Path(env_root).resolve()
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / "src").is_dir() and (parent / "configs").is_dir():
                return parent
        return Path.cwd()

    @property
    def outputs_path(self) -> Path:
        """Output directory root path"""
        env_outputs = os.environ.get("OUTPUT_DIR")
        if env_outputs:
            return Path(env_outputs)
        return self._resolve_path(self.outputs_dir)

    @property
    def resource_path(self) -> Path:
        """Resource directory root path"""
        env_resource = os.environ.get("RESOURCE_BASE_DIR")
        if env_resource:
            return Path(env_resource)
        return self._resolve_path(self.resource_dir)

    @property
    def checkpoints_dir(self) -> Path:
        """Model checkpoint save directory"""
        return self.outputs_path / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        """Log save directory"""
        return self.outputs_path / "logs"

    @property
    def figures_dir(self) -> Path:
        """Chart save directory"""
        return self.outputs_path / "figures"

    @property
    def experiments_dir(self) -> Path:
        """Experiment output directory"""
        return self.outputs_path / "experiments"

    @property
    def queries_dir(self) -> Path:
        """Query file directory"""
        return self.resource_path / "queries"

    @property
    def matrices_dir(self) -> Path:
        """Matrix data directory"""
        return self.resource_path / "matrices"

    @property
    def similarity_dir(self) -> Path:
        """Shared similarity matrix directory."""
        return self.matrices_dir / "similarity"

    @property
    def orders_dir(self) -> Path:
        """Traversal order directory"""
        return self.resource_path / "orders"

    def get_experiment_dir(self, experiment_name: str) -> Path:
        """Get output directory for specified experiment"""
        return self.experiments_dir / experiment_name


@dataclass
class DatasetProfileConfig:
    """Path and spatial boundary configuration for a single dataset."""

    description: str = ""
    trajectory_path: Optional[str] = None
    query_root: Optional[str] = None
    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 1.0
    max_y: float = 1.0

    def get_bbox_tuple(self) -> Tuple[float, float, float, float]:
        return self.min_x, self.min_y, self.max_x, self.max_y


@dataclass
class DatasetCatalogConfig:
    """Dataset catalog configuration."""

    active: str = "tdrive"
    profiles: Dict[str, DatasetProfileConfig] = field(default_factory=_default_dataset_profiles)


@dataclass
class IndexConfig:
    """Quadtree and TShape spatial index configuration
    
    Attributes:
        max_level: Maximum depth level of the quadtree
        alpha: Number of cells to expand elements in X direction
        beta: Number of cells to expand elements in Y direction
        min_x: Minimum X coordinate of original bounding box (longitude)
        min_y: Minimum Y coordinate of original bounding box (latitude)
        max_x: Maximum X coordinate of original bounding box (longitude)
        max_y: Maximum Y coordinate of original bounding box (latitude)
        use_original_bbox: Whether to use original bounding box (instead of normalized [0,0,1,1])
        use_prune: Whether to enable node pruning logic
        min_cell_trajs: Minimum trajectory count threshold in node, nodes below this will be pruned
        quadcode_include_muted: Whether quadCode traversal order includes muted nodes
        enable_sig_optimize: Whether to enable trajectory signature optimization
        parallel_signatures: Whether to compute signatures in parallel
        signature_workers: Number of worker processes for signature computation
    """
    max_level: int = 8
    alpha: int = 2
    beta: int = 2

    use_original_bbox: bool = True

    use_prune: bool = True
    min_cell_trajs: int = 0
    quadcode_include_muted: bool = False
    enable_sig_optimize: bool = False
    
    parallel_signatures: bool = True
    signature_workers: Optional[int] = None

    def get_bbox_tuple(self) -> Tuple[float, float, float, float]:
        """Get bounding box tuple
        
        Returns:
            (min_x, min_y, max_x, max_y) quadruple
        """
        return self.min_x, self.min_y, self.max_x, self.max_y


@dataclass
class DataConfig:
    """Trajectory data loading configuration
    
    Attributes:
        num_trajectories: Number of trajectories to load, -1 means load all
        source: Trajectory source, 'dataset' uses current active dataset, 'synthetic' uses synthetic data
        use_similarity_matrix: Whether to use pre-computed similarity matrix
        similarity_matrix_path: Similarity matrix file path (relative to resource/matrices/similarity/)
        similarity_num_workers: Number of worker processes when computing similarity matrix (None means use CPU core count)
        similarity_use_gpu: Whether to use GPU acceleration for similarity matrix computation
        similarity_gpu_batch_size: GPU computation batch size
        storage_mode: Trajectory storage mode ('memory'/'disk'/'auto')
        storage_dir: Disk storage directory path (required when storage_mode='disk')
        disk_cache_mb: Disk storage LRU cache size (MB)
        parallel_trajectory_assign: Whether to assign trajectories in parallel
        trajectory_assign_workers: Number of trajectory assignment worker processes
    """
    num_trajectories: int = 3000
    source: str = "dataset"
    use_similarity_matrix: bool = True
    similarity_matrix_path: Optional[str] = None
    similarity_num_workers: Optional[int] = 4
    
    similarity_use_gpu: bool = True
    similarity_gpu_batch_size: int = 1024
    
    parallel_trajectory_assign: bool = True
    trajectory_assign_workers: Optional[int] = None
    
    storage_mode: str = "auto"
    storage_dir: Optional[str] = None
    disk_cache_mb: int = 2048
    
    def get_similarity_matrix_path(self, paths: "PathConfig") -> Optional[Path]:
        """Get complete similarity matrix path"""
        if not self.similarity_matrix_path:
            return None
        path = Path(self.similarity_matrix_path)
        if path.is_absolute():
            return path
        return paths.similarity_dir / self.similarity_matrix_path


@dataclass
class RewardConfig:
    """Reward model and evaluation configuration
    
    Attributes:
        tau_loc: Location cost coefficient (controls position access cost weight)
        tau_scan: Scan cost coefficient (controls node scan cost weight)
        local_reward_weight: Weight of local reward in total reward
        global_reward_weight: Weight of global reward in total reward
        reward_schedule_episodes: Number of episodes for reward weight transition from warmup to target
        global_reward_start_scale: Global reward weight scaling ratio at training start
        local_reward_start_scale: Local reward weight scaling ratio at training start
        global_reward_scale: Global reward scaling multiplier
        global_reward_num_evals: Number of global reward computations
        global_reward_query_sample_size: Number of queries sampled for each global reward estimation during training, None means use all
        global_reward_frontload_exponent: Global reward checkpoint frontload exponent, larger values favor earlier triggers
        query_distribution_type: Query distribution type ('uniform'/'skewed'/'gaussian')
        query_sample_ratio: Sampling ratio from pre-partitioned file
    """
    tau_loc: float = 1.0
    tau_scan: float = 0.1
    local_reward_weight: float = 0.4
    global_reward_weight: float = 1.0
    reward_schedule_episodes: int = 100
    global_reward_start_scale: float = 0.25
    local_reward_start_scale: float = 1.0
    global_reward_scale: float = 2.0
    global_reward_num_evals: int = 1
    global_reward_query_sample_size: Optional[int] = 64
    global_reward_frontload_exponent: float = 1.5

    query_distribution_type: str = "skewed"
    query_sample_ratio: float = 1.0


@dataclass
class NetworkConfig:
    """Neural network architecture configuration
    
    Attributes:
        hidden_dims: Hidden layer dimensions list of neural network
        device: Device option ('auto'/'cuda'/'cpu')
        dropout: Dropout ratio
        state_dim: State feature dimension
    """
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    device: str = 'auto'
    dropout: float = 0.1
    state_dim: int = 14

    def get_torch_device(self) -> torch.device:
        """Get PyTorch device
        
        Returns:
            torch.device object, automatically detects CUDA availability
        """
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)


@dataclass
class TrainConfig:
    """Training process configuration
    
    Attributes:
        lr_actor: Actor network learning rate
        lr_critic: Critic network learning rate
        gamma: Discount factor
        gae_lambda: GAE (Generalized Advantage Estimation) lambda parameter, used to balance bias and variance
        num_episodes: Total training episodes
        eval_interval: Interval of episodes between full evaluations
        save_interval: Interval of episodes between model saves
        
        eps_clip: PPO clipping threshold
        k_epochs: Number of times to repeat learning from each round of interaction data
        gradient_clip_norm: Gradient clipping threshold
        
        entropy_coef_start: Entropy coefficient at training start (encourages exploration)
        entropy_coef_end: Entropy coefficient at training end (encourages exploitation)
        entropy_decay_episodes: Number of episodes for entropy coefficient decay (-1 means no decay, use fixed value entropy_coef_start)
        
        topk_actions: If set, filter Top-K action candidates based on node similarity
        topk_decay_episodes: Number of episodes for topK decay
        topk_multipliers: (start multiplier, end multiplier), used to dynamically adjust topk
        
        enable_early_stopping: Whether to enable early stopping mechanism
        early_stopping_patience: Patience value, stop if no improvement for consecutive rounds
        early_stopping_min_delta: Minimum improvement threshold
        max_negative_streak: Maximum consecutive negative reward rounds
    """
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_episodes: int = 1000
    eval_interval: int = 20
    save_interval: int = 100

    eps_clip: float = 0.2
    k_epochs: int = 4
    gradient_clip_norm: float = 1.0

    entropy_coef_start: float = 0.05
    entropy_coef_end: float = 0.001
    entropy_decay_episodes: int = -1

    topk_actions: Optional[int] = None
    topk_decay_episodes: int = 100
    topk_multipliers: Tuple[float, float] = (2.0, 1.0)

    enable_early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.01
    max_negative_streak: int = 5


@dataclass
class TShapeConfig:
    """Unified configuration class for TShape index system
    
    This is the top-level class of the configuration system, integrating all sub-configuration modules.
    
    Attributes:
        index: Quadtree index configuration
        data: Data loading configuration
        reward: Reward function configuration
        train: Training process configuration
        network: Neural network configuration
        paths: Path configuration
    
    Examples:
        >>> # Load from YAML file
        >>> config = TShapeConfig.from_yaml('default.yaml')
        >>> 
        >>> # Access configuration
        >>> max_level = config.index.max_level
        >>> num_trajs = config.data.num_trajectories
        >>> 
        >>> # Save configuration
        >>> config.save_yaml('configs/my_config.yaml')
    """
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    datasets: DatasetCatalogConfig = field(default_factory=DatasetCatalogConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TShapeConfig':
        """Create configuration object from dictionary
        
        Args:
            data: Dictionary containing configuration information
            
        Returns:
            TShapeConfig instance
        """
        config_dict = {}

        if 'experiment' in data:
            config_dict['experiment'] = ExperimentConfig(**data['experiment'])
        if 'datasets' in data:
            datasets_data = data['datasets'].copy()
            profiles_data = datasets_data.get('profiles', {}) or {}
            datasets_data['profiles'] = {
                name: DatasetProfileConfig(**profile_data)
                for name, profile_data in profiles_data.items()
            }
            config_dict['datasets'] = DatasetCatalogConfig(**datasets_data)
        if 'index' in data:
            config_dict['index'] = IndexConfig(**data['index'])
        if 'data' in data:
            config_dict['data'] = DataConfig(**data['data'])
        if 'reward' in data:
            config_dict['reward'] = RewardConfig(**data['reward'])
        if 'train' in data:
            train_data = data['train'].copy()
            if 'topk_multipliers' in train_data and isinstance(train_data['topk_multipliers'], list):
                train_data['topk_multipliers'] = tuple(train_data['topk_multipliers'])
            config_dict['train'] = TrainConfig(**train_data)
        if 'network' in data:
            network_data = data['network'].copy()
            config_dict['network'] = NetworkConfig(**network_data)
        if 'paths' in data:
            config_dict['paths'] = PathConfig(**data['paths'])

        config = cls(**config_dict)

        # If a YAML config defines dataset profiles but leaves trajectory_path empty,
        # allow environment variables to provide the concrete local dataset path.
        for dataset_name, profile in config.datasets.profiles.items():
            if profile.trajectory_path:
                continue
            env_name = _dataset_env_var(dataset_name)
            if env_name:
                profile.trajectory_path = os.environ.get(env_name)

        return config

    @classmethod
    def from_yaml(cls, path: str) -> 'TShapeConfig':
        """Load configuration from YAML file
        
        Args:
            path: YAML configuration file path
            
        Returns:
            TShapeConfig instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> 'TShapeConfig':
        """Load configuration from JSON file
        
        Args:
            path: JSON configuration file path
            
        Returns:
            TShapeConfig instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary
        
        Returns:
            Dictionary containing all configuration information
        """
        data = asdict(self)
        self._convert_tuples_to_lists(data)
        return data

    @staticmethod
    def _convert_tuples_to_lists(obj):
        """Recursively convert tuples to lists in dictionary"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, tuple):
                    obj[key] = list(value)
                elif isinstance(value, dict):
                    TShapeConfig._convert_tuples_to_lists(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            TShapeConfig._convert_tuples_to_lists(item)

    def save_yaml(self, path: str):
        """Save as YAML file
        
        Args:
            path: Save path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def save_json(self, path: str):
        """Save as JSON file
        
        Args:
            path: Save path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def _resolve_optional_path(self, path: Optional[str]) -> Optional[Path]:
        if not path:
            return None

        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (Path.cwd() / candidate).resolve()

    def get_active_dataset_profile(self) -> Optional[DatasetProfileConfig]:
        """Get currently active dataset configuration."""
        return self.datasets.profiles.get(self.datasets.active)

    def get_dataset_trajectory_path(self) -> Optional[Path]:
        """Get current dataset trajectory file path."""
        profile = self.get_active_dataset_profile()
        if profile and profile.trajectory_path:
            return self._resolve_optional_path(profile.trajectory_path)
        return None

    def get_query_dataset_root(self) -> Path:
        """Get query set root directory corresponding to current dataset."""
        profile = self.get_active_dataset_profile()
        if profile and profile.query_root:
            resolved = self._resolve_optional_path(profile.query_root)
            if resolved is not None:
                return resolved
        raise ValueError(f"Current dataset {self.datasets.active} has no query_root configured")

    def get_effective_bbox_tuple(self) -> Tuple[float, float, float, float]:
        """Get current effective bounding box, prioritizing active dataset's boundary."""
        profile = self.get_active_dataset_profile()
        if profile is not None:
            return profile.get_bbox_tuple()
        return self.index.get_bbox_tuple()

    def get_original_bbox(self):
        """Get original bounding box object
        
        Returns:
            SpatialBoundingBox instance
        """
        from src.common import SpatialBoundingBox
        min_x, min_y, max_x, max_y = self.get_effective_bbox_tuple()
        return SpatialBoundingBox(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y
        )

    def get_original_bbox_tuple(self) -> Tuple[float, float, float, float]:
        """Get original bounding box in tuple form
        
        Returns:
            (min_x, min_y, max_x, max_y) quadruple
        """
        return self.get_effective_bbox_tuple()

    def get_default_similarity_matrix_filename(self) -> str:
        """Get default similarity matrix filename.

        Shared matrix needs to be uniquely identified by key parameters that affect
        the cell set, to avoid different experiments mistakenly reusing historical matrices.
        """
        min_trajs = self.index.min_cell_trajs
        min_trajs_token = "none" if min_trajs is None else str(min_trajs)
        query_dist = (self.reward.query_distribution_type or "unknown").lower()
        return (
            f"sim_mtx_{self.datasets.active}_"
            f"{query_dist}_"
            f"R{self.index.max_level}_"
            f"M{min_trajs_token}_"
            f"A{self.index.alpha}_"
            f"B{self.index.beta}_"
            f"T{self.data.num_trajectories}.npz"
        )

    def get_default_similarity_matrix_path(self) -> Path:
        """Get default shared similarity matrix path."""
        return self.paths.similarity_dir / self.get_default_similarity_matrix_filename()

    def get_effective_similarity_matrix_path(self) -> Path:
        """Get similarity matrix path actually used by current configuration."""
        explicit_path = self.data.get_similarity_matrix_path(self.paths)
        if explicit_path is not None:
            return explicit_path
        return self.get_default_similarity_matrix_path()

    def get_experiment_similarity_matrix_path(self) -> Path:
        """Get current experiment's private similarity matrix path."""
        return self.experiment.get_similarity_dir(self.paths) / self.get_default_similarity_matrix_filename()

