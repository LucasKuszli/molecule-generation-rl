"""Training hyper-parameter and path configurations for every pipeline stage.

Each stage (pretrain, transfer, GNN, REINFORCE) owns a ``*Config`` class
that bundles training parameters, model parameters, and artifact paths.
"""

from src.config.data import root_dir

# Pretraining settings.


class PretrainingTrainingParameters:
    """Optimiser and schedule settings for SMILES-LSTM pretraining."""

    num_epochs: int = 40
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    patience: int = 3


class PretrainingModelParameters:
    """Architecture settings for the character-level SMILES LSTM."""

    hidden_size: int = 128
    embedding_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1


class PretrainingConfig:
    """Aggregate config for the pretraining stage.

    Args:
        seed(int): random seed that selects the artifact sub-directory.
    """

    def __init__(self, seed: int = 42):
        self.training = PretrainingTrainingParameters()
        self.model = PretrainingModelParameters()
        self.seed = seed
        self.vocab_path = root_dir / "data" / "training_data" / "vocab.json"
        self.sample_n = 512
        self.checkpoint_outdir = root_dir / "artifacts" / "pretrain" / f"seed_{seed}"
        self.eval_outdir = root_dir / "evaluation" / "pretrain" / f"seed_{seed}"
        self.img_outdir = root_dir / "img" / "pretrain" / f"seed_{seed}"


# Transfer learning settings.


class TransferLearningTrainingParameters:
    """Optimiser settings for fine-tuning on HIV actives."""

    num_epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    patience: int = 10


class TransferLearningConfig:
    """Aggregate config for the transfer-learning stage.

    Args:
        seed(int): random seed that selects the artifact sub-directory.
    """

    def __init__(self, seed: int = 42):
        self.training = TransferLearningTrainingParameters()
        self.model = PretrainingModelParameters()
        self.seed = seed
        self.sample_n = 512
        self.checkpoint_indir = root_dir / "artifacts" / "pretrain" / f"seed_{seed}"
        self.checkpoint_outdir = (
            root_dir / "artifacts" / "transfer_learning" / f"seed_{seed}"
        )
        self.eval_outdir = (
            root_dir / "evaluation" / "transfer_learning" / f"seed_{seed}"
        )
        self.img_outdir = root_dir / "img" / "transfer_learning" / f"seed_{seed}"


# GNN settings.


class GNNTrainingParameters:
    """Training settings for the GINE HIV-activity classifier."""

    num_epochs: int = 100
    batch_size: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    inactive_ratio: float = 5.0  # max inactives per active (0 = no limit)


class GNNModelParameters:
    """Architecture settings for the GINE graph neural network."""

    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    edge_dim: int = 3


class GNNConfig:
    """Aggregate config for GNN training, evaluation, and tuning.

    Args:
        seed(int): random seed that selects the artifact sub-directory.
    """

    def __init__(self, seed: int = 42):
        self.training = GNNTrainingParameters()
        self.model = GNNModelParameters()
        self.seed = seed
        self.checkpoint_outdir = root_dir / "artifacts" / "gnn" / f"seed_{seed}"
        self.eval_outdir = root_dir / "evaluation" / "gnn" / f"seed_{seed}"
        self.img_outdir = root_dir / "img" / "gnn" / f"seed_{seed}"
        self.tuning_outdir = root_dir / "artifacts" / "gnn" / "tuning"


# Reinforcement learning settings.


class RLTrainingParameters:
    """Hyper-parameters for REINFORCE fine-tuning.

    The composite reward is:
        w_activity * P(active) + w_validity * int(valid)
        + w_sa * (1 - SA / 10)
    """

    num_epochs: int = 50
    batch_size: int = 1000
    learning_rate: float = 1e-4
    sigma: float = 10.0
    grad_clip: float = 1.0
    patience: int = 15

    w_activity: float = 2.0
    w_validity: float = 1.0
    w_sa: float = 0.1
    prior_weight: float = 0.01


class RLConfig:
    """Aggregate config for REINFORCE fine-tuning of a SMILES LSTM.

    Args:
        seed(int): random seed.
        stage(str): ``"pretrain_rl"`` or ``"transfer_rl"``.
    """

    def __init__(self, seed: int = 42, stage: str = "pretrain_rl"):
        self.training = RLTrainingParameters()
        self.seed = seed
        self.stage = stage

        if stage == "pretrain_rl":
            self.checkpoint_indir = root_dir / "artifacts" / "pretrain" / f"seed_{seed}"
        elif stage == "transfer_rl":
            self.checkpoint_indir = (
                root_dir / "artifacts" / "transfer_learning" / f"seed_{seed}"
            )
        else:
            raise ValueError(f"Unknown RL stage: {stage!r}")

        self.checkpoint_outdir = root_dir / "artifacts" / stage / f"seed_{seed}"
        self.eval_outdir = root_dir / "evaluation" / stage / f"seed_{seed}"
        self.img_outdir = root_dir / "img" / stage / f"seed_{seed}"
        self.gnn_checkpoint = (
            root_dir / "artifacts" / "gnn" / f"seed_{seed}" / "best_gnn.pt"
        )

        # KL prior must match the model the RL agent was initialised from.
        if stage == "pretrain_rl":
            self.prior_checkpoint = (
                root_dir / "artifacts" / "pretrain" / f"seed_{seed}" / "best_prior.pt"
            )
        else:
            self.prior_checkpoint = (
                root_dir
                / "artifacts"
                / "transfer_learning"
                / f"seed_{seed}"
                / "best_transfer.pt"
            )
