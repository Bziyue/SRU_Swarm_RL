"""RSL-RL agent configurations for static-mesh drone navigation."""

from isaaclab.utils import configclass

from isaaclab_nav_task.navigation.config.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


def _make_drone_policy_cfg(*, use_teammate_attention: bool = False) -> RslRlPpoActorCriticCfg:
    extra_kwargs = {}
    if use_teammate_attention:
        extra_kwargs = {
            "ego_input_dim": 16,
            "ego_embed_dim": 32,
            "teammate_feature_dim": 4,
            "max_teammates": 4,
            "teammate_embed_dim": 64,
            "teammate_attention_heads": 4,
        }

    return RslRlPpoActorCriticCfg(
        class_name="ActorCriticSRU",
        # Keep the original exploration level while preserving the 10 Hz rollout time scale.
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_hidden_size=512,
        rnn_type="lstm_sru",
        rnn_num_layers=1,
        dropout=0.2,
        num_cameras=1,
        image_input_dims=(64, 5, 8),
        height_input_dims=(64, 7, 7),
        **extra_kwargs,
    )


@configclass
class DroneStaticNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Keep a ~3.2 s rollout horizon after moving to 10 Hz control.
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 500
    logger = "tensorboard"
    seed = 60
    experiment_name = "drone_static_navigation_ppo"
    empirical_normalization = False
    reward_shifting_value = 0.05
    policy = _make_drone_policy_cfg(use_teammate_attention=False)
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.02,
        use_clipped_value_loss=True,
        clip_param=0.2,
        value_clip_param=0.2,
        entropy_coef=0.00375,
        num_learning_epochs=5,
        # Doubling rollout steps increases the batch size, so split updates into
        # more mini-batches to keep each gradient step closer to the previous setup.
        num_mini_batches=8,
        learning_rate=1.0e-3,
        schedule="adaptive",
        # Match the original 5 Hz per-second discount / GAE decay after moving to
        # 10 Hz control.
        gamma=0.9974968672,
        lam=0.9746794345,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DroneStaticNavPPORunnerDevCfg(DroneStaticNavPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "drone_static_navigation_ppo_dev"
        self.logger = "tensorboard"



@configclass
class DroneStaticNavPPOCompatRunnerCfg(DroneStaticNavPPORunnerCfg):
    experiment_name = "drone_static_navigation_ppo_swarm_compat"
    policy = _make_drone_policy_cfg(use_teammate_attention=True)


@configclass
class DroneStaticNavPPOCompatRunnerDevCfg(DroneStaticNavPPOCompatRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "drone_static_navigation_ppo_swarm_compat_dev"
        self.logger = "tensorboard"
