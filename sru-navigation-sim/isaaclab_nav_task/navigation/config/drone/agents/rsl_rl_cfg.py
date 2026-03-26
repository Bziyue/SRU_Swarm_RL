"""RSL-RL agent configurations for static-mesh drone navigation."""

from isaaclab.utils import configclass

from isaaclab_nav_task.navigation.config.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DroneStaticNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Keep roughly the same rollout duration in wall-clock time after moving the
    # high-level policy from 5 Hz to 20 Hz: 16 * 0.2 s ~= 64 * 0.05 s.
    num_steps_per_env = 64
    max_iterations = 15000
    save_interval = 500
    logger = "tensorboard"
    seed = 60
    experiment_name = "drone_static_navigation_ppo"
    empirical_normalization = False
    reward_shifting_value = 0.05
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSRU",
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
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.02,
        use_clipped_value_loss=True,
        clip_param=0.2,
        value_clip_param=0.2,
        entropy_coef=0.00375,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        # Match the original 5 Hz per-second discount / GAE decay after moving to
        # 20 Hz control.
        gamma=0.9987476502,
        lam=0.9872585449,
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


@configclass
class DroneStaticNavPPOCompatRunnerDevCfg(DroneStaticNavPPOCompatRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "drone_static_navigation_ppo_swarm_compat_dev"
        self.logger = "tensorboard"
