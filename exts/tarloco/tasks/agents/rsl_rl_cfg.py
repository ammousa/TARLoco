# Copyright (c) 2025, Amr Mousa, University of Manchester
# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# This file is based on code from the isaaclab repository:
# https://github.com/isaac-sim/IsaacLab/
#
# The original code is licensed under the BSD 3-Clause License.
# See the `licenses/` directory for details.
#
# This version includes significant modifications by Amr Mousa (2025).

from isaaclab.utils import configclass

from exts.tarloco.tasks.algorithms import *

# =============================
#         MLP POLICIES
# =============================


@configclass
class Go1RoughPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "TAR_workspace"
    empirical_normalization = True
    policy = RslRlPpoPolicyCfg(
        class_name="ActorCriticMlp",
        init_noise_std=1.0,
        clip_action=100.0,
        squash_mode="clip",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        lr_max=1.0e-3,
        lr_min=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="Adam",
    )


@configclass
class Go1RoughPpoExpertRunnerCfg(Go1RoughPpoRunnerCfg):
    policy = RslRlPpoSlrPolicyCfg(
        class_name="ActorCriticMlpDblEncExpert",
        num_hist=1,
        latent_dims=20,
        init_noise_std=1.0,
        clip_action=100.0,
        squash_mode="clip",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        mlp_encoder_dims=[256, 128, 64],
        activation="elu",
        trans_hidden_dims=[256, 128],  # not used in this policy, but required for the config to be valid
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        lr_max=1.0e-3,
        lr_min=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="Adam",
    )

# =============================
#         RNN POLICIES
# =============================


@configclass
class Go1RoughRnnRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100  # Checkpointing interval
    experiment_name = "TAR_workspace"
    empirical_normalization = True
    policy = RslRlRnnPpoPolicyCfg(
        class_name="ActorCriticRnnDblEnc",
        init_noise_std=1.0,
        clip_action=100.0,
        squash_mode="clip",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # RNN specific
        arch_type="augmented",
        rnn_type="lstm",
        rnn_hidden_dims=[512],
        rnn_out_features=20,  # latent_dims of augmented lstm
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        lr_max=1.0e-3,
        lr_min=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="Adam",
    )


# =============================
#         TAR POLICIES
# =============================


@configclass
class Go1RoughPpoTarRunnerCfg(Go1RoughPpoRunnerCfg):
    policy = RslRlPpoTarPolicyCfg(
        class_name="ActorCriticTar",
        num_hist=10,
        num_hist_short=4,
        latent_dims=45,
        init_noise_std=1.0,
        clip_action=100.0,
        squash_mode="clip",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        mlp_encoder_dims=[256, 128, 64],
        activation="elu",
        trans_hidden_dims=[64],
    )
    algorithm = RslRlPpoTarAlgorithmCfg(
        class_name="PPOTAR",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        lr_max=1.0e-3,
        lr_min=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="Adam",
        aux_loss_coef=[1.0],
    )


@configclass
class Go1RoughRnnTarRunnerCfg(Go1RoughRnnRunnerCfg):
    policy = RslRlRnnTarPolicyCfg(
        class_name="ActorCriticTarRnn",
        init_noise_std=1.0,
        num_hist=5,
        num_hist_short=4,
        latent_dims=45,
        clip_action=100.0,
        squash_mode="clip",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # RNN specific
        arch_type="integrated",
        rnn_type="lstm",
        rnn_hidden_dims=[256],
        rnn_out_features=0,  # latent_dims of augmented lstm only, default is 0
        # tar settings
        trans_hidden_dims=[64],
    )
    algorithm = RslRlPpoTarAlgorithmCfg(
        class_name="PPOTAR",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        lr_max=1.0e-3,
        lr_min=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="Adam",
        aux_loss_coef=[1.0],
    )


@configclass
class Go1RoughTcnTarRunnerCfg(Go1RoughPpoTarRunnerCfg):
    policy = RslRlPpoTarPolicyCfg(
        class_name="ActorCriticTarTcn",
        num_hist=50,
        num_hist_short=4,
        latent_dims=45,
        init_noise_std=1.0,
        clip_action=100.0,
        squash_mode="clip",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        mlp_encoder_dims=[256, 128, 64],
        activation="elu",
        trans_hidden_dims=[64],
    )
    algorithm = RslRlPpoTarAlgorithmCfg(
        class_name="PPOTAR",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        lr_max=1.0e-3,
        lr_min=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="Adam",
        aux_loss_coef=[1.0],
    )

# =============================
#         SLR POLICIES
# =============================


@configclass
class Go1RoughPpoSlrRunnerCfg(Go1RoughPpoRunnerCfg):
    policy = RslRlPpoSlrPolicyCfg(
        class_name="ActorCriticMlpSlr",
        num_hist=10,
        latent_dims=20,
        init_noise_std=1.0,
        clip_action=100.0,
        squash_mode="clip",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        mlp_encoder_dims=[256, 128, 64],
        activation="elu",
        trans_hidden_dims=[256, 128],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOSLR",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        lr_max=1.0e-3,
        lr_min=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="Adam",
    )


# =============================
#         HIM POLICIES
# =============================
@configclass
class Go1PpoHimRunnerCfg(Go1RoughPpoRunnerCfg):
    policy = RslRlPpoPolicyCfg(  # teacher policy if distil is used
        class_name="ActorCriticHIM",
        init_noise_std=1.0,
        clip_action=100.0,
        squash_mode="clip",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOHIM",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        lr_max=1.0e-3,
        lr_min=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        optimizer="Adam",
    )
