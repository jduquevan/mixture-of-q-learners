"""
When test_during_training is set to True, an additional number of parallel test environments are used to evaluate the agent during training using greedy actions,
but not for training purposes. Stopping training for evaluation can be very expensive, as an episode in Atari can last for hundreds of thousands of steps.
"""

import copy
import time
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

from flax import struct
import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import wandb

import envpool
import gym
import numpy as np
from packaging import version
from functools import partial

is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")
assert is_legacy_gym, "Current version supports only gym<=0.23.1"

# (random,human)
ATARI_SCORES = {
    "Alien-v5": (227.8, 7127.7),
    "Amidar-v5": (5.8, 1719.5),
    "Assault-v5": (222.4, 742.0),
    "Asterix-v5": (210.0, 8503.3),
    "Asteroids-v5": (719.1, 47388.7),
    "Atlantis-v5": (12850.0, 29028.1),
    "BankHeist-v5": (14.2, 753.1),
    "BattleZone-v5": (2360.0, 37187.5),
    "BeamRider-v5": (363.9, 16926.5),
    "Berzerk-v5": (123.7, 2630.4),
    "Bowling-v5": (23.1, 160.7),
    "Boxing-v5": (0.1, 12.1),
    "Breakout-v5": (1.7, 30.5),
    "Centipede-v5": (2090.9, 12017.0),
    "ChopperCommand-v5": (811.0, 7387.8),
    "CrazyClimber-v5": (10780.5, 35829.4),
    "Defender-v5": (2874.5, 18688.9),
    "DemonAttack-v5": (152.1, 1971.0),
    "DoubleDunk-v5": (-18.6, -16.4),
    "Enduro-v5": (0.0, 860.5),
    "FishingDerby-v5": (-91.7, -38.7),
    "Freeway-v5": (0.0, 29.6),
    "Frostbite-v5": (65.2, 4334.7),
    "Gopher-v5": (257.6, 2412.5),
    "Gravitar-v5": (173.0, 3351.4),
    "Hero-v5": (1027.0, 30826.4),
    "IceHockey-v5": (-11.2, 0.9),
    "Jamesbond-v5": (29.0, 302.8),
    "Kangaroo-v5": (52.0, 3035.0),
    "Krull-v5": (1598.0, 2665.5),
    "KungFuMaster-v5": (258.5, 22736.3),
    "MontezumaRevenge-v5": (0.0, 4753.3),
    "MsPacman-v5": (307.3, 6951.6),
    "NameThisGame-v5": (2292.3, 8049.0),
    "Phoenix-v5": (761.4, 7242.6),
    "Pitfall-v5": (-229.4, 6463.7),
    "Pong-v5": (-20.7, 14.6),
    "PrivateEye-v5": (24.9, 69571.3),
    "Qbert-v5": (163.9, 13455.0),
    "Riverraid-v5": (1338.5, 17118.0),
    "RoadRunner-v5": (11.5, 7845.0),
    "Robotank-v5": (2.2, 11.9),
    "Seaquest-v5": (68.4, 42054.7),
    "Skiing-v5": (-17098.1, -4336.9),
    "Solaris-v5": (1236.3, 12326.7),
    "SpaceInvaders-v5": (148.0, 1668.7),
    "StarGunner-v5": (664.0, 10250.0),
    "Surround-v5": (-10.0, 6.5),
    "Tennis-v5": (-23.8, -8.3),
    "TimePilot-v5": (3568.0, 5229.2),
    "Tutankham-v5": (11.4, 167.6),
    "UpNDown-v5": (533.4, 11693.2),
    "Venture-v5": (0.0, 1187.5),
    "VideoPinball-v5": (16256.9, 17667.9),
    "WizardOfWor-v5": (563.5, 4756.5),
    "YarsRevenge-v5": (3092.9, 54576.9),
    "Zaxxon-v5": (32.5, 9173.3),
}


@struct.dataclass
class LogEnvState:
    handle: jnp.array
    lives: jnp.array
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


class JaxLogEnvPoolWrapper(gym.Wrapper):
    def __init__(self, env, reset_info=True, async_mode=True):
        super(JaxLogEnvPoolWrapper, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.env_name = env.name
        self.env_random_score, self.env_human_score = ATARI_SCORES[self.env_name]
        # get if the env has lives
        self.has_lives = False
        env.reset()
        info = env.step(np.zeros(self.num_envs, dtype=int))[-1]
        if info["lives"].sum() > 0:
            self.has_lives = True
            print("env has lives")
        self.reset_info = reset_info
        handle, recv, send, step = env.xla()
        self.init_handle = handle
        self.send_f = send
        self.recv_f = recv
        self.step_f = step

    def reset(self, **kwargs):
        observations = super(JaxLogEnvPoolWrapper, self).reset(**kwargs)

        env_state = LogEnvState(
            jnp.array(self.init_handle),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
            jnp.zeros(self.num_envs, dtype=jnp.float32),
        )
        return observations, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        new_handle, (observations, rewards, dones, infos) = self.step_f(
            state.handle, action
        )

        new_episode_return = state.episode_returns + infos["reward"]
        new_episode_length = state.episode_lengths + 1
        state = state.replace(
            handle=new_handle,
            episode_returns=(new_episode_return)
            * (1 - infos["terminated"])
            * (1 - infos["TimeLimit.truncated"]),
            episode_lengths=(new_episode_length)
            * (1 - infos["terminated"])
            * (1 - infos["TimeLimit.truncated"]),
            returned_episode_returns=jnp.where(
                infos["terminated"] + infos["TimeLimit.truncated"],
                new_episode_return,
                state.returned_episode_returns,
            ),
            returned_episode_lengths=jnp.where(
                infos["terminated"] + infos["TimeLimit.truncated"],
                new_episode_length,
                state.returned_episode_lengths,
            ),
        )

        if self.reset_info:
            elapsed_steps = infos["elapsed_step"]
            terminated = infos["terminated"] + infos["TimeLimit.truncated"]
            infos = {}
        normalize_score = lambda x: (x - self.env_random_score) / (
            self.env_human_score - self.env_random_score
        )
        infos["returned_episode_returns"] = state.returned_episode_returns
        infos["normalized_returned_episode_returns"] = normalize_score(
            state.returned_episode_returns
        )
        infos["returned_episode_lengths"] = state.returned_episode_lengths
        infos["elapsed_step"] = elapsed_steps
        infos["returned_episode"] = terminated

        return (
            observations,
            state,
            rewards,
            dones,
            infos,
        )


class CNN(nn.Module):
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        return x

class MaskedLSTMCell(nn.Module):
    features: int
    def setup(self):
        self.cell = nn.LSTMCell(features=self.features)

    def __call__(self, carry, inputs):
        (c, h) = carry                       # (B,H)
        x_t, m_t = inputs                    # x_t: (B,F), m_t: (B,)
        m_t = m_t[:, None]                   # (B,1)
        carry = (c * m_t, h * m_t)           # reset where episode ended
        return self.cell(carry, x_t)

class QNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False
    hidden_size: int = 256

    def setup(self):
        self.lstm_scan = nn.scan(
            MaskedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1, out_axes=1
        )(features=self.hidden_size)

    @nn.compact
    def __call__(self, x: jnp.ndarray, carry, train: bool, dones: jnp.ndarray | None = None):
        if x.ndim == 4:                        # (B,C,H,W) -> (B,H,W,C)
            x = jnp.transpose(x, (0, 2, 3, 1))
        elif x.ndim == 5:                      # (B,T,C,H,W) -> (B*T,H,W,C)
            B, T, C, H, W = x.shape
            x = jnp.transpose(x, (0, 1, 3, 4, 2))      # (B,T,H,W,C)
            x = x.reshape(B * T, H, W, C)
        
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        feat = CNN(norm_type=self.norm_type)(x, train)

        if dones is None:
            B = feat.shape[0]
            feat = feat[:, None, :]                          # (B,1,F)
            masks = jnp.ones((B, 1), jnp.float32)            # (B,1)
            (cT, hT), ys = self.lstm_scan(carry, (feat, masks))   # ys: (B,1,H)
            out = ys[:, 0, :]                                # (B,H)
        else:
            B = dones.shape[0]
            T = dones.shape[1]
            feat = feat.reshape(B, T, -1)                    # (B,T,F)
            masks = jnp.concatenate(
                [jnp.ones((B, 1), jnp.float32), 1.0 - dones[:, :-1].astype(jnp.float32)],
                axis=1
            )                                                # (B,T)
            (cT, hT), out = self.lstm_scan(carry, (feat, masks))  # out: (B,T,H)

        x = nn.Dense(self.action_dim)(out)
        return x, (cT, hT)


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array

@struct.dataclass
class RNNState:
    c: jnp.ndarray
    h: jnp.ndarray

class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
    
def expected_q_eps_greedy(q, eps):
    # q: [B, A]; eps: scalar or [B]
    # Your sampler chooses a random action uniformly over ALL actions with prob ε,
    # so  Eπ[Q] = (1-ε)*max(Q) + ε*mean(Q)
    max_q  = jnp.max(q, axis=-1)
    mean_q = jnp.mean(q, axis=-1)
    return (1.0 - eps) * max_q + eps * mean_q

def nearest_neighbor_upscale(frames, scale_factor=3):
    # frames shape: (1, 4, 400, 400)
    batch, time, height, width = frames.shape
    
    # Repeat along height and width dimensions
    upscaled = jnp.repeat(frames, scale_factor, axis=2)  # height
    upscaled = jnp.repeat(upscaled, scale_factor, axis=3)  # width
    
    return upscaled


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    def make_env(num_envs):
        env = envpool.make(
            config["ENV_NAME"],
            env_type="gym",
            num_envs=num_envs,
            seed=config["SEED"],
            **config["ENV_KWARGS"],
        )
        env.num_envs = num_envs
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.name = config["ENV_NAME"]
        env = JaxLogEnvPoolWrapper(env)
        return env

    total_envs = (
        (config["NUM_ENVS"] + config["TEST_ENVS"])
        if config.get("TEST_DURING_TRAINING", False)
        else config["NUM_ENVS"]
    )
    env = make_env(total_envs)

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions

    # here reset must be out of vmap and jit
    init_obs, env_state = env.reset()
    
    if config["SCALE_FACTOR"] > 1:
        init_obs = nearest_neighbor_upscale(init_obs, scale_factor=config["SCALE_FACTOR"])

    def train(rng):
        original_seed = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        h_0 = jnp.zeros((config["NUM_ENVS"] + config["TEST_ENVS"], config["HIDDEN_SIZE"]), jnp.float32)
        c_0 = jnp.zeros_like(h_0)

        network = QNetwork(
            action_dim=env.single_action_space.n,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )

        def create_agent(rng, c_0, h_0):
            init_x = jnp.zeros((1, *env.single_observation_space.shape))
            
            if config["SCALE_FACTOR"] > 1:
                init_x = nearest_neighbor_upscale(init_x, scale_factor=config["SCALE_FACTOR"])

            network_variables = network.init(rng, init_x, (c_0, h_0),  train=False)

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng, c_0, h_0)
        rnn_state = RNNState(c=c_0, h=h_0)

        # TRAINING LOOP
        def _update_step(runner_state, unused):
            train_state, expl_state, test_metrics, rng = runner_state
            _, _, initial_rnn_state = expl_state
            c_initial, h_initial = initial_rnn_state.c, initial_rnn_state.h

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rnn_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                c, h = rnn_state.c, rnn_state.h

                q_vals, (c_new, h_new) = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    (c, h),
                    train=False,
                )

                # different eps for each env
                _rngs = jax.random.split(rng_a, total_envs)
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                if config.get("TEST_DURING_TRAINING", False):
                    eps = jnp.concatenate((eps, jnp.zeros(config["TEST_ENVS"])))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    env_state, new_action
                )
                
                if config["SCALE_FACTOR"] > 1:
                    new_obs = nearest_neighbor_upscale(new_obs, scale_factor=config["SCALE_FACTOR"])

                # ── mask new carry --------------------------------------------------------
                mask = (1. - new_done).reshape((-1, 1))
                h_next = h_new * mask
                c_next = c_new * mask
                next_rnn_state = RNNState(h=h_next, c=c_next)

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, next_rnn_state, rng), (transition, info)

            # step the env
            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            if config.get("TEST_DURING_TRAINING", False):
                # remove testing envs
                transitions = jax.tree.map(
                    lambda x: x[:, : -config["TEST_ENVS"]], transitions
                )
                c_initial = c_initial[: -config["TEST_ENVS"], :]
                h_initial = h_initial[: -config["TEST_ENVS"], :]
                
            if config["CURIOSITY_REWARD"] == 'MSE_FRAME':
                mean_frame = jnp.mean(transitions.next_obs, axis=(0, 1, 2)) / 255.0
                curiosity_reward = jnp.abs(transitions.next_obs / 255.0 - mean_frame).mean(axis=(2, 3, 4))
                curiosity_reward = (curiosity_reward) / (curiosity_reward.std() + 1e-3)
                curiosity_reward = curiosity_reward * config["CURIOSITY_REWARD_SCALE"]
                avg_curiosity_reward = curiosity_reward.mean()
                std_curiosity_reward = curiosity_reward.std()
                transitions = transitions.replace(reward=transitions.reward + curiosity_reward)
            if config["CURIOSITY_REWARD"] == 'MSE_FRAME':
                mean_frame = jnp.mean(transitions.next_obs, axis=(0, 1, 2)) / 255.0
                curiosity_reward = jnp.softmax(transitions.q_val, axis=-1)
                curiosity_reward = (curiosity_reward) / (curiosity_reward.std() + 1e-3)
                curiosity_reward = curiosity_reward * config["CURIOSITY_REWARD_SCALE"]
                avg_curiosity_reward = curiosity_reward.mean()
                std_curiosity_reward = curiosity_reward.std()
                transitions = transitions.replace(reward=transitions.reward + curiosity_reward)
            elif config["CURIOSITY_REWARD"] == 'DISABLED':
                pass
            else:
                raise ValueError(f"Invalid curiosity reward type: {config['CURIOSITY_REWARD']}")

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            rnn_T = expl_state[2]  # RNN state AFTER T environment steps
            if config.get("TEST_DURING_TRAINING", False) and config["TEST_ENVS"] > 0:
                last_c = rnn_T.c[: -config["TEST_ENVS"], :]
                last_h = rnn_T.h[: -config["TEST_ENVS"], :]
            else:
                last_c = rnn_T.c
                last_h = rnn_T.h

            last_q_vals, (_, _) = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                transitions.next_obs[-1],
                (last_c, last_h),
                train=False,
            )
            if config["IS_SARSA"] is False:
                last_q = jnp.max(last_q_vals, axis=-1)
            else:
                last_q = expected_q_eps_greedy(last_q_vals, eps_scheduler(train_state.n_updates))

            def _compute_targets(last_q, q_vals, reward, done):
                def _get_target(lambda_returns_and_next_q, rew_q_done):
                    reward, q, done = rew_q_done
                    lambda_returns, next_q = lambda_returns_and_next_q
                    target_bootstrap = reward + config["GAMMA"] * (1 - done) * next_q
                    delta = lambda_returns - next_q
                    lambda_returns = (
                        target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                    )
                    lambda_returns = (1 - done) * lambda_returns + done * reward
                    if config["IS_SARSA"] is False:
                        next_q = jnp.max(q, axis=-1)
                    else:
                        next_q = expected_q_eps_greedy(q, eps_scheduler(train_state.n_updates))
                    return (lambda_returns, next_q), lambda_returns

                lambda_returns = reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                if config["IS_SARSA"] is False:
                    last_q = jnp.max(q_vals[-1], axis=-1)
                else:
                    last_q = expected_q_eps_greedy(q_vals[-1], eps_scheduler(train_state.n_updates))
                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_q),
                    jax.tree.map(lambda x: x[:-1], (reward, q_vals, done)),
                    reverse=True,
                )
                targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                return targets

            lambda_targets = _compute_targets(
                last_q, transitions.q_val, transitions.reward, transitions.done
            )

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):
                    train_state, rng = carry
                    minibatch, target, c_init, h_init = minibatch_and_target

                    def _loss_fn(params):
                        (q_vals, _), updates = network.apply(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            minibatch.obs,
                            (c_init, h_init),
                            train=True,
                            dones=minibatch.done,
                            mutable=["batch_stats"],
                        )  # (batch_size*2, num_actions)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, (updates, chosen_action_qvals)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    num_minibatches = config["NUM_MINIBATCHES"]

                    x = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
                    x = jax.random.permutation(rng, x)  # shuffle the transitions
                    x = x.reshape(
                        (num_minibatches, -1) + x.shape[1:] 
                    )  # num_mini_updates, batch_size/num_mini_updates, ...
                    return x

                def preprocess_hidden(x, rng):
                    num_minibatches = config["NUM_MINIBATCHES"]
                    x = jax.random.permutation(rng, x)
                    x = x.reshape(
                        (num_minibatches, -1) + x.shape[1:] 
                    )  # num_mini_updates, batch_size/num_mini_updates, ...
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...
                targets = jax.tree.map(
                    lambda x: preprocess_transition(x, _rng), lambda_targets
                )
                initial_h = jax.tree.map(
                    lambda x: preprocess_hidden(x, _rng), h_initial
                )
                initial_c = jax.tree.map(
                    lambda x: preprocess_hidden(x, _rng), c_initial
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets, initial_h, initial_c)
                )

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            if config.get("TEST_DURING_TRAINING", False):
                test_infos = jax.tree.map(lambda x: x[:, -config["TEST_ENVS"] :], infos)
                infos = jax.tree.map(lambda x: x[:, : -config["TEST_ENVS"]], infos)
                infos.update({"test_" + k: v for k, v in test_infos.items()})

            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "env_frame": train_state.timesteps
                * env.observation_space.shape[
                    0
                ],  # first dimension of the observation space is number of stacked frames
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
                "max_reward": transitions.reward.max(),
            }
            
            if config["CURIOSITY_REWARD"] == 'MSE_FRAME':
                metrics["curiosity_reward_avg"] = avg_curiosity_reward
                metrics["curiosity_reward_std"] = std_curiosity_reward

            metrics.update({k: v.mean() for k, v in infos.items()})
            if config.get("TEST_DURING_TRAINING", False):
                metrics.update({f"test_{k}": v.mean() for k, v in test_infos.items()})

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])
                    print(metrics)

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        # test metrics not supported yet
        test_metrics = None

        # train
        rng, _rng = jax.random.split(rng)
        expl_state = (init_obs, env_state, rnn_state)
        runner_state = (train_state, expl_state, test_metrics, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def single_run(config):
    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]

    wandb.init(
        id=config["RUN_ID"],
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config["RUN_ID"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    if config["NUM_SEEDS"] > 1:
        raise NotImplementedError("Vmapped seeds not supported yet.")
    else:
        outs = jax.jit(make_train(config))(rng)
    print(f"Took {time.time() - t0} seconds to complete.")

    # save params
    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f"{alg_name}_{env_name}_seed{config['SEED']}_config.yaml"
            ),
        )

        # assumes not vmpapped seeds
        params = model_state.params
        save_path = os.path.join(
            save_dir,
            f"{alg_name}_{env_name}_seed{config['SEED']}.safetensors",
        )
        save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {
        **default_config,
        **default_config["alg"],
    }  # merge the alg config with the main config

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config["alg"][k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])

        if config["NUM_SEEDS"] > 1:
            raise NotImplementedError("Vmapped seeds not supported yet.")
        else:
            outs = jax.jit(make_train(config))(rng)

    sweep_config = {
        "name": f"pqn_atari_{default_config['ENV_NAME']}",
        "method": "bayes",
        "metric": {
            "name": "test_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {"values": [0.0005, 0.0001, 0.00005]},
            "LAMBDA": {"values": [0.3, 0.6, 0.9]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    if config["DEBUG"]:
        jax.config.update("jax_disable_jit", True)
        import debugpy
        debugpy.listen(5678)
        print("Waiting for client to attach...")
        debugpy.wait_for_client()
        print("Client attached")
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
