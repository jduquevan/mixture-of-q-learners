
"""
When test_during_training is set to True, an additional number of parallel test environments are used to evaluate the agent during training using greedy actions,
but not for training purposes. Stopping training for evaluation can be very expensive, as an episode in Atari can last for hundreds of thousands of steps.
"""

import copy
import time
import os
import jax
import flax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Sequence, Tuple, Callable, Literal, Optional

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
    fire_countdown: jnp.array


class JaxLogEnvPoolWrapper(gym.Wrapper):
    def __init__(self, env, reset_info=True, async_mode=True, force_fire_on_reset=True, fire_action=1, fire_steps=1):
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
        
        self.force_fire_on_reset = force_fire_on_reset
        self.fire_action = int(fire_action)
        self.fire_steps = int(fire_steps)

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
            jnp.full((self.num_envs,), self.fire_steps, dtype=jnp.int32)
            if self.force_fire_on_reset else
            jnp.zeros((self.num_envs,), dtype=jnp.int32),)
        return observations, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        if self.force_fire_on_reset:
            fa = jnp.asarray(self.fire_action, dtype=action.dtype)
            override = state.fire_countdown > 0    
            action = jnp.where(override, fa, action)

        new_handle, (observations, rewards, dones, infos) = self.step_f(
            state.handle, action
        )

        new_episode_return = state.episode_returns + infos["reward"]
        new_episode_length = state.episode_lengths + 1
        terminated = infos["terminated"] + infos["TimeLimit.truncated"]

        if self.force_fire_on_reset:
            next_fire = jnp.where(
                terminated.astype(jnp.bool_),
                jnp.asarray(self.fire_steps, dtype=jnp.int32),
                jnp.maximum(state.fire_countdown - 1, 0),
            )
        else:
            next_fire = state.fire_countdown

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
            fire_countdown=next_fire,
        )

        if self.reset_info:
            elapsed_steps = infos["elapsed_step"]
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

class ActorCritic(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False       

    def setup(self):
        self.encoder = CNN(norm_type=self.norm_type, name="cnn")
        self.actor_head  = nn.Dense(self.action_dim, name="actor_head")
        self.critic_head = nn.Dense(1, name="critic_head")
    
    def actor(self, features, train: bool):
        return self.actor_head(features)

    def critic(self, features, train: bool):
        return self.critic_head(features)

    def encode(self, x, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0
        return self.encoder(x, train=train)

    def __call__(self, features, *, train: bool = False):
        return self.critic(features, train), self.actor(features, train)

@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    val: chex.Array
    log_p: chex.Array

class EncoderState(TrainState):
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
    ema_return: jnp.ndarray = 0.0
    mix_counter: jnp.ndarray = 0

class CriticState(TrainState):
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
    ema_return: jnp.ndarray = 0.0
    ema_train_return: jnp.ndarray = 0.0
    mix_counter: jnp.ndarray = 0

class ActorState(TrainState):
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0
    ema_return: jnp.ndarray = 0.0
    ema_train_return: jnp.ndarray = 0.0
    mix_counter: jnp.ndarray = 0

@jax.jit
def _update_ema(prev_ema, new_value, alpha):
    return alpha * new_value + (1 - alpha) * prev_ema

def _policy_from_logits(logits: jax.Array):
    """Return categorical π(a|s), log π(a|s) and entropy H[π]"""
    log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    probs     = jnp.exp(log_probs)
    entropy   = -jnp.sum(probs * log_probs, axis=-1)
    return probs, log_probs, entropy    

def preprocess_agent_transition(x, rng, config):
    # x: (num_steps, num_envs, ...)
    flattened = x.reshape(-1, *x.shape[2:])
    shuffled = jax.random.permutation(rng, flattened)
    return shuffled.reshape(config["NUM_MINIBATCHES"], -1, *x.shape[2:])

def preprocess_transitions_per_agent(x, rng, config):
    # x: (num_steps, total_envs, ...), with total_envs = NUM_AGENTS * NUM_ENVS.
    num_steps = x.shape[0]
    total_envs = x.shape[1]
    num_agents = config["NUM_AGENTS"]
    num_envs = config["NUM_ENVS"]
    # First, transpose to (total_envs, num_steps, ...)
    x = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
    # Then, reshape total_envs into (num_agents, num_envs)
    x = x.reshape((num_agents, num_envs, num_steps) + x.shape[2:])
    # Finally, transpose to (num_agents, num_steps, num_envs, ...)
    x = jnp.transpose(x, (0, 2, 1) + tuple(range(3, x.ndim)))
    # Split rng for each agent
    rngs = jax.random.split(rng, num_agents)
    return jax.vmap(lambda x_agent, r: preprocess_agent_transition(x_agent, r, config), in_axes=(0, 0))(x, rngs)

def compute_agent_metric(metric, config):
    num_agents = config["NUM_AGENTS"]
    num_steps = config["NUM_STEPS"]
    num_envs = config["NUM_ENVS"]
    test_envs = config.get("TEST_ENVS", 0)
    envs_per_agent = num_envs + test_envs

    metric = jnp.transpose(metric, (1, 0) + tuple(range(2, metric.ndim)))
    metric = metric.reshape((num_agents, envs_per_agent, num_steps) + metric.shape[2:])
    metric = metric[:, num_envs:]
    metric = jnp.transpose(metric, (2, 0, 1) + tuple(range(3, metric.ndim)))
    return jnp.mean(metric, axis=(0, 2))

def compute_agent_metrics(metrics, config):
    return jax.tree_util.tree_map(
        lambda m: compute_agent_metric(m, config) if isinstance(m, jnp.ndarray) and m.ndim >= 2 else m,
        metrics,
    )

def initialize_agents(config, env, rng, network):
    num_agents = config["NUM_AGENTS"]
    dummy_x = jnp.zeros((1, *env.single_observation_space.shape))
    dummy_feat = jnp.zeros((1, 512))

    encoder_vars = network.init(rng, dummy_x, train=True, method=ActorCritic.encode)
    encoder_params = encoder_vars["params"]

    critic_vars = network.init(rng, dummy_feat, train=True, method=ActorCritic.critic)
    critic_params = critic_vars["params"]

    actor_params = jax.vmap(lambda k:
        network.init(k, dummy_feat, train=False, method=ActorCritic.actor)["params"]
    )(jax.random.split(rng, config["NUM_AGENTS"]))


    lr_sched = optax.linear_schedule(
        init_value=config["LR"],
        end_value=1e-20,
        transition_steps=config["NUM_UPDATES_DECAY"]
                       * config["NUM_MINIBATCHES"]
                       * config["NUM_EPOCHS"],
    ) if config.get("LR_LINEAR_DECAY", False) else config["LR"]

    tx = lambda lr: optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.radam(lr),
    )

    encoder_states = EncoderState.create(
        apply_fn=network.apply,
        params=encoder_params,
        tx=tx(lr_sched),
    )

    critic_states = CriticState.create(
        apply_fn=network.apply, 
        params=critic_params,
        tx=tx(lr_sched),
    )

    critic_states = critic_states.replace(
        ema_return=jnp.zeros((config["NUM_AGENTS"],), dtype=jnp.float32),
        ema_train_return=jnp.zeros((config["NUM_AGENTS"],), dtype=jnp.float32),
    )

    def _index(tree, i):
        return jax.tree_util.tree_map(lambda x: x[i], tree)
    
    def make_head_states(i):
        return (
            ActorState.create(
                apply_fn=network.apply, 
                params=_index(actor_params,  i),
                tx=tx(lr_sched),
            )
        )

    actor_states = jax.vmap(make_head_states)(jnp.arange(num_agents))
    return critic_states, actor_states, encoder_states

def _compute_gae(values, rewards, dones, gamma, lam):
    T, N = rewards.shape

    def scan(carry, t):
        gae, adv = carry                       # gae: (N,)
        delta = rewards[t] + gamma * (1 - dones[t]) * values[t+1] - values[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        adv   = adv.at[t].set(gae)
        return (gae, adv), None

    advantages = jnp.zeros_like(rewards)       # (T, N)
    init_gae   = jnp.zeros_like(rewards[0])
    (_, advantages), _ = jax.lax.scan(
        scan,
        (init_gae, advantages),
        jnp.arange(T-1, -1, -1)
    )
    returns = advantages + values[:-1]
    return advantages, returns

def make_train(config):
    num_envs = config["NUM_ENVS"]
    num_agents = config["NUM_AGENTS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // num_envs
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // num_envs
    )

    assert (config["NUM_STEPS"] * num_envs) % config[
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
    
    envs_per_agent = num_envs + config["TEST_ENVS"] if config.get("TEST_DURING_TRAINING", False) else num_envs
    total_envs = (envs_per_agent * num_agents)
    env = make_env(total_envs)

    score_min   = jnp.asarray(env.env_random_score, dtype=jnp.float32)
    score_denom = jnp.asarray(env.env_human_score - env.env_random_score, dtype=jnp.float32)

    # here reset must be out of vmap and jit
    init_obs, env_state = env.reset()

    def train(rng):
        original_seed = rng[0]

        # INIT NETWORK AND OPTIMIZER
        network = ActorCritic(
            action_dim = env.single_action_space.n,
            norm_type  = config["NORM_TYPE"],
            norm_input = config.get("NORM_INPUT", False),
        )

        rng, _rng = jax.random.split(rng)
        critic_train_states, actor_train_states, encoder_train_states = initialize_agents(config, env, rng, network)

        # TRAINING LOOP
        def _update_step(runner_state, unused):
            num_envs = config["NUM_ENVS"]
            envs_per_agent = num_envs + config["TEST_ENVS"] if config.get("TEST_DURING_TRAINING", False) else num_envs
            critic_train_states, actor_train_states, encoder_train_states, expl_state, test_metrics, rng = runner_state
            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                features = network.apply(
                    {"params": encoder_train_states.params},
                    last_obs,
                    train=False,
                    method=ActorCritic.encode
                )
                vals = network.apply(
                    {"params": critic_train_states.params},
                    features,
                    train=False,
                    method=ActorCritic.critic
                )
                logits = jax.vmap(
                    lambda ts, features: network.apply(
                        {"params": ts.params},
                        features,
                        train=False,
                        method=ActorCritic.actor)
                )(
                    actor_train_states,
                    features.reshape((config["NUM_AGENTS"], envs_per_agent, *features.shape[1:]))
                )
                
                rng_a_batched = jnp.repeat(jnp.expand_dims(rng_a, axis=0), config["NUM_AGENTS"], axis=0)
                _rngs = jax.vmap(lambda key: jax.random.split(key, config["NUM_ENVS"] + config["TEST_ENVS"]))(rng_a_batched)
                new_action = jax.vmap(                                           # ← over agents
                    lambda l_row, k_row: jnp.concatenate(
                        [                                                        # training envs
                        jax.vmap(lambda l, k: jax.random.categorical(k, l))
                            (l_row[:num_envs], k_row[:num_envs]),
                        jnp.argmax(l_row[num_envs:], axis=-1)                  # test envs
                        ], axis=0)
                )(logits, _rngs)

                probs, log_ps, _ = _policy_from_logits(logits)
                log_ps = jnp.take_along_axis(log_ps, new_action[..., None], -1).reshape((config["NUM_AGENTS"]*envs_per_agent, 1))
                new_action = new_action.reshape((config["NUM_AGENTS"]*envs_per_agent, *new_action.shape[2:]))

                new_obs, new_env_state, reward, new_done, info = env.step(
                    env_state, new_action
                )

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    val=vals,
                    log_p=log_ps,
                )
                return (new_obs, new_env_state, rng), (transition, info)

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
                def filter_transitions(x, config):
                    num_agents = config["NUM_AGENTS"]
                    num_steps = config["NUM_STEPS"]
                    envs_per_agent = num_envs + config["TEST_ENVS"] if config.get("TEST_DURING_TRAINING", False) else num_envs
                    x = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
                    x = x.reshape((num_agents, envs_per_agent, num_steps) + x.shape[2:])
                    x = x[:, : -config["TEST_ENVS"]]
                    x = x.reshape((num_agents*num_envs, num_steps) + x.shape[3:])
                    return jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))

                transitions = jax.tree.map(
                    lambda x: filter_transitions(x, config), transitions
                )

            critic_train_states = critic_train_states.replace(
                timesteps=critic_train_states.timesteps
                + config["NUM_STEPS"] * config["NUM_AGENTS"] 
            )
            actor_train_states = actor_train_states.replace(
                timesteps=actor_train_states.timesteps
                + config["NUM_STEPS"] * config["NUM_AGENTS"] 
            )
            encoder_train_states = encoder_train_states.replace(
                timesteps=encoder_train_states.timesteps
                + config["NUM_STEPS"] * config["NUM_AGENTS"] 
            )
            
            next_obs = transitions.next_obs.reshape(config["NUM_STEPS"], config["NUM_AGENTS"], config["NUM_ENVS"], *transitions.next_obs.shape[2:])
            next_obs_last = next_obs[-1]
            features = network.apply(
                {"params": encoder_train_states.params},
                next_obs_last.reshape((config["NUM_AGENTS"]*num_envs, *next_obs_last.shape[2:])),
                train=False,
                method=ActorCritic.encode
            )
            last_next_value = network.apply(
                {"params": critic_train_states.params},
                features,
                train=False,
                method=ActorCritic.critic
            ).squeeze(-1)
            
            values = jnp.concatenate([transitions.val.squeeze(-1),last_next_value.reshape((1, config["NUM_AGENTS"]*config["NUM_ENVS"]))], axis=0)

            gae_advantages, lambda_targets  = _compute_gae(
                values   = values,                 # (T+1, N)
                rewards  = transitions.reward,     # (T  , N)
                dones    = transitions.done,       # (T  , N)
                gamma    = config["GAMMA"],
                lam      = config["LAMBDA"],
            )
            
            gae_advantages = (gae_advantages - gae_advantages.mean()) / (gae_advantages.std() + 1e-8)
            
            def _learn_epoch(carry, _):
                critic_train_states, actor_train_states, encoder_train_states, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(lambda x: preprocess_transitions_per_agent(x, _rng, config), transitions)
                minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), minibatches)
                targets = jax.tree_map(lambda x: preprocess_transitions_per_agent(x, _rng, config), lambda_targets)
                targets = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), targets)
                advantages = jax.tree_map(lambda x: preprocess_transitions_per_agent(x, _rng, config), gae_advantages)
                advantages = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), advantages)

                def _learn_phase(carry, minibatch_and_target):
                    critic_train_states, actor_train_states, encoder_train_states, rng = carry
                    minibatch, target, advantage = minibatch_and_target

                    A = config["NUM_AGENTS"]
                    B = minibatch.obs.shape[1] 
                    S = min(config.get("DIV_SAMPLES", 64), A * B)
                    agent_ids = jnp.arange(A, dtype=jnp.int32) 

                    rng, rng_div = jax.random.split(rng)
                    obs_pool = minibatch.obs.reshape(A * B, *minibatch.obs.shape[2:])
                    idx = jax.random.choice(rng_div, obs_pool.shape[0], (S,), replace=False)
                    div_obs = obs_pool[idx]

                    # encode once (no grad to encoder for diversity)
                    div_feats = network.apply({"params": encoder_train_states.params},
                                            div_obs, train=False, method=ActorCritic.encode)
                    div_feats = jax.lax.stop_gradient(div_feats)

                    logits_all = jax.vmap(lambda ts:
                        network.apply({"params": ts.params}, div_feats,
                                    train=False, method=ActorCritic.actor)
                    )(actor_train_states)
                    probs_all, logprobs_all, _ = _policy_from_logits(logits_all)
                    logprobs_all = jax.lax.stop_gradient(logprobs_all)

                    div_coef = config.get("DIVERSITY_COEFF", 0.0) * (A > 1)

                    def agent_loss_and_update(cts, ats, minibatch, target, ets, logprobs_all, adv_i, i):
                        # Critic loss and update
                        def _critic_loss_fn(critic_params, encoder_params):
                            features = network.apply(
                                {"params": encoder_params},
                                minibatch.obs, train=True,
                                method=ActorCritic.encode
                            )
                            pred_val = network.apply(
                                {"params": critic_params},
                                features, train=True,
                                method=ActorCritic.critic
                            )
                            pred_val = pred_val.squeeze(-1)
                            loss = 0.5 * jnp.mean((pred_val - target) ** 2)
                            return loss
                        critic_loss, grads = jax.value_and_grad(_critic_loss_fn, argnums=(0, 1))(cts.params, ets.params)
                        critic_grad = grads[0]
                        encoder_c_grad = grads[1]

                        # Actor loss and update
                        def _actor_loss_fn(actor_params, encoder_params):
                            features  = network.apply(
                                {"params": encoder_params},
                                minibatch.obs, train=True,
                                method=ActorCritic.encode
                            )
                            logits = network.apply(
                                {"params": actor_params},
                                features, train=True,
                                method=ActorCritic.actor
                            )
                            probs, logp, entropy = _policy_from_logits(logits)
                            
                            logp_a = jnp.take_along_axis(logp, minibatch.action[..., None], -1).squeeze(-1)
                            ratio  = jnp.exp(logp_a - minibatch.log_p.squeeze(-1))

                            clip_eps = config["CLIP_EPS"]
                            unclipped = ratio * adv_i
                            clipped   = jnp.clip(ratio, 1-clip_eps, 1+clip_eps) * adv_i
                            pg_loss   = -jnp.mean(jnp.minimum(unclipped, clipped))

                            ent_loss = -jnp.mean(entropy)
                            ent_coeff = config["ENT_COEFF"]

                            # Exempt always 1 agent (base policy)
                            is_exempt = (i == 0)
                            logits_div = network.apply({"params": actor_params},
                                       div_feats, train=True, method=ActorCritic.actor)
                            probs_div, logprobs_div, _ = _policy_from_logits(logits_div)  # [S,Act]

                            sum_logp_others  = jnp.sum(logprobs_all, axis=0) - logprobs_all[i]
                            mean_logp_others = sum_logp_others / jnp.maximum(1, A - 1)
                            ce_mean = -jnp.mean(jnp.sum(probs_div * mean_logp_others, axis=-1))


                            div_coef_i = jnp.where(is_exempt, 0.0, div_coef)

                            loss = pg_loss + ent_coeff * ent_loss - div_coef_i * ce_mean
                            aux  = {"entropy": jnp.mean(entropy), "kl": ce_mean}
                            return loss, aux

                        (actor_loss, aux), grads = jax.value_and_grad(_actor_loss_fn, argnums=(0, 1), has_aux=True)(ats.params, ets.params)
                        entropies = aux["entropy"]
                        kl = aux["kl"]
                        encoder_a_grad = grads[1]

                        actor_updates, new_actor_opt_state = ats.tx.update(grads[0], ats.opt_state, ats.params)
                        new_actor_params = optax.apply_updates(ats.params, actor_updates)
                        ats = ats.replace(params=new_actor_params, opt_state=new_actor_opt_state, grad_steps=ats.grad_steps+1)
                        
                        return critic_loss, actor_loss, entropies, kl, ats, critic_grad, encoder_c_grad, encoder_a_grad
                    
                    critic_loss, actor_loss, entropies, kls, new_actor_train_states, critic_grads, encoder_c_grad, encoder_a_grad = jax.vmap(agent_loss_and_update, in_axes=(None, 0,0,0, None, None,0,0))(
                        critic_train_states, actor_train_states, minibatch, target, encoder_train_states, logprobs_all, advantage, agent_ids
                    )

                    def mean_over_agents(tree): return jax.tree_map(lambda x: x.mean(0), tree)
                    critic_grad = mean_over_agents(critic_grads)
                    encoder_grad = jax.tree_map(lambda gc, ga: mean_over_agents(gc) + mean_over_agents(ga), encoder_c_grad, encoder_a_grad)
                    critic_updates, new_critic_opt_state = critic_train_states.tx.update(critic_grad, critic_train_states.opt_state, critic_train_states.params)
                    encoder_updates, new_encoder_opt_state = encoder_train_states.tx.update(encoder_grad, encoder_train_states.opt_state, encoder_train_states.params)
                    new_critic_train_states = critic_train_states.replace(
                        params=optax.apply_updates(critic_train_states.params, critic_updates),
                        opt_state=new_critic_opt_state,
                        grad_steps=critic_train_states.grad_steps + 1
                    )
                    new_encoder_train_states = encoder_train_states.replace(
                        params=optax.apply_updates(encoder_train_states.params, encoder_updates),
                        opt_state=new_encoder_opt_state,
                        grad_steps=encoder_train_states.grad_steps + 1
                    )

                    return (new_critic_train_states, new_actor_train_states, new_encoder_train_states, rng), (critic_loss, actor_loss, entropies, kls)
        
                rng, _rng = jax.random.split(rng)
                (critic_train_states, actor_train_states, encoder_train_states, rng), (critic_losses, actor_losses, entropies, kls) = jax.lax.scan(_learn_phase, (critic_train_states, actor_train_states, encoder_train_states, rng), (minibatches, targets, advantages))
                
                mean_kls = jnp.mean(kls, axis=0)
                mean_entropies = jnp.mean(entropies, axis=0)
                mean_critic_losses = jnp.mean(critic_losses, axis=0)
                mean_actor_losses = jnp.mean(actor_losses, axis=0)
                return (critic_train_states, actor_train_states, encoder_train_states, rng), (mean_critic_losses, mean_actor_losses, mean_entropies, mean_kls)

            rng, _rng = jax.random.split(rng)
            (critic_train_states, actor_train_states, encoder_train_states, rng), (critic_losses, actor_losses, entropies, kls) = jax.lax.scan(
                _learn_epoch, (critic_train_states, actor_train_states, encoder_train_states, rng), None, config["NUM_EPOCHS"]
            )
            critic_train_states = critic_train_states.replace(n_updates=critic_train_states.n_updates + 1)
            actor_train_states = actor_train_states.replace(n_updates=actor_train_states.n_updates + 1)
            encoder_train_states = encoder_train_states.replace(n_updates=encoder_train_states.n_updates + 1)

            mean_critic_losses = jnp.mean(critic_losses, axis=0)
            mean_actor_losses = jnp.mean(actor_losses, axis=0)
            mean_entropies = jnp.mean(entropies, axis=0)
            mean_kls = jnp.mean(kls, axis=0)
            env_metrics = compute_agent_metrics(infos, config)

            num_agents = config["NUM_AGENTS"]
            num_steps  = config["NUM_STEPS"]
            num_envs   = config["NUM_ENVS"]
            test_envs  = config.get("TEST_ENVS", 0)
            envs_per_agent = num_envs + test_envs

            v = infos["returned_episode_returns"]      # (T, total_envs)
            m = infos["returned_episode"]              # (T, total_envs) in {0,1}

            v_e = jnp.transpose(v, (1, 0)).reshape((num_agents, envs_per_agent, num_steps))
            m_e = jnp.transpose(m, (1, 0)).reshape((num_agents, envs_per_agent, num_steps))

            v_tr = jnp.transpose(v_e[:, :num_envs], (2, 0, 1))   # (T, A, num_envs)
            m_tr = jnp.transpose(m_e[:, :num_envs], (2, 0, 1))   # (T, A, num_envs)

            num = jnp.sum(v_tr * m_tr, axis=(0, 2))
            den = jnp.maximum(jnp.sum(m_tr, axis=(0, 2)), 1.0)
            has_eps = den > 0
            train_raw_returns = num / den         

            train_norm_returns = (train_raw_returns - score_min) / score_denom
            train_norm_returns = jnp.where(has_eps, train_norm_returns, critic_train_states.ema_train_return)

            ema_target = env_metrics["normalized_returned_episode_returns"] 
            critic_train_states = critic_train_states.replace(
                ema_return = _update_ema(critic_train_states.ema_return, ema_target, config["EMA_ALPHA"]),
                ema_train_return = _update_ema(critic_train_states.ema_train_return, train_norm_returns, config["EMA_ALPHA"])
            )
            actor_train_states  = actor_train_states.replace(
                ema_return = critic_train_states.ema_return,
                ema_train_return = critic_train_states.ema_train_return
            )
            critic_train_states = critic_train_states.replace(mix_counter = critic_train_states.mix_counter + 1)
            actor_train_states  = actor_train_states.replace(mix_counter  = actor_train_states.mix_counter  + 1)

            def _merge_params(stacked_params, weights):
                return jax.tree_util.tree_map(
                    lambda x: jnp.tensordot(weights, x, axes=1),   # (A, …) → (…)
                    stacked_params,
                )

            def _merge_tree_weighted(stacked_tree, w):
                def merge_leaf(x):
                    if jnp.issubdtype(x.dtype, jnp.floating):
                        return jnp.tensordot(w, x, axes=1)   # weighted avg of EMA-like floats
                    elif jnp.issubdtype(x.dtype, jnp.integer):
                        return jnp.max(x, axis=0)            # step counters etc.
                    else:
                        return x[0]                          # e.g., bools or objects
                return jax.tree_util.tree_map(merge_leaf, stacked_tree)


            def _repeat_pytree(tree, n):
                """Return a PyTree whose leaves are stacked copies of t along new axis 0."""
                return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n, axis=0), tree)

            do_mix = jnp.max(critic_train_states.mix_counter) >= config["MIXING_STEPS"]

            def _mix_states(_):
                A   = config["NUM_AGENTS"]
                tau = config.get("MIX_TAU", 2.0)

                w = jax.nn.softmax(critic_train_states.ema_train_return / tau)   # shape (A,)
                # w = jax.nn.softmax(jnp.ones(A,))

                merged_actor  = _merge_params(actor_train_states.params,   w)
                merged_actor_opt = _merge_tree_weighted(actor_train_states.opt_state, w)

                actor_pack  = _repeat_pytree(merged_actor,  A)
                actor_pack_opt    = _repeat_pytree(merged_actor_opt,    config["NUM_AGENTS"])

                # def _reset_opt(ts: TrainState):
                #     return ts.replace(opt_state = ts.tx.init(ts.params))

                # new_actor_states  = jax.vmap(_reset_opt)(
                #     actor_train_states.replace(
                #         params      = actor_pack,
                #         opt_state   = actor_pack_opt,
                #         mix_counter = jnp.zeros_like(actor_train_states.mix_counter),
                #     )
                # )

                new_actor_states =  actor_train_states.replace(
                    params      = actor_pack,
                    opt_state   = actor_pack_opt,
                    mix_counter = jnp.zeros_like(actor_train_states.mix_counter),
                )
                return new_actor_states

            def _skip_states(_):
                return actor_train_states

            actor_train_states = jax.lax.cond(
                do_mix, _mix_states, _skip_states, operand=None
            )
            tau = config.get("MIX_TAU", 2.0)
            mix_w = jax.nn.softmax(actor_train_states.ema_train_return / tau)

            metrics = {}
            for i in range(config["NUM_AGENTS"]):
                metrics[f"agent_{i}/train_return"] = train_raw_returns[i]
                metrics[f"agent_{i}/mix_weight_from_ema"] = mix_w[i]
                metrics[f"agent_{i}/ema_return"] = actor_train_states.ema_return[i]
                metrics[f"agent_{i}/ema_train_return"] = actor_train_states.ema_train_return[i]
                metrics[f"agent_{i}/env_step"] = actor_train_states.timesteps[i]
                metrics[f"agent_{i}/update_steps"] = actor_train_states.n_updates[i]
                metrics[f"agent_{i}/env_frame"] = actor_train_states.timesteps[i] * env.single_observation_space.shape[0]
                metrics[f"agent_{i}/grad_steps"] = actor_train_states.grad_steps[i]
                metrics[f"agent_{i}/td_loss"] = mean_critic_losses[i]
                metrics[f"agent_{i}/policy_loss"] = mean_actor_losses[i]
                metrics[f"agent_{i}/entropy"] = mean_entropies[i]
                metrics[f"agent_{i}/relative_kl"] = mean_kls[i]
                
                for k, v in env_metrics.items():
                    metrics[f"agent_{i}/{k}"] = v[i]
            
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
                    wandb.log(metrics, step=metrics["agent_0/update_steps"])

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (critic_train_states, actor_train_states, encoder_train_states, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        # test metrics not supported yet
        test_metrics = None

        # train
        rng, _rng = jax.random.split(rng)
        expl_state = (init_obs, env_state)
        runner_state = (critic_train_states, actor_train_states, encoder_train_states, expl_state, test_metrics, _rng)

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
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f"{config['ALG_NAME']}_{config['ENV_NAME']}",
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

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print(config)
    if config["DEBUG"]:
        jax.config.update("jax_disable_jit", True)
    print("Config:\n", OmegaConf.to_yaml(config))
    

    single_run(config)


if __name__ == "__main__":
    main()
