
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

class ActorCritic(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False       

    def setup(self):
        self.encoder = CNN(norm_type=self.norm_type, name="cnn")
        self.actor_head  = nn.Dense(self.action_dim, name="actor_head")
        self.critic_head = nn.Dense(1, name="critic_head")
        self.encoder_bn = nn.BatchNorm(momentum=0.99, epsilon=1e-5, name="encoder_bn")
    
    def actor(self, x, train: bool):
        feat = self._encode(x, train)
        return self.actor_head(feat)

    def critic(self, x, train: bool):
        feat = self._encode(x, train)
        return self.critic_head(feat)

    def _encode(self, x, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))
        if self.norm_input:
            x = self.encoder_bn(x, use_running_average=not train)
        else:
            _ = self.encoder_bn(x, use_running_average=True)
            x = x / 255.0
        return self.encoder(x, train)

    def __call__(self, x, *, train: bool = False):
        return self.critic(x, train), self.actor(x, train)

@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    val: chex.Array
    log_p: chex.Array

class CriticState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0

class ActorState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0

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

def create_agent(env, network, config, lr, rng):
    init_x = jnp.zeros((1, *env.single_observation_space.shape))
    variables = network.init(rng, init_x, train=False)

    params_critic = flax.core.freeze({
        **variables['params'],
        'actor_head': {}
    })
    params_actor  = flax.core.freeze({
        **variables['params'],
        'critic_head': {}
    })

    tx_critic = optax.chain(
        optax.clip_by_global_norm(config['MAX_GRAD_NORM']),
        optax.radam(lr)
    )
    tx_actor  = optax.chain(
        optax.clip_by_global_norm(config['MAX_GRAD_NORM']),
        optax.radam(lr)
    )

    critic_state = CriticState.create(
        apply_fn  = network.apply,
        params    = params_critic,
        batch_stats = variables['batch_stats'],
        tx        = tx_critic,
    )
    actor_state  = ActorState.create(
        apply_fn  = network.apply,
        params    = params_actor,
        batch_stats = variables['batch_stats'],
        tx        = tx_actor,
    )
    return critic_state, actor_state

def initialize_agents(config, env, rng, network):
    num_agents = config.get("NUM_AGENTS", 1)
    rngs = jax.random.split(rng, num_agents)
    lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
    lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]
    batched_critic_states, batched_actor_states = jax.vmap(lambda r: create_agent(env, network, config, lr, r))(rngs)
    return batched_critic_states, batched_actor_states

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

    # here reset must be out of vmap and jit
    init_obs, env_state = env.reset()

    def train(rng):
        original_seed = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        # INIT NETWORK AND OPTIMIZER
        network = ActorCritic(
            action_dim = env.single_action_space.n,
            norm_type  = config["NORM_TYPE"],
            norm_input = config.get("NORM_INPUT", False),
        )

        rng, _rng = jax.random.split(rng)
        critic_train_states, actor_train_states = initialize_agents(config, env, rng, network)

        # TRAINING LOOP
        def _update_step(runner_state, unused):
            num_envs = config["NUM_ENVS"]
            envs_per_agent = num_envs + config["TEST_ENVS"] if config.get("TEST_DURING_TRAINING", False) else num_envs
            critic_train_states, actor_train_states, expl_state, test_metrics, rng = runner_state
            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                logits = jax.vmap(
                    lambda ts, obs: network.apply(
                        {"params": ts.params, 'batch_stats': ts.batch_stats},
                        obs,
                        train=False,
                        method=ActorCritic.actor)
                )(
                    actor_train_states,
                    last_obs.reshape((config["NUM_AGENTS"], envs_per_agent, *last_obs.shape[1:]))
                )
                vals = jax.vmap(
                    lambda ts, obs: network.apply(
                        {"params": ts.params, "batch_stats": ts.batch_stats},
                        obs,
                        train=False,
                        method=ActorCritic.critic)
                )(
                    critic_train_states,
                    last_obs.reshape((config["NUM_AGENTS"], envs_per_agent, *last_obs.shape[1:]))
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
                vals = vals.reshape((config["NUM_AGENTS"]*envs_per_agent, *vals.shape[2:]))

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
            
            next_obs = transitions.next_obs.reshape(config["NUM_STEPS"], config["NUM_AGENTS"], config["NUM_ENVS"], *transitions.next_obs.shape[2:])
            next_obs_last = next_obs[-1]
            last_next_value = jax.vmap(                               # over agents
                lambda cts, obs: network.apply(
                    {"params": cts.params,
                    "batch_stats": cts.batch_stats},
                    obs,
                    train=False,
                    method=ActorCritic.critic
                ).squeeze(-1)
            )(critic_train_states, next_obs_last) 

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
                critic_train_states, actor_train_states, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(lambda x: preprocess_transitions_per_agent(x, _rng, config), transitions)
                minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), minibatches)
                targets = jax.tree_map(lambda x: preprocess_transitions_per_agent(x, _rng, config), lambda_targets)
                targets = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), targets)
                advantages = jax.tree_map(lambda x: preprocess_transitions_per_agent(x, _rng, config), gae_advantages)
                advantages = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), advantages)

                def _learn_phase(carry, minibatch_and_target):
                    critic_train_states, actor_train_states, rng = carry
                    minibatch, target, advantage = minibatch_and_target
                    def agent_loss_and_update(cts, ats, minibatch, target):
                        # Critic loss and update
                        def _critic_loss_fn(params):
                            pred_val, updates = network.apply(
                                {"params": params, "batch_stats": cts.batch_stats},
                                minibatch.obs, train=True, mutable=["batch_stats"],
                                method=ActorCritic.critic
                            )
                            pred_val = pred_val.squeeze(-1)
                            loss = 0.5 * jnp.mean((pred_val - target) ** 2)
                            return loss, updates["batch_stats"]
                        (critic_loss, new_critic_bs), grads = jax.value_and_grad(_critic_loss_fn, has_aux=True)(cts.params)
                        updates, new_opt_state = cts.tx.update(grads, cts.opt_state, cts.params)
                        new_critic_params = optax.apply_updates(cts.params, updates)
                        cts = cts.replace(params=new_critic_params, batch_stats=new_critic_bs, opt_state=new_opt_state, grad_steps=cts.grad_steps+1)

                        # Actor loss and update
                        def _actor_loss_fn(actor_params):
                            logits, updates = network.apply(
                                {"params": actor_params, "batch_stats": ats.batch_stats},
                                minibatch.obs, train=True, mutable=["batch_stats"],
                                method=ActorCritic.actor
                            )
                            probs, logp, entropy = _policy_from_logits(logits)
                            
                            logp_a = jnp.take_along_axis(logp, minibatch.action[..., None], -1).squeeze(-1)
                            ratio  = jnp.exp(logp_a - minibatch.log_p.squeeze(-1))

                            clip_eps = config["CLIP_EPS"]
                            unclipped = ratio * advantage
                            clipped   = jnp.clip(ratio, 1-clip_eps, 1+clip_eps) * advantage
                            pg_loss   = -jnp.mean(jnp.minimum(unclipped, clipped))

                            ent_loss = -jnp.mean(entropy)
                            ent_coeff = config["ENT_COEFF"]

                            return pg_loss + ent_coeff * ent_loss, (updates["batch_stats"], entropy.mean())

                        (actor_loss, (new_actor_bs, entropies)), grads = jax.value_and_grad(_actor_loss_fn, has_aux=True)(ats.params)
                        updates, new_opt_state = ats.tx.update(grads, ats.opt_state, ats.params)
                        new_actor_params = optax.apply_updates(ats.params, updates)
                        ats = ats.replace(params=new_actor_params, batch_stats=new_actor_bs, opt_state=new_opt_state, grad_steps=ats.grad_steps+1)
                        
                        return critic_loss, actor_loss, entropies, cts, ats
                    
                    critic_loss, actor_loss, entropies, new_critic_train_states, new_actor_train_states = jax.vmap(agent_loss_and_update, in_axes=(0,0,0,0))(
                        critic_train_states, actor_train_states, minibatch, target
                    )
                    return (new_critic_train_states, new_actor_train_states, rng), (critic_loss, actor_loss, entropies)
        
                rng, _rng = jax.random.split(rng)
                (critic_train_states, actor_train_states, rng), (critic_losses, actor_losses, entropies) = jax.lax.scan(_learn_phase, (critic_train_states, actor_train_states, rng), (minibatches, targets, advantages))
                
                mean_entropies = jnp.mean(entropies, axis=0)
                mean_critic_losses = jnp.mean(critic_losses, axis=0)
                mean_actor_losses = jnp.mean(actor_losses, axis=0)
                return (critic_train_states, actor_train_states, rng), (mean_critic_losses, mean_actor_losses, mean_entropies)

            rng, _rng = jax.random.split(rng)
            (critic_train_states, actor_train_states, rng), (critic_losses, actor_losses, entropies) = jax.lax.scan(
                _learn_epoch, (critic_train_states, actor_train_states, rng), None, config["NUM_EPOCHS"]
            )
            critic_train_states = critic_train_states.replace(n_updates=critic_train_states.n_updates + 1)
            actor_train_states = actor_train_states.replace(n_updates=actor_train_states.n_updates + 1)
            mean_critic_losses = jnp.mean(critic_losses, axis=0)
            mean_actor_losses = jnp.mean(actor_losses, axis=0)
            mean_entropies = jnp.mean(entropies, axis=0)
            env_metrics = compute_agent_metrics(infos, config)
            metrics = {}
            for i in range(config["NUM_AGENTS"]):
                eps_val = eps_scheduler(critic_train_states.n_updates[i])
                metrics[f"agent_{i}/env_step"] = critic_train_states.timesteps[i]
                metrics[f"agent_{i}/update_steps"] = critic_train_states.n_updates[i]
                metrics[f"agent_{i}/env_frame"] = critic_train_states.timesteps[i] * env.single_observation_space.shape[0]
                metrics[f"agent_{i}/grad_steps"] = critic_train_states.grad_steps[i]
                metrics[f"agent_{i}/td_loss"] = mean_critic_losses[i]
                metrics[f"agent_{i}/policy_loss"] = mean_actor_losses[i]
                metrics[f"agent_{i}/entropy"] = mean_entropies[i]
                metrics[f"agent_{i}/epsilon"] = eps_val
                
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

            runner_state = (critic_train_states, actor_train_states, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        # test metrics not supported yet
        test_metrics = None

        # train
        rng, _rng = jax.random.split(rng)
        expl_state = (init_obs, env_state)
        runner_state = (critic_train_states, actor_train_states, expl_state, test_metrics, _rng)

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
    print(config)
    if config["DEBUG"]:
        jax.config.update("jax_disable_jit", True)
        # import debugpy
        # debugpy.listen(5678)
        # print("Waiting for client to attach...")
        # debugpy.wait_for_client()
        # print("Client attached")
    print("Config:\n", OmegaConf.to_yaml(config))
    
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
