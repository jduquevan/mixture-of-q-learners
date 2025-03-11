import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

from flax import struct
from flax.training.train_state import TrainState
import optax

import flax.linen as nn
import hydra
from omegaconf import OmegaConf
import wandb

import envpool
import gym
from packaging import version
from functools import partial

is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")
assert is_legacy_gym, "Current version supports only gym<=0.23.1"

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

@struct.dataclass
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    next_obs: jnp.ndarray
    q_val: jnp.ndarray

@struct.dataclass
class ReplayBufferState:
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    next_obs: jnp.ndarray
    idx: jnp.int32
    size: jnp.int32
    capacity: jnp.int32

class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0

class JaxLogEnvPoolWrapper(gym.Wrapper):
    def __init__(self, env, reset_info=True, async_mode=True):
        super(JaxLogEnvPoolWrapper, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.env_name = env.name
        self.env_random_score, self.env_human_score = ATARI_SCORES[self.env_name]
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
        # Update num_envs based on the returned observations.
        self.num_envs = observations.shape[0]
        init_handle_arr = jnp.array(self.init_handle)
        if init_handle_arr.shape[0] != self.num_envs:
            factor = self.num_envs // init_handle_arr.shape[0]
            tiled_handle = jnp.tile(init_handle_arr, (factor,))
        else:
            tiled_handle = init_handle_arr
        env_state = LogEnvState(
            handle=tiled_handle,
            lives=jnp.zeros(self.num_envs, dtype=jnp.float32),
            episode_returns=jnp.zeros(self.num_envs, dtype=jnp.float32),
            episode_lengths=jnp.zeros(self.num_envs, dtype=jnp.float32),
            returned_episode_returns=jnp.zeros(self.num_envs, dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros(self.num_envs, dtype=jnp.float32),
        )
        return observations, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        new_handle, (observations, rewards, dones, infos) = self.step_f(state.handle, action)
        # Ensure new_handle has one entry per environment.
        if new_handle.shape[0] != self.num_envs:
            factor = self.num_envs // new_handle.shape[0]
            new_handle = jnp.tile(new_handle, (factor,))
        new_episode_return = state.episode_returns + infos["reward"]
        new_episode_length = state.episode_lengths + 1
        state = state.replace(
            handle=new_handle,
            episode_returns=(new_episode_return) * (1 - infos["terminated"]) * (1 - infos["TimeLimit.truncated"]),
            episode_lengths=(new_episode_length) * (1 - infos["terminated"]) * (1 - infos["TimeLimit.truncated"]),
            returned_episode_returns=jnp.where(infos["terminated"] + infos["TimeLimit.truncated"],
                                            new_episode_return, state.returned_episode_returns),
            returned_episode_lengths=jnp.where(infos["terminated"] + infos["TimeLimit.truncated"],
                                            new_episode_length, state.returned_episode_lengths),
        )
        if self.reset_info:
            elapsed_steps = infos["elapsed_step"]
            terminated = infos["terminated"] + infos["TimeLimit.truncated"]
            infos = {}
        else:
            elapsed_steps = infos.get("elapsed_step", 0)
            terminated = infos.get("terminated", 0) + infos.get("TimeLimit.truncated", 0)
        normalize_score = lambda x: (x - self.env_random_score) / (self.env_human_score - self.env_random_score)
        infos["returned_episode_returns"] = state.returned_episode_returns
        infos["normalized_returned_episode_returns"] = normalize_score(state.returned_episode_returns)
        infos["returned_episode_lengths"] = state.returned_episode_lengths
        infos["elapsed_step"] = elapsed_steps
        infos["returned_episode"] = terminated
        return observations, state, rewards, dones, infos

class CNN(nn.Module):
    norm_type: str

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                    padding="VALID", kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                    padding="VALID", kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                    padding="VALID", kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=512, kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.relu(x)
        return x

class QNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = x / 255.0
        x = CNN(norm_type=self.norm_type)(x, train=train)
        q_values = nn.Dense(features=self.action_dim, kernel_init=nn.initializers.kaiming_normal())(x)
        return q_values

def replay_buffer_push(rb: ReplayBufferState,
                       obs: jnp.ndarray,
                       actions: jnp.ndarray,
                       rewards: jnp.ndarray,
                       dones: jnp.ndarray,
                       next_obs: jnp.ndarray):
    obs = jnp.transpose(obs, (1, 0) + tuple(range(2, obs.ndim)))
    actions = jnp.transpose(actions, (1, 0))
    rewards = jnp.transpose(rewards, (1, 0))
    dones = jnp.transpose(dones, (1, 0))
    next_obs = jnp.transpose(next_obs, (1, 0) + tuple(range(2, next_obs.ndim)))
    batch_size = obs.shape[0]
    indices = (rb.idx + jnp.arange(batch_size)) % rb.capacity
    def scatter(arr, update, idxs):
         return arr.at[idxs].set(update)
    new_obs      = scatter(rb.obs, obs, indices)
    new_actions  = scatter(rb.actions, actions, indices)
    new_rewards  = scatter(rb.rewards, rewards, indices)
    new_dones    = scatter(rb.dones, dones, indices)
    new_next_obs = scatter(rb.next_obs, next_obs, indices)
    new_idx = (rb.idx + batch_size) % rb.capacity
    new_size = jnp.minimum(rb.size + batch_size, rb.capacity)
    return rb.replace(
         obs=new_obs,
         actions=new_actions,
         rewards=new_rewards,
         dones=new_dones,
         next_obs=new_next_obs,
         idx=new_idx,
         size=new_size,
    )

def replay_buffer_sample(rb: ReplayBufferState, rng: jax.random.PRNGKey, batch_size: int):
    max_idx = rb.size
    idxs = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=max_idx)
    batch_obs      = rb.obs[idxs]
    batch_actions  = rb.actions[idxs]
    batch_rewards  = rb.rewards[idxs]
    batch_dones    = rb.dones[idxs]
    batch_next_obs = rb.next_obs[idxs]
    batch_obs = jnp.transpose(batch_obs, (1, 0) + tuple(range(2, batch_obs.ndim)))
    batch_actions = jnp.transpose(batch_actions, (1, 0))
    batch_rewards = jnp.transpose(batch_rewards, (1, 0))
    batch_dones = jnp.transpose(batch_dones, (1, 0))
    batch_next_obs = jnp.transpose(batch_next_obs, (1, 0) + tuple(range(2, batch_next_obs.ndim)))
    return batch_obs, batch_actions, batch_rewards, batch_dones, batch_next_obs

def eps_greedy_exploration(rng, q_vals, eps):
    a_greedy = jnp.argmax(q_vals, -1)
    explore = jax.random.uniform(rng) < eps
    a_random = jax.random.randint(rng, (), 0, q_vals.shape[-1])
    return jnp.where(explore, a_random, a_greedy)

def step_env(carry, _, train_states, network, env, config, eps_scheduler):
    # carry: (last_obs, env_state, rng)
    # last_obs: (N, E, obs_shape)
    # env_state: each field (N, E, ...)
    # rng: (N, key_size)
    last_obs, env_state, rng = carry
    N, E = last_obs.shape[:2]
    rng_split = jax.vmap(lambda key: jax.random.split(key, 3))(rng)  # (N, 3, key_size)
    new_rng = rng_split[:, 0, :]
    rng_a   = rng_split[:, 1, :]
    _rng_extra = rng_split[:, 2, :]
    def agent_apply(ts, obs):
        return network.apply({"params": ts.params, "batch_stats": ts.batch_stats}, obs, train=False)
    q_vals = jax.vmap(agent_apply)(train_states, last_obs)  # (N, E, action_dim)
    eps_value = eps_scheduler(train_states.n_updates[0])
    eps = jnp.full((N, E), eps_value)
    rngs = jax.vmap(lambda key: jax.random.split(key, E))(rng_a)  # (N, E, key_size)
    def compute_actions(q, keys, eps_row):
        return jax.vmap(eps_greedy_exploration)(keys, q, eps_row)
    new_action = jax.vmap(compute_actions)(q_vals, rngs, eps)  # (N, E)
    def merge_axes(x):
        return x.reshape(N * E, *x.shape[2:])
    merged_env_state = jax.tree_map(merge_axes, env_state)
    merged_action = new_action.reshape(-1)
    merged_obs, merged_env_state_out, merged_reward, merged_done, merged_info = env.step(merged_env_state, merged_action)
    def split_axes(x):
        return x.reshape(N, E, *x.shape[1:])
    new_obs = jax.tree_map(split_axes, merged_obs)
    new_env_state = jax.tree_map(split_axes, merged_env_state_out)
    reward = split_axes(merged_reward)
    done = split_axes(merged_done)
    info = jax.tree_map(split_axes, merged_info)
    transition = Transition(
        obs=last_obs,
        action=new_action,
        reward=config.get("REW_SCALE", 1) * reward,
        done=done,
        next_obs=new_obs,
        q_val=q_vals,
    )
    return (new_obs, new_env_state, new_rng), (transition, info)

@partial(jax.jit, static_argnums=(3,))
def generate_trajectory(network, train_states, config, env, init_obs, env_state, rng, eps_scheduler):
    # init_obs: (N, E, obs_shape), env_state: (N, E, ...), rng: (N, key_size)
    init_carry = (init_obs, env_state, rng)
    final_carry, (transitions, infos) = jax.lax.scan(
        lambda carry, _: step_env(carry, None, train_states, network, env, config, eps_scheduler),
        init_carry,
        None,
        length=config['alg']["NUM_STEPS"],
    )
    return final_carry, (transitions, infos)

def make_env(num_envs, config):
    env = envpool.make(
        config['alg']["ENV_NAME"],
        env_type="gym",
        num_envs=num_envs,
        seed=config["SEED"],
        **config.get("ENV_KWARGS", {})
    )
    env.num_envs = num_envs
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    env.name = config['alg']["ENV_NAME"]
    env = JaxLogEnvPoolWrapper(env)
    return env

def create_agent(rng, config, env):
    dummy_input = jnp.ones((1, *env.single_observation_space.shape), dtype=jnp.float32)
    model = QNetwork(
        action_dim=env.single_action_space.n,
        norm_type=config['alg']["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
    )
    variables = model.init(rng, dummy_input, train=True)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    lr = config['alg']["LR"]
    if config.get("LR_LINEAR_DECAY", False):
        lr = optax.linear_schedule(
            init_value=config['alg']["LR"],
            end_value=1e-20,
            transition_steps=(
                config['alg']["NUM_UPDATES_DECAY"] *
                config['alg']["NUM_MINIBATCHES"] *
                config['alg']["NUM_EPOCHS"]
            ),
        )
    tx = optax.chain(
        optax.clip_by_global_norm(config['alg']["MAX_GRAD_NORM"]),
        optax.radam(learning_rate=lr),
    )
    train_state = CustomTrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=tx,
    )
    train_state = train_state.replace(
        timesteps=jnp.array(train_state.timesteps),
        n_updates=jnp.array(train_state.n_updates),
        grad_steps=jnp.array(train_state.grad_steps)
    )
    return train_state

def initialize_agents(config, env, rng):
    num_agents = config['alg']["NUM_AGENTS"]
    rngs = jax.random.split(rng, num_agents)
    vectorized_create_agent = jax.vmap(lambda r: create_agent(r, config, env))
    batched_train_states = vectorized_create_agent(rngs)
    return batched_train_states

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["DEBUG"]:
        jax.config.update("jax_disable_jit", True)
    rng = jax.random.PRNGKey(config["SEED"])
    num_agents = config['alg']["NUM_AGENTS"]
    # E: number of environments per agent.
    E = config['alg']["NUM_ENVS"]
    total_envs = num_agents * E
    env = make_env(total_envs, config)
    init_obs, env_state = env.reset()  # init_obs: (total_envs, 4, 84, 84)
    batched_init_obs = init_obs.reshape(num_agents, E, *init_obs.shape[1:])
    batched_env_state = jax.tree_map(lambda x: x.reshape(num_agents, E, *x.shape[1:]), env_state)
    eps_scheduler = optax.linear_schedule(
        init_value=config['alg']["EPS_START"],
        end_value=config['alg']["EPS_FINISH"],
        transition_steps=int(config['alg']["TOTAL_TIMESTEPS_DECAY"] * config['alg']["EPS_DECAY"]),
    )
    network = QNetwork(
        action_dim=env.single_action_space.n,
        norm_type=config['alg']["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
    )
    agent_train_states = initialize_agents(config, env, rng)
    agent_rngs = jax.random.split(rng, num_agents)
    final_carry, (transitions, infos) = generate_trajectory(
        network, agent_train_states, config, env,
        batched_init_obs, batched_env_state, agent_rngs, eps_scheduler
    )
    print("Generated trajectory shapes:")
    print(jax.tree_map(lambda x: x.shape, transitions))
    print("Train method completed.")

if __name__ == "__main__":
    main()
