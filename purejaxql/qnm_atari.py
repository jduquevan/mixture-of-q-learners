import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Tuple

from flax import struct
from flax.training.train_state import TrainState
import optax
import os
import uuid

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
        infos["cumulative_episode_returns"] = state.episode_returns
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
    def __call__(self, x, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else: 
            normalize = lambda x: x

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                    padding="VALID", kernel_init=nn.initializers.kaiming_normal())(x)
        x = normalize(x) 
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                    padding="VALID", kernel_init=nn.initializers.kaiming_normal())(x)
        x = normalize(x) 
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                    padding="VALID", kernel_init=nn.initializers.kaiming_normal())(x)
        x = normalize(x) 
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=512, kernel_init=nn.initializers.kaiming_normal())(x)
        x = normalize(x) 
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

@struct.dataclass
class DiskTransition:
    """
    Separate dataclass to store transitions for the disk + GPU buffer.
    (Doesn't store q_val by default; adapt as needed.)
    """
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    next_obs: jnp.ndarray

def get_memmap_dir(config):
    """
    Returns a unique directory for memmap replay data.
    If WANDB is enabled and we have a wandb.run, we use wandb.run.id.
    Otherwise, we generate a random UUID.
    """
    base = config['alg']['MEMMAP_DIR'] # The base path from config, default "."

    if config["WANDB_MODE"] != "disabled" and wandb.run is not None:
        # If W&B is actually running, wandb.run.id is something like '1abc234'
        run_id = wandb.run.id
        return os.path.join(base, run_id)
    else:
        # If W&B is disabled or wandb.run not present, use a random subdirectory
        random_id = str(uuid.uuid4())
        return os.path.join(base, random_id)

class DiskGPUMixedReplayBuffer:
    """
    A replay buffer that:
      - Stores the entire buffer on disk (via np.memmap).
      - Uses a small 'mini-buffer' on GPU with shape (mini_buffer_size, N, *obs_shape).
      - Inserts "blocks" of shape (B, N, *obs_shape) at once.
      - Reads the same shape (B, N, *obs_shape) at once when sampling.
    """

    def __init__(
        self,
        capacity: int,
        mini_buffer_size: int,
        n_agents: int,            # <--- new argument for N
        traj_len: int,
        obs_shape: tuple,
        memmap_dir: str,
        obs_dtype=np.uint8,
        reward_dtype=np.float32,
        action_dtype=np.int32,
    ):
        os.makedirs(memmap_dir, exist_ok=True)
        mode = "w+"

        self.capacity = capacity   # how many "slots" in the ring
        self.mini_buffer_size = mini_buffer_size
        self.n_agents = n_agents
        self.obs_shape = obs_shape

        # On-disk arrays. We'll store shape: (capacity, N, *obs_shape)
        self.obs_memmap_path      = os.path.join(memmap_dir, "obs.dat")
        self.next_obs_memmap_path = os.path.join(memmap_dir, "next_obs.dat")
        self.actions_memmap_path  = os.path.join(memmap_dir, "actions.dat")
        self.rewards_memmap_path  = os.path.join(memmap_dir, "rewards.dat")
        self.dones_memmap_path    = os.path.join(memmap_dir, "dones.dat")

        self.obs_memmap = np.memmap(
            self.obs_memmap_path,
            mode=mode, dtype=obs_dtype,
            shape=(capacity, n_agents, traj_len) + obs_shape
        )
        self.next_obs_memmap = np.memmap(
            self.next_obs_memmap_path,
            mode=mode, dtype=obs_dtype,
            shape=(capacity, n_agents, traj_len) + obs_shape
        )
        self.actions_memmap = np.memmap(
            self.actions_memmap_path,
            mode=mode, dtype=action_dtype,
            shape=(capacity, n_agents, traj_len)
        )
        self.rewards_memmap = np.memmap(
            self.rewards_memmap_path,
            mode=mode, dtype=reward_dtype,
            shape=(capacity, n_agents, traj_len)
        )
        self.dones_memmap = np.memmap(
            self.dones_memmap_path,
            mode=mode, dtype=np.bool_,
            shape=(capacity, n_agents, traj_len)
        )

        self.disk_size = 0
        self.disk_idx  = 0

        # -- GPU mini-buffer arrays: shape (mini_buffer_size, N, *obs_shape)
        self.obs_gpu      = jnp.zeros((mini_buffer_size, n_agents, traj_len) + obs_shape, dtype=obs_dtype)
        self.next_obs_gpu = jnp.zeros((mini_buffer_size, n_agents, traj_len) + obs_shape, dtype=obs_dtype)
        self.actions_gpu  = jnp.zeros((mini_buffer_size, n_agents, traj_len), dtype=action_dtype)
        self.rewards_gpu  = jnp.zeros((mini_buffer_size, n_agents, traj_len), dtype=reward_dtype)
        self.dones_gpu    = jnp.zeros((mini_buffer_size, n_agents, traj_len), dtype=bool)

        self.mb_read_idx = 0  # next free slot for writing (or next read offset)

    def _host_to_device(self, obs_np, next_obs_np, act_np, rew_np, done_np):
        return (
            jnp.array(obs_np),
            jnp.array(next_obs_np),
            jnp.array(act_np),
            jnp.array(rew_np),
            jnp.array(done_np)
        )

    def _device_to_host(self, obs_j, next_obs_j, act_j, rew_j, done_j):
        return (
            np.array(obs_j, copy=False),
            np.array(next_obs_j, copy=False),
            np.array(act_j, copy=False),
            np.array(rew_j, copy=False),
            np.array(done_j, copy=False),
        )

    def push_block(self, obs_block, next_obs_block, act_block, rew_block, done_block):
        """
        Insert a block of shape (B, N, S, *obs_shape) into the GPU mini-buffer
        at [mb_read_idx : mb_read_idx + B].
        If it overflows, flush to disk, reset to 0, then insert.
        """
        B = obs_block.shape[0]
        start_idx = self.mb_read_idx
        end_idx = start_idx + B

        if end_idx >= self.mini_buffer_size:
            self.flush_gpu_to_disk()
            self.mb_read_idx = 0
            start_idx = 0
            end_idx = B

        self.obs_gpu = self.obs_gpu.at[start_idx:end_idx].set(obs_block)
        self.next_obs_gpu = self.next_obs_gpu.at[start_idx:end_idx].set(next_obs_block)
        self.actions_gpu = self.actions_gpu.at[start_idx:end_idx].set(act_block)
        self.rewards_gpu = self.rewards_gpu.at[start_idx:end_idx].set(rew_block)
        self.dones_gpu = self.dones_gpu.at[start_idx:end_idx].set(done_block)

        self.mb_read_idx = end_idx

    def sample_block(self, B: int):
        """
        Return a block of shape (B, N, S, ...) from the GPU mini-buffer,
        starting at mb_read_idx. If we don't have B items left,
        we flush & refill from disk, then read from 0.
        """
        start_idx = self.mb_read_idx
        end_idx = start_idx + B

        if end_idx > self.mini_buffer_size:
            # flush and refill logic
            self.flush_gpu_to_disk()
            self.refill_from_disk(start_idx=0, count=None)
            self.mb_read_idx = 0
            start_idx = 0
            end_idx = B

        obs_block      = self.obs_gpu[start_idx:end_idx]
        next_obs_block = self.next_obs_gpu[start_idx:end_idx]
        act_block      = self.actions_gpu[start_idx:end_idx]
        rew_block      = self.rewards_gpu[start_idx:end_idx]
        done_block     = self.dones_gpu[start_idx:end_idx]

        self.mb_read_idx = end_idx
        return obs_block, next_obs_block, act_block, rew_block, done_block

    def flush_gpu_to_disk(self, start_disk_idx=None, count=None):
        """
        Bulk-write [0 : self.mb_read_idx] from GPU arrays => disk arrays.
        Usually called if the ring is full or if we want to persist data.
        """
        if start_disk_idx is None:
            start_disk_idx = self.disk_idx
        if count is None:
            count = self.mb_read_idx

        end_disk_idx = start_disk_idx + count
        if end_disk_idx > self.capacity:
            raise ValueError("Not enough capacity on disk to flush block. Handle ring or bigger capacity.")

        # device -> host
        obs_np, next_obs_np, act_np, rew_np, done_np = self._device_to_host(
            self.obs_gpu[:count],
            self.next_obs_gpu[:count],
            self.actions_gpu[:count],
            self.rewards_gpu[:count],
            self.dones_gpu[:count],
        )

        # write to memmap
        self.obs_memmap[start_disk_idx : end_disk_idx]      = obs_np
        self.next_obs_memmap[start_disk_idx : end_disk_idx] = next_obs_np
        self.actions_memmap[start_disk_idx : end_disk_idx]  = act_np
        self.rewards_memmap[start_disk_idx : end_disk_idx]  = rew_np
        self.dones_memmap[start_disk_idx : end_disk_idx]    = done_np

        self.disk_size = max(self.disk_size, end_disk_idx)
        self.disk_idx  = end_disk_idx
        self.mb_read_idx = 0  # we've "consumed" them from the mini-buffer

    def refill_from_disk(self, start_idx=0, count=None):
        """
        Bulk-read from disk => GPU mini-buffer [0..count], overwriting what's there.
        Typically for sampling older transitions, if you want that flow.
        """
        if count is None:
            count = self.mini_buffer_size
        available = self.disk_size - start_idx
        if available <= 0:
            return  # nothing

        count = min(count, available)
        end_idx = start_idx + count

        # read from memmap
        obs_np      = self.obs_memmap     [start_idx : end_idx]
        next_obs_np = self.next_obs_memmap[start_idx : end_idx]
        act_np      = self.actions_memmap [start_idx : end_idx]
        rew_np      = self.rewards_memmap [start_idx : end_idx]
        done_np     = self.dones_memmap   [start_idx : end_idx]

        # host -> device
        obs_j, next_obs_j, act_j, rew_j, done_j = self._host_to_device(
            obs_np, next_obs_np, act_np, rew_np, done_np
        )

        # store to GPU
        self.obs_gpu      = self.obs_gpu.at[:count].set(obs_j)
        self.next_obs_gpu = self.next_obs_gpu.at[:count].set(next_obs_j)
        self.actions_gpu  = self.actions_gpu.at[:count].set(act_j)
        self.rewards_gpu  = self.rewards_gpu.at[:count].set(rew_j)
        self.dones_gpu    = self.dones_gpu.at[:count].set(done_j)

        self.mb_read_idx = 0  # we can now read from slot 0

    def __len__(self):
        return self.disk_size

def constant_zero_eps_scheduler(_):
    return 0.0

@partial(jax.jit, static_argnums=(0, 2, 3, 7, 8)) 
def generate_trajectory(network, train_states, num_steps, env, init_obs, env_state, rng, eps_scheduler, num_eval_envs=0):  
    def step_env(carry, _, agent_train_states, network, env, eps_scheduler, num_eval_envs):  #
        # carry: (last_obs, env_state, rng)
        last_obs, env_state, rng = carry
        N, E_total = last_obs.shape[:2]  
        rng_split = jax.vmap(lambda key: jax.random.split(key, 3))(rng)
        new_rng = rng_split[:, 0, :]
        rng_a   = rng_split[:, 1, :]
        _rng_extra = rng_split[:, 2, :]

        def agent_apply(ts, obs):
            return network.apply({"params": ts.params, "batch_stats": ts.batch_stats}, obs, train=False)
        q_vals = jax.vmap(agent_apply)(agent_train_states, last_obs)  # (N, E_total, action_dim)

        # Use the eps_scheduler value for training environments and force 0 for evaluation envs:
        eps_value = eps_scheduler(agent_train_states.n_updates[0])
        train_count = E_total - num_eval_envs  
        eps_train = jnp.full((N, train_count), eps_value) 
        eps_test = jnp.zeros((N, num_eval_envs))          
        eps = jnp.concatenate([eps_train, eps_test], axis=1)  

        rngs = jax.vmap(lambda key: jax.random.split(key, E_total))(rng_a)  
        def eps_greedy_exploration(rn, q_vals_, eps_):
            a_greedy = jnp.argmax(q_vals_, -1)
            explore = jax.random.uniform(rn) < eps_
            a_random = jax.random.randint(rn, (), 0, q_vals_.shape[-1])
            return jnp.where(explore, a_random, a_greedy)

        def compute_actions(q, keys, eps_row):
            return jax.vmap(eps_greedy_exploration)(keys, q, eps_row)
            
        new_action = jax.vmap(compute_actions)(q_vals, rngs, eps) 
        merged_action = new_action.reshape(-1)
        merged_obs, merged_env_state_out, merged_reward, merged_done, merged_info = env.step(env_state, merged_action)

        def split_axes(x):
            return x.reshape(N, E_total, *x.shape[1:])

        new_obs = jax.tree_map(split_axes, merged_obs)
        new_env_state = merged_env_state_out
        reward = split_axes(merged_reward)
        done = split_axes(merged_done)
        info = jax.tree_map(split_axes, merged_info)

        transition = Transition(
            obs=last_obs,
            action=new_action,
            reward=reward,
            done=done,
            next_obs=new_obs,
            q_val=q_vals,
        )
        return (new_obs, new_env_state, new_rng), (transition, info)

    init_carry = (init_obs, env_state, rng)
    final_carry, (transitions, infos) = jax.lax.scan(
        lambda carry, _: step_env(carry, None, train_states, network, env, eps_scheduler, num_eval_envs),  # CHANGED: pass num_eval_envs
        init_carry,
        None,
        length=num_steps,
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

def compute_targets_stable(reward: jnp.ndarray,
                           done: jnp.ndarray,
                           q_next: jnp.ndarray,
                           gamma: float,
                           lam: float) -> jnp.ndarray:
    """
    Compute TD(λ) targets
    Args:
      reward: shape [T, ...] each step's reward
      done:   shape [T, ...] each step's done mask (0 or 1)
      q_next: shape [T, ..., num_actions] each step's Q-values
      gamma:  discount factor
      lam:    λ parameter
    Returns:
      shape [T, ...] of λ-returns
    """
    last_q = jnp.max(q_next[-1], axis=-1)
    lambda_returns = reward[-1] + gamma * (1.0 - done[-1]) * last_q

    def _backward_scan(carry, x):
        """
        carry: (lambda_returns, next_q)
        x: (reward_t, qvals_t, done_t)
        """
        ret, next_q_val = carry
        rew_t, qvals_t, done_t = x
        target_bootstrap = rew_t + gamma * (1.0 - done_t) * next_q_val
        delta = ret - next_q_val
        ret = target_bootstrap + gamma * lam * delta
        ret = (1.0 - done_t) * ret + done_t * rew_t
        next_q_val = jnp.max(qvals_t, axis=-1)

        return (ret, next_q_val), ret

    (final_ret, _), targets = jax.lax.scan(
        _backward_scan,
        (lambda_returns, last_q),
        (reward[:-1], q_next[:-1], done[:-1]),
        reverse=True
    )
    targets = jnp.concatenate([targets, lambda_returns[None]], axis=0)
    return targets

@partial(jax.jit, static_argnums=(2, 3))
def train_step(agent_train_states, batch, gamma, lam):
    """
    Performs one gradient update using TD(λ) returns.
    
    agent_train_states: a batched CustomTrainState (for each agent)
    batch: a tuple (obs_b, nxt_b, act_b, rew_b, don_b) with shapes:
         obs_b:  (B, N, S, *obs_shape)
         nxt_b:  (B, N, S, *obs_shape)
         act_b:  (B, N, S)
         rew_b:  (B, N, S)
         don_b:  (B, N, S)
    gamma: discount factor (float)
    lam: TD(λ) parameter (float, typically between 0 and 1)
    
    Returns: (new_train_states, loss_vals)
    """
    obs_b, nxt_b, act_b, rew_b, don_b = batch

    def per_agent_multistep_loss(params, bstat, obs, nxt, act, rew, done):
        B_, S_ = obs.shape[:2]
        obs_r = obs.reshape(B_, S_, *obs.shape[2:])
        nxt_r = nxt.reshape(B_, S_, *nxt.shape[2:])
        act_r = act.reshape(B_, S_)
        rew_r = rew.reshape(B_, S_)
        done_r = done.reshape(B_, S_)
        qvals, vars_out = agent_train_states.apply_fn(
            {"params": params, "batch_stats": bstat},
            obs_r.reshape(B_ * S_, *obs_r.shape[2:]),
            train=True, mutable=["batch_stats"]
        )
        qvals = qvals.reshape(B_, S_, -1)
        next_qvals = agent_train_states.apply_fn(
            {"params": params, "batch_stats": bstat},
            nxt_r.reshape(B_ * S_, *nxt_r.shape[2:]),
            train=False
        )
        next_qvals = next_qvals.reshape(B_, S_, -1)
        g_all = jax.vmap(lambda r, d, q: compute_targets_stable(r, d, q, gamma, lam))(
            rew_r, done_r, next_qvals
        )
        chosen_q = jnp.take_along_axis(qvals, act_r[..., None], axis=-1).squeeze(-1)
        loss = 0.5 * jnp.mean((chosen_q - g_all) ** 2)
        return loss, vars_out["batch_stats"]

    def agent_loss_and_grad(p, bstat, obs, nxt, act, rew, done):
        (lval, new_bstat), grads = jax.value_and_grad(per_agent_multistep_loss, has_aux=True)(
            p, bstat, obs, nxt, act, rew, done
        )
        return lval, (new_bstat, grads)

    # Rearrange batch dimensions from (B, N, S, ...) to (N, B, S, ...)
    obs_b_t = jnp.transpose(obs_b, (1, 0) + tuple(range(2, obs_b.ndim)))
    nxt_b_t = jnp.transpose(nxt_b, (1, 0) + tuple(range(2, nxt_b.ndim)))
    act_b_t = jnp.transpose(act_b, (1, 0) + tuple(range(2, act_b.ndim)))
    rew_b_t = jnp.transpose(rew_b, (1, 0) + tuple(range(2, rew_b.ndim)))
    don_b_t = jnp.transpose(don_b, (1, 0) + tuple(range(2, don_b.ndim)))

    loss_vals, (new_bstats, grads_all) = jax.vmap(
        agent_loss_and_grad, in_axes=(0, 0, 0, 0, 0, 0, 0)
    )(agent_train_states.params,
      agent_train_states.batch_stats,
      obs_b_t, nxt_b_t, act_b_t, rew_b_t, don_b_t)

    def agent_opt_update(p, g, st):
        updates, new_st = agent_train_states.tx.update(g, st, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_st

    new_params, new_opt_states = jax.vmap(agent_opt_update)(
        agent_train_states.params, grads_all, agent_train_states.opt_state
    )

    new_train_states = agent_train_states.replace(
        params=new_params,
        batch_stats=new_bstats,
        opt_state=new_opt_states,
        n_updates=agent_train_states.n_updates + 1,
        grad_steps=agent_train_states.grad_steps + 1
    )
    return new_train_states, loss_vals

@partial(jax.jit, static_argnums=(2, 3, 4))
def batched_train_step(
    agent_train_states: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    gamma: float,
    lam: float,
    num_minibatches: int,
) -> Tuple[TrainState, jnp.ndarray]:
    """
    Batches the training data into minibatches and performs a gradient update for each minibatch.

    Args:
        agent_train_states: A batched CustomTrainState (for each agent).
        batch: A tuple (obs_b, nxt_b, act_b, rew_b, don_b) with shapes:
             obs_b:  (B, N, S, *obs_shape)
             nxt_b:  (B, N, S, *obs_shape)
             act_b:  (B, N, S)
             rew_b:  (B, N, S)
             don_b:  (B, N, S)
        gamma: Discount factor (float).
        lam: TD(λ) parameter (float, typically between 0 and 1).
        num_minibatches: The number of minibatches to divide the training data into.

    Returns:
        A tuple containing:
            - new_train_states: The updated training states after processing all minibatches.
            - jnp.mean(loss_vals): Mean of loss values across minibatches.
    """
    obs_b, nxt_b, act_b, rew_b, don_b = batch

    batch_size = obs_b.shape[0]
    minibatch_size = int(batch_size // num_minibatches)

    def scan_minibatch(carry, i):
        agent_train_states = carry
        start = i * minibatch_size

        mini_obs = jax.lax.dynamic_slice(obs_b, (start, 0, 0, 0, 0, 0), (minibatch_size,) + obs_b.shape[1:]) 
        mini_nxt = jax.lax.dynamic_slice(nxt_b, (start, 0, 0, 0, 0, 0), (minibatch_size,) + nxt_b.shape[1:])  
        mini_act = jax.lax.dynamic_slice(act_b, (start, 0, 0), (minibatch_size,) + act_b.shape[1:])
        mini_rew = jax.lax.dynamic_slice(rew_b, (start, 0, 0), (minibatch_size,) + rew_b.shape[1:])
        mini_don = jax.lax.dynamic_slice(don_b, (start, 0, 0), (minibatch_size,) + don_b.shape[1:])

        mini_batch = (mini_obs, mini_nxt, mini_act, mini_rew, mini_don)

        new_train_states, loss_vals = train_step(
            agent_train_states, mini_batch, gamma, lam
        )
        return new_train_states, loss_vals

    # Perform scan to train over all minibatches
    init_carry = agent_train_states
    new_train_states, loss_vals = jax.lax.scan(
        scan_minibatch, init_carry, jnp.arange(num_minibatches)
    )

    # The result of the training gets compiled and then gets returned
    return new_train_states, jnp.mean(loss_vals)

def qnm_algorithm(
    config,
    agent_train_states,
    disk_replay_buffer,
    network,
    rng,
    env,
    eps_scheduler,
    batched_init_obs,
    batched_env_state,
    current_iter=0
):
    # Hyperparameters
    N = config['alg']['NUM_AGENTS']
    S = config['alg']['NUM_STEPS']
    gamma = config['alg'].get('GAMMA', 0.99)
    lam = config['alg'].get('LAMBDA', 0.9)
    E_train = config['alg']["NUM_ENVS"]
    E_eval = config['alg'].get("TEST_ENVS", 0) if config['alg'].get("TEST_DURING_TRAINING", False) else 0  # NEW

    # --- Training Rollout with combined envs ---
    (next_obs, next_env_state, new_rng), (transitions, infos) = generate_trajectory(
        network,
        agent_train_states,
        S,
        env,
        batched_init_obs,
        batched_env_state,
        rng,
        eps_scheduler,
        num_eval_envs=E_eval 
    )

    agent_train_states = agent_train_states.replace(
        timesteps=agent_train_states.timesteps + S * (E_train + E_eval)  
    )

    # --- Split transitions and infos into training and evaluation parts ---
    if config['alg'].get('TEST_DURING_TRAINING', False) and E_eval > 0:
        # Assuming the env dimension is the 2nd axis
        slicing_fnct_train = lambda x: x[:, :, :E_train]
        slicing_fnct_eval = lambda x: x[:, :, E_train:]
        transitions_train = jax.tree_map(slicing_fnct_train, transitions)  
        transitions_eval = jax.tree_map(slicing_fnct_eval, transitions)   
        infos_train = jax.tree_map(slicing_fnct_train, infos)
        infos_eval = jax.tree_map(slicing_fnct_eval, infos)
    else:
        transitions_train = transitions
        infos_train = infos
        infos_eval = None

    qvals = jnp.mean(
        jnp.take_along_axis(
            transitions_train.q_val,
            transitions_train.action[..., None],
            axis=-1
        ).squeeze(-1),
        axis=(0, 2)
    )

    # Log training metrics using infos_train
    metrics = {k: jnp.mean(v, axis=(0, 2)) for k, v in infos_train.items()}
    metrics_np = {k: np.array(val) for k, val in metrics.items()}
    for i in range(N):
        agent_metrics = {k: float(metrics_np[k][i]) for k in metrics_np}
        wandb.log(
            {f"agent_{i}/env_metrics": agent_metrics},
            step=int(agent_train_states.n_updates[i])
        )

    # --- Rearranging transitions for training update (only training envs) ---
    obs_block  = jnp.transpose(transitions_train.obs,      (2, 1, 0) + tuple(range(3, transitions_train.obs.ndim)))
    next_block = jnp.transpose(transitions_train.next_obs, (2, 1, 0) + tuple(range(3, transitions_train.next_obs.ndim)))
    act_block  = jnp.transpose(transitions_train.action,   (2, 1, 0))
    rew_block  = jnp.transpose(transitions_train.reward,   (2, 1, 0))
    don_block  = jnp.transpose(transitions_train.done,     (2, 1, 0))

    if config['alg'].get('SKIP_REPLAY_BUFFER', False):
        batch = (obs_block, next_block, act_block, rew_block, don_block)
    else:
        disk_replay_buffer.push_block(obs_block, next_block, act_block, rew_block, don_block)
        if len(disk_replay_buffer) < config["alg"].get("MIN_DISK_SIZE", 1):
            print("Not enough transitions on disk yet. Skipping update.")
            return agent_train_states, new_rng, None, None
        B = config["alg"].get("MINIBATCH_SIZE", 128)
        batch = disk_replay_buffer.sample_block(B)

    # --- Training Step with Epochs remains the same ---
    def train_epoch(carry, _):
        agent_train_states, loss_val = carry
        new_agent_train_states, loss_vals = batched_train_step(
            agent_train_states, batch, gamma, lam, config['alg']['NUM_MINIBATCHES']
        )
        loss_val = jnp.mean(loss_vals)
        return (new_agent_train_states, loss_val), None
        
    init_carry = (agent_train_states, jnp.array(0.0))
    (new_agent_train_states, final_loss_val), _ = jax.lax.scan(
        train_epoch, init_carry, jnp.arange(config['alg']['NUM_EPOCHS'])
    )

    loss_vals = final_loss_val

    for i in range(N):
        eps_val = float(eps_scheduler(new_agent_train_states.n_updates[i]))
        agent_metrics = {
            "env_step": float(new_agent_train_states.timesteps[i]),
            "update_steps": float(new_agent_train_states.n_updates[i]),
            "env_frame": float(new_agent_train_states.timesteps[i] * env.single_observation_space.shape[0]),
            "grad_steps": float(new_agent_train_states.grad_steps[i]),
            "td_loss": float(loss_vals),
            "qvals": float(qvals[i]),
            "epsilon": eps_val,
        }
        wandb.log(
            {f"agent_{i}/metrics": agent_metrics},
            step=int(new_agent_train_states.n_updates[i])
        )

    # --- Optionally log evaluation metrics from the evaluation envs ---
    if infos_eval is not None:
        eval_metrics = {k: jnp.mean(v, axis=(0, 2)) for k, v in infos_eval.items()}
        eval_metrics_np = {k: np.array(val) for k, val in eval_metrics.items()}
        for i in range(N):
            agent_eval_metrics = {k: float(eval_metrics_np[k][i]) for k in eval_metrics_np}
            wandb.log(
                {f"agent_{i}/eval_env_metrics": agent_eval_metrics},
                step=int(new_agent_train_states.n_updates[i])
            )

    # Return the updated train states, RNG, and the new env state and observations (which include both training and eval)
    return new_agent_train_states, new_rng, next_env_state, next_obs

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    config["NUM_UPDATES_DECAY"] = (
        config['alg']["TOTAL_TIMESTEPS_DECAY"] // config['alg']["NUM_STEPS"] // config['alg']["NUM_ENVS"]
    )
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["DEBUG"]:
        jax.config.update("jax_disable_jit", True)

    rng = jax.random.PRNGKey(config["SEED"])
    num_agents = config['alg']["NUM_AGENTS"]
    E_train = config['alg']["NUM_ENVS"]
    E_eval = config['alg'].get("TEST_ENVS", 0) if config['alg'].get("TEST_DURING_TRAINING", False) else 0
    total_envs_per_agent = E_train + E_eval
    S = config['alg']["NUM_STEPS"]
    mini_buffer_size = config["alg"]["MINI_BUFFER_SIZE"]
    total_envs = num_agents * total_envs_per_agent

    # Split the key into one per agent
    agent_rngs = jax.random.split(rng, num_agents)

    # 1) Create training environment
    env = make_env(total_envs, config)
    init_obs, env_state = env.reset()
    batched_init_obs = init_obs.reshape(num_agents, total_envs_per_agent, *init_obs.shape[1:])

    # 2) Create the QNetwork and agent train states
    network = QNetwork(
        action_dim=env.single_action_space.n,
        norm_type=config['alg']["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
    )
    agent_train_states = initialize_agents(config, env, rng)

    # 3) Create the replay buffer
    memmap_dir = get_memmap_dir(config)
    disk_replay_buffer = DiskGPUMixedReplayBuffer(
        capacity=1_000_000,
        mini_buffer_size=mini_buffer_size,
        n_agents=num_agents,
        traj_len=S,
        obs_shape=(4, 84, 84),
        memmap_dir=memmap_dir,
    )

    # 4) Epsilon schedule for training
    eps_scheduler = optax.linear_schedule(
        init_value=config['alg']["EPS_START"],
        end_value=config['alg']["EPS_FINISH"],
        transition_steps=int(config["NUM_UPDATES_DECAY"] * config['alg']["EPS_DECAY"]),
    )

    alg_name = config.get("ALG_NAME", "qnm")
    env_name = config.get("ENV_NAME", "NAN")

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )
    obs = batched_init_obs

    num_iters = int(config['alg']["TRAINING_ITERATIONS"])
    for iteration in range(num_iters):
        agent_train_states, agent_rngs, env_state, obs =  qnm_algorithm(
            config,
            agent_train_states,
            disk_replay_buffer,
            network,
            agent_rngs,  
            env,
            eps_scheduler,
            obs,
            env_state,
            current_iter=iteration
        )
        print(f"Iteration {iteration} complete. Disk buffer size={len(disk_replay_buffer)}.")

    print("Done training.")

if __name__ == "__main__":
    main()