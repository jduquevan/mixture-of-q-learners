# mixture of q-learners for atari
"""
When test_during_training is set to True, an additional number of parallel test environments are used to evaluate the agent during training using greedy actions,
but not for training purposes. Stopping training for evaluation can be very expensive, as an episode in Atari can last for hundreds of thousands of steps.
"""

import copy
import time
import os
import jax
import jax.numpy as jnp
import jax.random as rax
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


class QNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        x = CNN(norm_type=self.norm_type)(x, train)
        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def preprocess_agent_transition(x, rng, NUM_MINIBATCHES):
    # x: (num_steps, num_envs, ...)
    flattened = x.reshape(-1, *x.shape[2:])
    shuffled = jax.random.permutation(rng, flattened)
    return shuffled.reshape(NUM_MINIBATCHES, -1, *x.shape[2:])

def preprocess_transitions_per_agent(x, rng, NUM_AGENTS, NUM_MINIBATCHES):
    # x: (num_steps, total_envs, ...), with total_envs = NUM_AGENTS * NUM_ENVS.
    num_steps = x.shape[0]
    total_envs = x.shape[1]
    num_envs = x.shape[1] // NUM_AGENTS
    # First, transpose to (total_envs, num_steps, ...)
    x = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
    # Then, reshape total_envs into (num_agents, num_envs)
    x = x.reshape((NUM_AGENTS, num_envs, num_steps) + x.shape[2:])
    # Finally, transpose to (num_agents, num_steps, num_envs, ...)
    x = jnp.transpose(x, (0, 2, 1) + tuple(range(3, x.ndim)))
    # Split rng for each agent
    rngs = jax.random.split(rng, NUM_AGENTS)
    return jax.vmap(lambda x_agent, r: preprocess_agent_transition(x_agent, r, NUM_MINIBATCHES), in_axes=(0, 0))(x, rngs)

def update_buffer(buf, x, index, NUM_AGENTS):
    num_steps = x.shape[0]
    num_envs = x.shape[1] // NUM_AGENTS
    # First, transpose to (total_envs, num_steps, ...)
    x = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
    # Then, reshape total_envs into (num_agents, num_envs, num_steps, ...)
    x = x.reshape((NUM_AGENTS, num_envs, num_steps) + x.shape[2:])
    # update the buffer
    offsets = (0, index, 0) + (0,) * (buf.ndim - 3)
    # write the whole block for all agents in one shot
    buf = jax.lax.dynamic_update_slice(buf, x, offsets)
    return buf
    
    
def compute_agent_metrics(metrics, NUM_AGENTS):
    def compute_agent_metric(metric):
        num_steps = metric.shape[0]
        envs_per_agent = metric.shape[1] // NUM_AGENTS
        
        metric = metric.reshape((num_steps, NUM_AGENTS, envs_per_agent) + metric.shape[2:])
        return jnp.mean(metric, axis=(0, 2))
    
    return jax.tree_util.tree_map(
        lambda m: compute_agent_metric(m),
        metrics,
    )

def eps_greedy_exploration(rng, q_vals, eps):
    rng_a, rng_e = jax.random.split(
        rng
    )  # a key for sampling random actions and one for picking
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    chosed_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape)
        < eps,  
        jax.random.randint(
            rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
        ), 
        greedy_actions,
    )
    return chosed_actions

def create_agent(env, network, lr, rng, MAX_GRAD_NORM):
    init_x = jnp.zeros((1, *env.single_observation_space.shape))
    network_variables = network.init(rng, init_x, train=False)

    tx = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.radam(learning_rate=lr),
    )

    train_state = CustomTrainState.create(
        apply_fn=network.apply,
        params=network_variables["params"],
        batch_stats=network_variables["batch_stats"],
        tx=tx,
    )
    train_state = train_state.replace(
        timesteps=jnp.array(train_state.timesteps),
        n_updates=jnp.array(train_state.n_updates),
        grad_steps=jnp.array(train_state.grad_steps)
    )
    return train_state

def initialize_agents(env, 
                      rng, 
                      network,
                      NUM_AGENTS,
                      LR,
                      LR_LINEAR_DECAY,
                      NUM_UPDATES_DECAY,
                      NUM_MINIBATCHES,
                      NUM_EPOCHS,
                      MAX_GRAD_NORM):
    rngs = jax.random.split(rng, NUM_AGENTS)
    lr_scheduler = optax.linear_schedule(
            init_value=LR,
            end_value=1e-20,
            transition_steps=(NUM_UPDATES_DECAY)
            * NUM_MINIBATCHES
            * NUM_EPOCHS,
        )
    lr = lr_scheduler if LR_LINEAR_DECAY else LR
    batched_train_states = jax.vmap(lambda r: create_agent(env, network, lr, r, MAX_GRAD_NORM))(rngs)
    return batched_train_states, lr

def _compute_targets(last_q, q_vals, reward, done, GAMMA, LAMBDA):
    def _get_target(lambda_returns_and_next_q, rew_q_done):
        reward, q, done = rew_q_done
        lambda_returns, next_q = lambda_returns_and_next_q
        target_bootstrap = reward + GAMMA * (1 - done) * next_q
        delta = lambda_returns - next_q
        lambda_returns = (
            target_bootstrap + GAMMA * LAMBDA * delta
        )
        lambda_returns = (1 - done) * lambda_returns + done * reward
        next_q = jnp.max(q, axis=-1)
        return (lambda_returns, next_q), lambda_returns

    lambda_returns = reward[-1] + GAMMA * (1 - done[-1]) * last_q
    last_q = jnp.max(q_vals[-1], axis=-1)
    _, targets = jax.lax.scan(
        _get_target,
        (lambda_returns, last_q),
        jax.tree.map(lambda x: x[:-1], (reward, q_vals, done)),
        reverse=True,
    )
    targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
    return targets

def agent_update(train_state, minibatch, target, network):
    def loss_fn(params):
        q_vals, updates = network.apply(
            {"params": params, "batch_stats": train_state.batch_stats},
            minibatch.obs,
            train=True,
            mutable=["batch_stats"]
        )
        chosen_q = jnp.take_along_axis(q_vals, jnp.expand_dims(minibatch.action, axis=-1), axis=-1).squeeze(-1)
        loss = 0.5 * jnp.mean((chosen_q - target) ** 2)
        return loss, updates["batch_stats"]
    (loss, new_bs), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)
    new_train_state = train_state.replace(
        params=new_params,
        batch_stats=new_bs,
        opt_state=new_opt_state,
        grad_steps=train_state.grad_steps + 1
    )
    return loss, new_train_state


###########################
#  code I need to know  #
###########################

def get_lr(train_state, lr_schedule):
    """
    Return the scalar learning-rate that was used **this** optimiser step.
    Works for both constant and scheduled LR.
    """
    # chain = (clip-grad, radam)  -> radam state is the second element
    radam_state = train_state.opt_state.inner_states[1]
    step        = radam_state.count              # optimiser step
    return lr_schedule(step) if callable(lr_schedule) else lr_schedule

def save_config(config):
    save_dir = os.path.join(config["SAVE_PATH"], config["RUN_ID"])
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(config,os.path.join(save_dir, f"config.yaml"))

def save_mix_checkpoint(config, model_state, step):
    from jaxmarl.wrappers.baselines import save_params
    save_mix_dir = os.path.join(config["SAVE_PATH"], config["RUN_ID"], 'mix_checkpoints')
    os.makedirs(save_mix_dir, exist_ok=True)
    params = model_state.params
    batch_stats = model_state.batch_stats
    save_params({"params": params, "batch_stats": batch_stats}, os.path.join(save_mix_dir, f'step_{step}.safetensors'))
    
def save_big_checkpoint(config, model_state, step):
    from jaxmarl.wrappers.baselines import save_params
    save_big_dir = os.path.join(config["SAVE_PATH"], config["RUN_ID"], 'big_checkpoints')
    os.makedirs(save_big_dir, exist_ok=True)
    params = model_state.params
    batch_stats = model_state.batch_stats
    save_params({"params": params, "batch_stats": batch_stats}, os.path.join(save_big_dir, f'step_{step}.safetensors'))
    
def orchestrate_mq_train(config):
    # ---- assert ----
    # check the save every is divisible by the number of updates, or the checkpoints will not be saved.
    # assert config["mix"]["NUM_UPDATES"] % config["mix"]["SAVE_CHECKPOINT_EVERY"] == 0
    # assert config["big"]["mid_rounds"]["NUM_UPDATES"] % config["big"]["SAVE_CHECKPOINT_EVERY"] == 0
    # assert config["big"]["final_round"]["NUM_UPDATES"] % config["big"]["SAVE_CHECKPOINT_EVERY"] == 0
    save_config(config)
    
    rng = jax.random.PRNGKey(config["SEED"])
    initialization_mix_rng, initialization_big_rng, train_rng = jax.random.split(rng, 3)
    env, obs, env_state, mix_eps_scheduler, network, mix_agent_train_states, config, mix_buffer, mix_buffer_index, mix_buffer_filled, mix_lr_function = initialize_mix(config, initialization_mix_rng)
    big_agent_train_states, big_lr_function, big_eps_scheduler_mid_rounds, big_eps_scheduler_final_round, big_buffer, big_buffer_index, big_buffer_filled = initialize_big_agent(config, env, initialization_big_rng, network) 
        
    for i in range(config["mq"]["rounds"]):
        print(f"MQ Round {i} ------------------------------")
        # ---- define functions ----
        def train_mix_fn(train_states, rng, obs, env_state, mix_buffer, mix_buffer_index, mix_buffer_filled):
            return train(env, obs, env_state, train_states, network, mix_eps_scheduler, mix_buffer, mix_buffer_index, mix_buffer_filled,
                         config["mix"]["BUFFER_PER_AGENT"], config["NUM_STEPS"], config["mix"]["NUM_AGENTS"], config["NUM_ENVS"], config["TEST_ENVS"],
                         config["mix"]["NUM_UPDATES"], config["NUM_EPOCHS"], config["NUM_MINIBATCHES"],config.get("TEST_DURING_TRAINING", False), 
                         config["WANDB_MODE"], config["REW_SCALE"], config["GAMMA"], config["LAMBDA"], mix_lr_function, rng, log_prefix='mix')
            
        def fill_big_buffer_fn(train_states, rng, obs, env_state, big_buffer, big_buffer_index, big_buffer_filled):
            return fill_big_agent_buffer_from_mix(config, env, network, big_eps_scheduler_mid_rounds, rng, train_states, obs, env_state,
                                                  big_buffer, big_buffer_index, big_buffer_filled)
            
        def train_big_fn(train_states, rng, obs, env_state, big_buffer, big_buffer_index, big_buffer_filled):
            n_updates = 1
            return train(env, obs, env_state, train_states, network, big_eps_scheduler_mid_rounds,
                         big_buffer, big_buffer_index, big_buffer_filled, config["big"]["BUFFER_SIZE"], config["NUM_STEPS"],
                         1, config["NUM_ENVS"] * config["mix"]["NUM_AGENTS"], config["TEST_ENVS"] * config["mix"]["NUM_AGENTS"],
                         n_updates, config["NUM_EPOCHS"], config["NUM_MINIBATCHES"],
                         config.get("TEST_DURING_TRAINING", False), config["WANDB_MODE"], config["REW_SCALE"], config["GAMMA"],
                         config["LAMBDA"], big_lr_function, rng, log_prefix='big')
            
        def train_mix_big_fn(big_agent_train_states, mix_agent_train_states, rng, obs, env_state, big_buffer, big_buffer_index, big_buffer_filled):
            def fn(carry, _):
                big_agent_train_states, (obs, env_state), rng, big_buffer, big_buffer_index, big_buffer_filled = carry
                
                # ---- fill the big buffer ----
                print(f"before filling big buffer, big_buffer_index: {big_buffer_index}, big_buffer_filled: {big_buffer_filled}")
                
                big_fill_rng = rax.fold_in(rng, 1)        
                big_buffer, big_buffer_index, big_buffer_filled, (obs, env_state) = fill_big_buffer_fn(mix_agent_train_states, big_fill_rng, obs, env_state, big_buffer, big_buffer_index, big_buffer_filled)
                
                print(f"after filling big buffer, big_buffer_index: {big_buffer_index}, big_buffer_filled: {big_buffer_filled}")
                
                
                # ---- train the big agent on transitions ----
                big_train_rng = rax.fold_in(rng, 2)
                outs = train_big_fn(big_agent_train_states, big_train_rng, obs, env_state, big_buffer, big_buffer_index, big_buffer_filled)
                runner_state = outs['runner_state']
                (big_agent_train_states, (obs, env_state), big_test_metrics, rng, big_buffer, big_buffer_index, big_buffer_filled) = runner_state
                
                return (big_agent_train_states, (obs, env_state), rng, big_buffer, big_buffer_index, big_buffer_filled), {'big_test_metrics': big_test_metrics}
            
            (big_agent_train_states, (obs, env_state), _rng, big_buffer, big_buffer_index, big_buffer_filled), big_test_metrics = jax.lax.scan(
                fn, (big_agent_train_states, (obs, env_state), rng, big_buffer, big_buffer_index, big_buffer_filled), (), config["big"]["mid_rounds"]["NUM_UPDATES"])
        
            return big_agent_train_states, (obs, env_state), big_test_metrics, _rng, big_buffer, big_buffer_index, big_buffer_filled
            
            
        jitted_train_mix = jax.jit(train_mix_fn)
        jitted_fill_big_buffer = jax.jit(fill_big_buffer_fn)
        jitted_train_big = jax.jit(train_big_fn)
        
        # ---- train the mix ----
        step_rng = rax.fold_in(train_rng, i)
        outs = jitted_train_mix(mix_agent_train_states, step_rng, obs, env_state, mix_buffer, mix_buffer_index, mix_buffer_filled)
        runner_state = outs['runner_state']
        (mix_agent_train_states, (obs, env_state), mix_test_metrics, _rng, mix_buffer, mix_buffer_index, mix_buffer_filled) = runner_state
        
        print(f"buffer_index: {mix_buffer_index}, buffer_filled: {mix_buffer_filled}")
        
        # ---- save the mix agent checkpoint ----
        mix_updates = mix_agent_train_states.n_updates[0]
        print(f"mix_updates: {mix_updates}")
        if mix_updates % config["mix"]["SAVE_CHECKPOINT_EVERY"] == 0:
            print(f"Saving checkpoint at step {mix_updates}")
            save_mix_checkpoint(config, mix_agent_train_states, mix_updates)
        
        for _ in range(config["big"]["mid_rounds"]["NUM_UPDATES"]):
            # ---- fill the big buffer ----
            print(f"before filling big buffer, big_buffer_index: {big_buffer_index}, big_buffer_filled: {big_buffer_filled}")
            
            big_fill_rng = rax.fold_in(step_rng, 1)        
            big_buffer, big_buffer_index, big_buffer_filled, (obs, env_state) = jitted_fill_big_buffer(mix_agent_train_states, big_fill_rng, obs, env_state, big_buffer, big_buffer_index, big_buffer_filled)
            
            print(f"after filling big buffer, big_buffer_index: {big_buffer_index}, big_buffer_filled: {big_buffer_filled}")
            
            
            # ---- train the big agent on transitions ----
            big_train_rng = rax.fold_in(step_rng, 2)
            outs = jitted_train_big(big_agent_train_states, big_train_rng, obs, env_state, big_buffer, big_buffer_index, big_buffer_filled)
            runner_state = outs['runner_state']
            (big_agent_train_states, (obs, env_state), big_test_metrics, _rng, big_buffer, big_buffer_index, big_buffer_filled) = runner_state
            
        # ---- save the big agent checkpoint ----
        big_updates = big_agent_train_states.n_updates[0]
        if big_updates % config["big"]["SAVE_CHECKPOINT_EVERY"] == 0:
            print(f"Saving checkpoint at step {big_updates}")
            save_big_checkpoint(config, big_agent_train_states, big_updates)
        print(f"big_updates: {big_updates}")
    
        print(f"after training big agent, buffer_index: {big_buffer_index}, buffer_filled: {big_buffer_filled}")
            
        # ---- now, clone the big agent weights to all the mix agents ----
        # TODO: implement this

        
        

        
        
    return outs

def make_env(num_envs, config):
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

def initialize_mix(config, rng):
    num_envs = config["NUM_ENVS"]
    num_agents = config["mix"]["NUM_AGENTS"]
    assert (config["NUM_STEPS"] * num_envs) % config["NUM_MINIBATCHES"] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    envs_per_agent = num_envs + config["TEST_ENVS"] if config.get("TEST_DURING_TRAINING", False) else num_envs
    total_envs = (envs_per_agent * num_agents)
    env = make_env(total_envs, config)

    # here reset must be out of vmap and jit
    init_obs, env_state = env.reset()
    

    eps_scheduler = optax.linear_schedule(config["EPS_START"], config["EPS_FINISH"], (config["mix"]["NUM_UPDATES_DECAY"]))
    # INIT NETWORK AND OPTIMIZER
    network = QNetwork(action_dim=env.single_action_space.n, norm_type=config["NORM_TYPE"], norm_input=config.get("NORM_INPUT", False))

    agent_train_states, lr_function = initialize_agents(env,
                                                        rng,
                                                        network,
                                                        config["mix"]["NUM_AGENTS"],
                                                        config["LR"],
                                                        config["LR_LINEAR_DECAY"],
                                                        config["mix"]["NUM_UPDATES_DECAY"],
                                                        config["NUM_MINIBATCHES"],
                                                        config["NUM_EPOCHS"],
                                                        config["MAX_GRAD_NORM"])
    
    B = config["mix"]["BUFFER_PER_AGENT"]
    S = config["NUM_STEPS"]
    A = config["mix"]["NUM_AGENTS"]
     # buffer
    buffer = Transition(
        obs=jnp.zeros((A, B, S, *env.single_observation_space.shape), dtype=jnp.uint8),
        action=jnp.zeros((A, B, S), dtype=jnp.int32),
        reward=jnp.zeros((A, B, S), dtype=jnp.float32),
        done=jnp.zeros((A, B, S), dtype=jnp.bool_),
        next_obs=jnp.zeros((A, B, S, *env.single_observation_space.shape), dtype=jnp.uint8),
        q_val=jnp.zeros((A, B, S, env.single_action_space.n), dtype=jnp.float32) #TODO: remove this
    )
    
    buffer_index = jnp.array(0)
    buffer_filled = jnp.array(0)
    
    return env, init_obs, env_state, eps_scheduler, network, agent_train_states, config, buffer, buffer_index, buffer_filled, lr_function

def initialize_big_agent(config, env, rng, network):
    
    eps_scheduler_mid_rounds = optax.linear_schedule(config["EPS_START"], config["EPS_FINISH"], (config["big"]["mid_rounds"]["NUM_UPDATES_DECAY"]))
    eps_scheduler_final_round = optax.linear_schedule(config["EPS_START"], config["EPS_FINISH"], (config["big"]["final_round"]["NUM_UPDATES_DECAY"]))
    agent_train_states, lr_function = initialize_agents(env, 
                                                        rng,
                                                        network,
                                                        1,
                                                        config["LR"],
                                                        config["LR_LINEAR_DECAY"],
                                                        config["big"]["mid_rounds"]["NUM_UPDATES_DECAY"], #TODO: remove the lr decay in general, but this one just works for mid rounds.
                                                        config["NUM_MINIBATCHES"],
                                                        config["NUM_EPOCHS"],
                                                        config["MAX_GRAD_NORM"])

    B = config["big"]["BUFFER_SIZE"]
    S = config["NUM_STEPS"]
    A = 1
     # buffer
    buffer = Transition(
        obs=jnp.zeros((A, B, S, *env.single_observation_space.shape), dtype=jnp.uint8),
        action=jnp.zeros((A, B, S), dtype=jnp.int32),
        reward=jnp.zeros((A, B, S), dtype=jnp.float32),
        done=jnp.zeros((A, B, S), dtype=jnp.bool_),
        next_obs=jnp.zeros((A, B, S, *env.single_observation_space.shape), dtype=jnp.uint8),
        q_val=jnp.zeros((A, B, S, env.single_action_space.n), dtype=jnp.float32) #TODO: remove this
    )
    buffer_index = jnp.array(0)
    buffer_filled = jnp.array(0)
    
    return agent_train_states, lr_function, eps_scheduler_mid_rounds, eps_scheduler_final_round, buffer, buffer_index, buffer_filled

def fill_big_agent_buffer_from_mix(config,
                             env,
                             network,
                             eps_scheduler, #TODO: should we use the eps scheduler or just a constant?
                             rng,
                             agent_train_states,
                             obs,
                             env_state,
                             buffer,
                             buffer_index,
                             buffer_filled):
    NUM_ENVS = config["NUM_ENVS"]
    TEST_ENVS = config["TEST_ENVS"]
    NUM_AGENTS = config["mix"]["NUM_AGENTS"]
    NUM_STEPS = config["NUM_STEPS"]
    TEST_DURING_TRAINING = config["TEST_DURING_TRAINING"]
    REW_SCALE = config["REW_SCALE"]
    
    expl_state = (obs, env_state)
   # ---- generate trajectories from the mix ----  
    envs_per_agent = NUM_ENVS + TEST_ENVS if TEST_DURING_TRAINING else NUM_ENVS
    # SAMPLE PHASE
    def _step_env(carry, _):
        last_obs, env_state, rng = carry
        rng, rng_a, rng_s = jax.random.split(rng, 3)
        q_vals = jax.vmap(
            lambda ts, obs: network.apply({"params": ts.params, "batch_stats": ts.batch_stats}, obs, train=False)
        )(agent_train_states, last_obs.reshape((NUM_AGENTS, envs_per_agent, *last_obs.shape[1:])))

        def get_eps_per_agent(n_updates):
            eps = eps_scheduler(n_updates)
            eps_train = jnp.full((NUM_ENVS,), eps)
            eps_test = jnp.zeros((TEST_ENVS,))
            return jnp.concatenate([eps_train, eps_test], axis=0)
        
        # different eps for each env
        eps_values = jax.vmap(get_eps_per_agent)(agent_train_states.n_updates)
        rng_a_batched = jnp.repeat(jnp.expand_dims(rng_a, axis=0), NUM_AGENTS, axis=0)
        _rngs = jax.vmap(lambda key: jax.random.split(key, NUM_ENVS + TEST_ENVS))(rng_a_batched)
        new_action = jax.vmap(lambda rngs, q, eps: jax.vmap(eps_greedy_exploration)(rngs, q, eps))(_rngs, q_vals, eps_values)
        new_action = new_action.reshape((NUM_AGENTS*envs_per_agent, *new_action.shape[2:]))
        q_vals = q_vals.reshape((NUM_AGENTS*envs_per_agent, *q_vals.shape[2:]))

        new_obs, new_env_state, reward, new_done, info = env.step(
            env_state, new_action
        )

        transition = Transition(
            obs=last_obs,
            action=new_action,
            reward=REW_SCALE * reward,
            done=new_done,
            next_obs=new_obs,
            q_val=q_vals,
        )
        return (new_obs, new_env_state, rng), (transition, info)

    # step the env
    rng, _rng = jax.random.split(rng)
    (*expl_state, rng), (transitions, infos) = jax.lax.scan(
        _step_env,
        (*expl_state, _rng),
        None,
        NUM_STEPS,
    )
    expl_state = tuple(expl_state)
    
    if TEST_DURING_TRAINING: # remove the testing envs from the transitions
        def filter_and_return_train(x):
            num_agents = NUM_AGENTS
            num_steps = NUM_STEPS
            envs_per_agent = NUM_ENVS + TEST_ENVS if TEST_DURING_TRAINING else NUM_ENVS
            x = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
            x = x.reshape((num_agents, envs_per_agent, num_steps) + x.shape[2:])
            x = x[:, : -TEST_ENVS] # the only difference from the filter_and_return_test is this line and the next one (as the number of envs is different)
            x = x.reshape((num_agents*NUM_ENVS, num_steps) + x.shape[3:])
            return jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
        
        
        infos = jax.tree_map(lambda x: filter_and_return_train(x), infos)
        transitions = jax.tree.map(lambda x: filter_and_return_train(x), transitions)
        
        
    # --- update the buffer ---
    num_envs = transitions.obs.shape[1]
    buffer = jax.tree.map(lambda x, y: update_buffer(x, y, buffer_index, 1), buffer, transitions) #TODO: check if this is working as expected, this is 1 because big agent is a buffer of 1.
    buffer_index += num_envs
    buffer_index = buffer_index % config["big"]["BUFFER_SIZE"]
    buffer_filled = jnp.minimum(buffer_filled+num_envs, config["big"]["BUFFER_SIZE"])
    
    return buffer, buffer_index, buffer_filled, expl_state
    
    
    

def train(
        env,
        init_obs,
        env_state,
        agent_train_states,
        network,
        eps_scheduler,
        buffer,
        buffer_index,
        buffer_filled,
        BUFFER_PER_AGENT,
        NUM_STEPS,
        NUM_AGENTS,
        NUM_ENVS,
        TEST_ENVS,
        NUM_UPDATES,
        NUM_EPOCHS,
        NUM_MINIBATCHES,
        TEST_DURING_TRAINING,
        WANDB_MODE,
        REW_SCALE,
        GAMMA,
        LAMBDA,
        lr_function,
        rng,
        log_prefix=''):
    original_seed = rng[0]

    # params_copy
    params_copy = jax.tree.map(lambda x: x.copy(), agent_train_states.params)
    

    # TRAINING LOOP
    def _update_step(runner_state, update_step):
        envs_per_agent = NUM_ENVS + TEST_ENVS if TEST_DURING_TRAINING else NUM_ENVS
        agent_train_states, expl_state, test_metrics, rng, buffer, buffer_index, buffer_filled = runner_state
        # SAMPLE PHASE
        def _step_env(carry, _):
            last_obs, env_state, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = jax.vmap(
                lambda ts, obs: network.apply({"params": ts.params, "batch_stats": ts.batch_stats}, obs, train=False)
            )(agent_train_states, last_obs.reshape((NUM_AGENTS, envs_per_agent, *last_obs.shape[1:])))

            def get_eps_per_agent(n_updates):
                eps = eps_scheduler(n_updates)
                eps_train = jnp.full((NUM_ENVS,), eps)
                eps_test = jnp.zeros((TEST_ENVS,))
                return jnp.concatenate([eps_train, eps_test], axis=0)
            
            # different eps for each env
            eps_values = jax.vmap(get_eps_per_agent)(agent_train_states.n_updates)
            rng_a_batched = jnp.repeat(jnp.expand_dims(rng_a, axis=0), NUM_AGENTS, axis=0)
            _rngs = jax.vmap(lambda key: jax.random.split(key, NUM_ENVS + TEST_ENVS))(rng_a_batched)
            new_action = jax.vmap(lambda rngs, q, eps: jax.vmap(eps_greedy_exploration)(rngs, q, eps))(_rngs, q_vals, eps_values)
            new_action = new_action.reshape((NUM_AGENTS*envs_per_agent, *new_action.shape[2:]))
            q_vals = q_vals.reshape((NUM_AGENTS*envs_per_agent, *q_vals.shape[2:]))

            new_obs, new_env_state, reward, new_done, info = env.step(
                env_state, new_action
            )

            transition = Transition(
                obs=last_obs,
                action=new_action,
                reward=REW_SCALE * reward,
                done=new_done,
                next_obs=new_obs,
                q_val=q_vals,
            )
            return (new_obs, new_env_state, rng), (transition, info)

        # step the env
        rng, _rng = jax.random.split(rng)
        (*expl_state, rng), (transitions, infos) = jax.lax.scan(
            _step_env,
            (*expl_state, _rng),
            None,
            NUM_STEPS,
        )
        expl_state = tuple(expl_state)
        
        if TEST_DURING_TRAINING:
            def filter_and_return_train(x):
                num_agents = NUM_AGENTS
                num_steps = NUM_STEPS
                envs_per_agent = NUM_ENVS + TEST_ENVS if TEST_DURING_TRAINING else NUM_ENVS
                x = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
                x = x.reshape((num_agents, envs_per_agent, num_steps) + x.shape[2:])
                x = x[:, : -TEST_ENVS] # the only difference from the filter_and_return_test is this line and the next one (as the number of envs is different)
                x = x.reshape((num_agents*NUM_ENVS, num_steps) + x.shape[3:])
                return jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
            
            def filter_and_return_test(x):
                num_agents = NUM_AGENTS
                num_steps = NUM_STEPS
                envs_per_agent = NUM_ENVS + TEST_ENVS if TEST_DURING_TRAINING else NUM_ENVS
                x = jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
                x = x.reshape((num_agents, envs_per_agent, num_steps) + x.shape[2:])
                x = x[:, -TEST_ENVS:]
                x = x.reshape((num_agents*TEST_ENVS, num_steps) + x.shape[3:])
                return jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))
            
            train_infos = jax.tree_map(lambda x: filter_and_return_train(x), infos)
            test_infos = jax.tree_map(lambda x: filter_and_return_test(x), infos)
            infos = train_infos # to support both cases where we have test envs or not
            
            # remove the testing envs from the transitions
            transitions = jax.tree.map(lambda x: filter_and_return_train(x), transitions)
            
            
        # update the buffer
        num_envs_per_agent = transitions.obs.shape[1] // NUM_AGENTS
        buffer = jax.tree.map(lambda x, y: update_buffer(x, y, buffer_index, NUM_AGENTS), buffer, transitions)
        buffer_index += num_envs_per_agent
        buffer_index = buffer_index % BUFFER_PER_AGENT
        buffer_filled = jnp.minimum(buffer_filled+num_envs_per_agent, BUFFER_PER_AGENT)
        
        # sample transitions to train on from the buffer
        sample_indices = jax.random.randint(rng, (NUM_AGENTS, num_envs_per_agent), 0, buffer_filled)
        transitions = jax.vmap(lambda x, indices: jax.tree.map(lambda x: x[indices, ...], x))(buffer, sample_indices)
        # from (num_agents, num_envs_per_agent, num_steps, ...) to (num_steps, num_agents*num_envs_per_agent, ...)
        transitions = jax.tree.map(lambda x: jnp.transpose(x, (2, 0, 1) + tuple(range(3, x.ndim))), transitions)
        transitions = jax.tree.map(lambda x: x.reshape((NUM_STEPS, NUM_AGENTS*num_envs_per_agent) + x.shape[3:]), transitions)

        agent_train_states = agent_train_states.replace(
            timesteps=agent_train_states.timesteps
            + NUM_STEPS * NUM_AGENTS # TODO: should we include the num_agents?
        )  # update timesteps count
        
        
        def _compute_q_vals(carry, last_obs):
            q_vals = jax.vmap(
                lambda ts, obs: network.apply({"params": ts.params, "batch_stats": ts.batch_stats}, obs, train=False)
            )(agent_train_states, last_obs.reshape((NUM_AGENTS, num_envs_per_agent, *last_obs.shape[1:])))
            q_vals = q_vals.reshape((NUM_AGENTS*num_envs_per_agent, *q_vals.shape[2:]))
            return None, q_vals
        
        _, q_vals = jax.lax.scan(
            _compute_q_vals,
            None, # carry
            transitions.obs,
            NUM_STEPS,
        )
        
        # recomputing the q_vals (FIXME: is this necessary?)
        reshaped_last_obs = transitions.next_obs[-1].reshape((NUM_AGENTS, num_envs_per_agent, *transitions.next_obs[-1].shape[1:]))
        last_q = jax.vmap(
            lambda ts, obs: network.apply({"params": ts.params, "batch_stats": ts.batch_stats}, obs, train=False)
        )(agent_train_states, reshaped_last_obs)
        last_q = jnp.max(last_q, axis=-1).reshape((num_envs_per_agent*NUM_AGENTS))
        
        lambda_targets = _compute_targets(
            last_q, q_vals, transitions.reward, transitions.done, GAMMA, LAMBDA
        )

        # NETWORKS UPDATE
        def _learn_epoch(carry, _):
            agent_train_states, rng = carry
            rng, _rng = jax.random.split(rng)
            minibatches = jax.tree_util.tree_map(lambda x: preprocess_transitions_per_agent(x, _rng, NUM_AGENTS, NUM_MINIBATCHES), transitions)
            minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), minibatches)
            targets = jax.tree_map(lambda x: preprocess_transitions_per_agent(x, _rng, NUM_AGENTS, NUM_MINIBATCHES), lambda_targets)
            targets = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), targets)

            def _learn_phase(carry, minibatch_and_target):
                agent_train_states, rng = carry
                minibatch, target = minibatch_and_target
                def agent_loss_and_update(ts, minibatch, target):
                    def _loss_fn(params):
                        q_vals, updates = network.apply(
                            {"params": params, "batch_stats": ts.batch_stats},
                            minibatch.obs, train=True, mutable=["batch_stats"]
                        )
                        chosen_q = jnp.take_along_axis(q_vals, jnp.expand_dims(minibatch.action, axis=-1), axis=-1).squeeze(-1)
                        loss = 0.5 * jnp.mean(((chosen_q - target) ** 2))
                        return loss, updates["batch_stats"]
                    (loss, new_bs), grads = jax.value_and_grad(_loss_fn, has_aux=True)(ts.params)
                    updates, new_opt_state = ts.tx.update(grads, ts.opt_state, ts.params)
                    new_params = optax.apply_updates(ts.params, updates)
                    return loss, ts.replace(params=new_params, batch_stats=new_bs, opt_state=new_opt_state, grad_steps=ts.grad_steps+1)
                
                losses, new_agent_train_states = jax.vmap(agent_loss_and_update, in_axes=(0,0,0))(
                    agent_train_states, minibatch, target
                )
                return (new_agent_train_states, rng), losses
            rng, _rng = jax.random.split(rng)
            (agent_train_states, rng), losses = jax.lax.scan(_learn_phase, (agent_train_states, rng), (minibatches, targets))

            mean_losses = jnp.mean(losses, axis=0)
            return (agent_train_states, rng), mean_losses

        rng, _rng = jax.random.split(rng)
        (agent_train_states, rng), losses = jax.lax.scan(
            _learn_epoch, (agent_train_states, rng), None, NUM_EPOCHS
        )
        agent_train_states = agent_train_states.replace(n_updates=agent_train_states.n_updates + 1)
        losses = jnp.mean(losses, axis=0)

        env_train_metrics = compute_agent_metrics(infos, NUM_AGENTS)
        if TEST_DURING_TRAINING:
            env_test_metrics = compute_agent_metrics(test_infos, NUM_AGENTS)
        else:
            env_test_metrics = {}
        
        metrics = {}
        # --- logging with dirty hacks ---
        for i in range(NUM_AGENTS):
            eps_val = eps_scheduler(agent_train_states.n_updates[i])
            metrics[f"{log_prefix}_agent_{i}/env_step"] = agent_train_states.timesteps[i]
            metrics[f"{log_prefix}_agent_{i}/update_steps"] = agent_train_states.n_updates[i]
            metrics[f"{log_prefix}_agent_{i}/env_frame"] = agent_train_states.timesteps[i] * env.single_observation_space.shape[0]
            metrics[f"{log_prefix}_agent_{i}/grad_steps"] = agent_train_states.grad_steps[i]
            metrics[f"{log_prefix}_agent_{i}/td_loss"] = losses[i]
            metrics[f"{log_prefix}_agent_{i}/epsilon"] = eps_val #TODO: add lr scheduler
            metrics[f"{log_prefix}_agent_{i}/buffer_filled"] = buffer_filled
            metrics[f"{log_prefix}_agent_{i}/buffer_index"] = buffer_index
            
            metrics[f"{log_prefix}_agent_{i}/lr"] = lr_function(agent_train_states.n_updates[i]) if callable(lr_function) else lr_function #FIXME: Is this actually n_updates that I should use here?
            
            for k, v in env_train_metrics.items():
                metrics[f"{log_prefix}_agent_{i}/train_{k}"] = v[i]
            for k, v in env_test_metrics.items():
                metrics[f"{log_prefix}_agent_{i}/test_{k}"] = v[i]
        
        # report on wandb if required
        if WANDB_MODE != "disabled":

            def callback(metrics):
                step = metrics["mix_agent_0/update_steps"]
                print(f"Logging metrics at step {step} to wandb.")
                wandb.log(metrics, step=step)

            jax.debug.callback(callback, metrics)

        runner_state = (agent_train_states, tuple(expl_state), test_metrics, rng, buffer, buffer_index, buffer_filled)

        return runner_state, metrics

    # test metrics not supported yet
    test_metrics = None

    # train
    rng, _rng = jax.random.split(rng)
    expl_state = (init_obs, env_state)
    
    runner_state = (agent_train_states, expl_state, test_metrics, _rng, buffer, buffer_index, buffer_filled)

    base_update_step = agent_train_states.n_updates[0]
    update_steps = jnp.arange(NUM_UPDATES) + base_update_step
    runner_state, metrics = jax.lax.scan(
        _update_step, runner_state, update_steps, NUM_UPDATES
    )

    return {"runner_state": runner_state,
            "metrics": metrics}

    

def single_run(config):
    # --- config ---
    config = {**config, **config["alg"]}
    assert config["NUM_SEEDS"] == 1, "Vmapped seeds not supported yet."

    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]

    # --- wandb ---
    wandb.init(
        id=config["RUN_ID"],
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        config=config,
        mode=config["WANDB_MODE"],
    )

    # --- train ---
    rng = jax.random.PRNGKey(config["SEED"])    
    outs = orchestrate_mq_train(config)

    # --- save params ---
    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], config["RUN_ID"])
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f"{alg_name}_{env_name}_seed{config['SEED']}_config.yaml"
            ),
        )

        params = model_state.params
        save_path = os.path.join(
            save_dir,
            f"{alg_name}_{env_name}_seed{config['SEED']}.safetensors",
        )
        batch_stats = model_state.batch_stats
        save_params({"params": params, "batch_stats": batch_stats}, save_path)

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    
       
    if config["DEBUG"]:
        jax.config.update("jax_disable_jit", True)
        import debugpy
        debugpy.listen(5678)
        print("Waiting for client to attach...")
        debugpy.wait_for_client()
        print("Client attached")
    
    
    single_run(config)


if __name__ == "__main__":
    main()
