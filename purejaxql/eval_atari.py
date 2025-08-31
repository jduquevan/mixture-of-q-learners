import jax
import jax.numpy as jnp
import numpy as np
import envpool
from di_atari import JaxLogEnvPoolWrapper, Transition, eps_greedy_exploration, QNetwork
from jaxmarl.wrappers.baselines import load_params
import hydra
from omegaconf import OmegaConf
import json

def make_eval(config, num_envs, num_steps, return_obs=False):
    
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
    
    env = make_env(num_envs)
    
    init_obs, init_env_state = env.reset()
    
    def eval(ckpt_params, ckpt_batch_stats, rng):
                
        network = QNetwork(
                action_dim=env.single_action_space.n,
                norm_type=config["NORM_TYPE"],
                norm_input=config.get("NORM_INPUT", False),
            )
        
            
        def _step_env(carry, eval_step):
            prev_obs, env_state, rng = carry
            rng, rng_a = jax.random.split(rng, 2)
            q_vals = network.apply({"params": ckpt_params, "batch_stats": ckpt_batch_stats}, prev_obs, train=False)
            eps = jnp.zeros((num_envs,))
            _rngs = jax.random.split(rng_a, num_envs)
            new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
            
            new_obs, new_env_state, reward, new_done, info = env.step(env_state, new_action)
            
            def callback(x):
                if x % 100 == 0:
                    print(f'eval_step: {x}')
            jax.debug.callback(callback, eval_step)
            
            if return_obs:
                return (new_obs, new_env_state, rng), (prev_obs, info)
            else:
                return (new_obs, new_env_state, rng), info
            
        _, outs = jax.lax.scan(
            _step_env,
            (init_obs, init_env_state, rng),
            jnp.arange(num_steps),
        )
        
        if return_obs:
            obs, infos = outs
        else:
            infos = outs
            
        # metrics = compute_agent_metrics(infos, config)
        
        if return_obs:
            return obs#, metrics
        else:
            return # metrics
        
    
    return eval

def single_run(config):
    print('evaluating...')
    config = {**config, **config["alg"]}
    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]
    print(f'alg_name: {alg_name}, env_name: {env_name}')
    
    load_path = config["LOAD_PATH"]
    ckpt = load_params(load_path)
    ckpt_params = ckpt["params"]
    ckpt_batch_stats = ckpt["batch_stats"]
    # remove the first batch dimension from params and batch stats
    # this because we were training a batch of agents, but now evaluating a single agent.
    # assert config['NUM_AGENTS'] == 1, "Evaluation only supports runs which trained a single agent."
    # ckpt_params = jax.tree_util.tree_map(lambda x: x[0], ckpt_params)
    # ckpt_batch_stats = jax.tree_util.tree_map(lambda x: x[0], ckpt_batch_stats)
    print(f'loaded params and batch stats from {load_path}.')
    
    rng = jax.random.PRNGKey(config["SEED"])
    
    # metrics = jax.jit(make_eval(config, num_envs=config["NUM_ENVS"], num_steps=config["NUM_STEPS"], return_obs=False))(ckpt_params, ckpt_batch_stats, rng)
    # print('metrics:', metrics, sep='\n')
    # # print metrics to text file
    # with open(config["METRICS_NAME"], 'w') as f:
    #     for key, value in metrics.items():
    #         f.write(f'{key}: {value}\n')
    
    print('making video...')
    obs = jax.jit(make_eval(config, num_envs=25, num_steps=config["VIDEO_STEPS"], return_obs=True))(ckpt_params, ckpt_batch_stats, rng)
    obs = jnp.einsum('lbthw->blthw', obs)
    b, l, t, h, w = obs.shape
    obs = obs[:, :, 0, :, :]
    # make it 3x3 grid
    assert b == 25, 'b should be 9'
    obs = jnp.einsum('bThw->bhwT', obs)
    b, h, w, T = obs.shape
    
    import imageio
    with imageio.get_writer(config["VIDEO_NAME"], fps=30, codec='libx264') as vid:
        for t in range(T):
            frame = obs[..., t]
            grid = np.block([[frame[i*5 + j] for j in range(5) ] for i in range(5)])
            frame = np.stack([grid] * 3, axis=-1) 
            vid.append_data(np.array(frame)) 
            
    print('obs.shape:', obs.shape, sep='\n')
    
@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print(config)
    if config["DEBUG"]:
        jax.config.update("jax_disable_jit", True)
        import debugpy
        debugpy.listen(5678)
        print("Waiting for client to attach...")
        debugpy.wait_for_client()
        print("Client attached")
    print("Config:\n", OmegaConf.to_yaml(config))
    
    single_run(config)

if __name__ == "__main__":
    main()