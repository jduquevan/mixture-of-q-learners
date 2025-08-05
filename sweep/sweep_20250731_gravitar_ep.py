from sweep_utils import submit_slurm_job, get_compute_command
### Jobs ###
BASE_RUN_ID = "20250731_gravitar_eps_lab"
BASE_PYTHON_COMMAND = "python purejaxql/ep_atari.py" \
            " RUN_ID={run_id}" \
            " WANDB_TAGS={wandb_tags}" \
            " SEED={seed}" \
            " cluster@_global_={cluster}" \
            " alg.ENV_NAME=Gravitar-v5" \
            " alg.NUM_ENVS={num_envs}" \
            " +alg.ENV_KWARGS.img_width={img_width}" \
            " +alg.ENV_KWARGS.img_height={img_height}" \
            " alg.NUM_UPDATES={num_updates}" \
            " alg.NUM_UPDATES_DECAY={num_updates_decay}" \
            " alg.ROUNDS={rounds}" \
            " alg.DEPTH_EXPLORATION_EPS_SCHEDULER={depth_exploration_eps_scheduler}" \
            " alg.ENV_WISE_EXPLORATION_EPS_SCHEDULER={env_wise_exploration_eps_scheduler}" \
            " alg.BUFFER_SIZE={buffer_size}" \
            " alg.NUM_TEST_ENVS={num_test_envs}" \
            " alg.EPS_FINISH={eps_finish}" \
            " alg.EXPLORATION_PERIOD={exploration_period}" \
                
def get_job(num_envs: int, 
                     img_size: int,
                     num_updates: int,
                     num_updates_decay: float,
                     rounds: int,
                     num_test_envs: int,
                     depth_exploration_eps_scheduler: str,
                     env_wise_exploration_eps_scheduler: str,
                     eps_finish: float,
                     exploration_period: int = 1000,
                     ):
    
    python_command = BASE_PYTHON_COMMAND
    python_kwargs = {
        'num_envs': num_envs,
        'img_width': img_size,
        'img_height': img_size,
        'num_updates': num_updates,
        'num_updates_decay': num_updates_decay,
        'rounds': rounds,
        'depth_exploration_eps_scheduler': depth_exploration_eps_scheduler,
        'env_wise_exploration_eps_scheduler': env_wise_exploration_eps_scheduler,
        'buffer_size': num_envs,
        'num_test_envs': num_test_envs,
        'eps_finish': eps_finish,
        'exploration_period': exploration_period,
    }
    return python_command, python_kwargs

def get_jobs():
    jobs = {
        
        'gravitar_baseline_high_eps': get_job(num_envs=256,
                                    num_test_envs=8,
                                    img_size=84,
                                    num_updates=100_000,
                                    num_updates_decay=5_000,
                                    rounds=1,
                                    depth_exploration_eps_scheduler="disabled",
                                    env_wise_exploration_eps_scheduler="disabled",
                                    eps_finish=0.0136),
        
        'gravitar_baseline_high_eps_512': get_job(num_envs=512,
                                    num_test_envs=8,
                                    img_size=400,
                                    num_updates=100_000,
                                    num_updates_decay=5_000,
                                    rounds=1,
                                    depth_exploration_eps_scheduler="disabled",
                                    env_wise_exploration_eps_scheduler="disabled",
                                    eps_finish=0.0136),
        
        'gravitar_baseline': get_job(num_envs=256,
                                    num_test_envs=8,
                                    img_size=84,
                                    num_updates=100_000,
                                    num_updates_decay=5_000,
                                    rounds=1,
                                    depth_exploration_eps_scheduler="disabled",
                                    env_wise_exploration_eps_scheduler="disabled",
                                    eps_finish=0.001),
        
        
        'gravitar_depth_exploration': get_job(num_envs=256,
                                              num_test_envs=8,
                                              img_size=84,
                                              num_updates=100_000,
                                              num_updates_decay=5_000,
                                              rounds=1,
                                              depth_exploration_eps_scheduler="depth_exploration_0.1_U0.3",
                                              env_wise_exploration_eps_scheduler="disabled",
                                              eps_finish=0.001),
        
        'gravitar_depth_exploration_low_eps': get_job(num_envs=256,
                                              num_test_envs=8,
                                              img_size=84,
                                              num_updates=100_000,
                                              num_updates_decay=5_000,
                                              rounds=1,
                                              depth_exploration_eps_scheduler="depth_exploration_0.1_U0.05",
                                              env_wise_exploration_eps_scheduler="disabled",
                                              eps_finish=0.001),
        
        'gravitar_env_wise_exploration': get_job(num_envs=256,
                                              num_test_envs=8,
                                              img_size=84,
                                              num_updates=100_000,
                                              num_updates_decay=5_000,
                                              rounds=1,
                                              depth_exploration_eps_scheduler="disabled",
                                              env_wise_exploration_eps_scheduler="envwise50D50R_U0.3",
                                              eps_finish=0.001),
        
        'gravitar_env_wise_exploration_low_eps': get_job(num_envs=256,
                                              num_test_envs=8,
                                              img_size=84,
                                              num_updates=100_000,
                                              num_updates_decay=5_000,
                                              rounds=1,
                                              depth_exploration_eps_scheduler="disabled",
                                              env_wise_exploration_eps_scheduler="envwise50D50R_U0.05",
                                              eps_finish=0.001),
        
        'gravitar_env_wise_exploration_periodic_1D0R_0.5D0.5R_U0.05': get_job(num_envs=1024,
                                                          num_test_envs=8,
                                                          img_size=84,
                                                          num_updates=100_000,
                                                          num_updates_decay=5_000,
                                                          rounds=1,
                                                          depth_exploration_eps_scheduler="disabled",
                                                          env_wise_exploration_eps_scheduler="envwise_periodic_1D0R_0.5D0.5R_U0.05",
                                                          eps_finish=0.001,
                                                          exploration_period=20000),
        
        'gravitar_depth_and_env_wise_exploration':get_job(num_envs=256,
                                              num_test_envs=8,
                                              img_size=84,
                                              num_updates=100_000,
                                              num_updates_decay=5_000,
                                              rounds=1,
                                              depth_exploration_eps_scheduler="depth_exploration_0.1_U0.3",
                                              env_wise_exploration_eps_scheduler="envwise50D50R_U0.3",
                                              eps_finish=0.001),
        
        'gravitar_depth_and_env_wise_exploration_low_eps':get_job(num_envs=256,
                                                                  num_test_envs=8,
                                                                  img_size=84,
                                                                  num_updates=100_000,
                                                                  num_updates_decay=5_000,
                                                                  rounds=1,
                                                                  depth_exploration_eps_scheduler="depth_exploration_0.1_U0.05",
                                                                  env_wise_exploration_eps_scheduler="envwise50D50R_U0.05",
                                                                  eps_finish=0.001),
        
        'gravitar_env_wise_exploration_low_eps_80D20R': get_job(num_envs=256,
                                              num_test_envs=8,
                                              img_size=84,
                                              num_updates=100_000,
                                              num_updates_decay=5_000,
                                              rounds=1,
                                              depth_exploration_eps_scheduler="disabled",
                                              env_wise_exploration_eps_scheduler="envwise80D20R_U0.05",
                                              eps_finish=0.001),
        
        
    }
    
    return jobs


### Submit Jobs ###
def submit_job(job_names: str | list[str], seeds: list[int], cluster: str = 'milad_mila', fake_submit: bool = True, compute_type: str = "a100l"):
    jobs = get_jobs()
    if isinstance(job_names, str):
        # Handle both comma-separated strings and single job names
        job_names = [name.strip() for name in job_names.split(',')]
    
    for seed in seeds: # because I rather have one seed of multiple jobs than multiple seeds of one job
        for job_name in job_names:
            python_command, python_kwargs = jobs[job_name]
            python_kwargs['seed'] = seed
            python_kwargs['run_id'] = f'{BASE_RUN_ID}_{job_name}_{seed}'
            python_kwargs['cluster'] = cluster
            python_kwargs['wandb_tags'] = "[" + f'{job_name}' + "]"
            
            formatted_python_command = python_command.format(**python_kwargs)
            submit_slurm_job(formatted_python_command, fake_submit=fake_submit, compute_type=compute_type, job_name=job_name)
    
if __name__ == '__main__':
    import fire
    fire.Fire()
