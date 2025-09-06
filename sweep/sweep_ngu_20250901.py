from sweep_utils import submit_slurm_job, get_compute_command
### Jobs ###
BASE_RUN_ID = "20250901_pqn_atari_ngu"
BASE_PYTHON_COMMAND = "python purejaxql/pqn_atari_ngu.py" \
            " RUN_ID={run_id}" \
            " WANDB_TAGS={wandb_tags}" \
            " SEED={seed}" \
            " cluster@_global_={cluster}" \
            " alg=pqn_atari_ngu" \
            " alg.ENV_NAME={env_name}" \
            " alg.NUM_ENVS={num_envs}" \
            " +alg.ENV_KWARGS.img_width={img_width}" \
            " +alg.ENV_KWARGS.img_height={img_height}" \
            " alg.TOTAL_TIMESTEPS={total_timesteps}" \
            " alg.EPS_FINISH={eps_finish}" \
            " alg.IS_SARSA=False" \
            " alg.NGU.BETA={beta}" \
            " +alg.HIDDEN_SIZE=256" \
                
def get_job(env_name: str,
            num_envs: int,
            img_size: int,
            total_timesteps: int,
            eps_finish: float, 
            beta: float):
    
    python_command = BASE_PYTHON_COMMAND
    python_kwargs = {
        'env_name': env_name,
        'num_envs': num_envs,
        'img_width': img_size,
        'img_height': img_size,
        'total_timesteps': total_timesteps,
        'eps_finish': eps_finish,
        'beta': beta,
    }
    return python_command, python_kwargs

def get_jobs():
    jobs = {
        'montezuma_ngu_400x400_512envs_0136eps_01beta': get_job(env_name='MontezumaRevenge-v5', num_envs=512, img_size=400, total_timesteps=25e7, eps_finish=0.0136, beta=0.01),
        'pitfall_ngu_400x400_512envs_0136eps_01beta': get_job(env_name='Pitfall-v5', num_envs=512, img_size=400, total_timesteps=25e7, eps_finish=0.0136, beta=0.01),
        'pitfall_ngu_400x400_512envs_03eps_01beta': get_job(env_name='Pitfall-v5', num_envs=512, img_size=400, total_timesteps=25e7, eps_finish=0.03, beta=0.01),
    }
    
    return jobs


### Submit Jobs ###
def submit_job(job_names: str | list[str], seeds: list[int], cluster: str = 'milad_mila', fake_submit: bool = True, compute_type: str = "a100l"):
    jobs = get_jobs()
    if isinstance(job_names, str):
        # Handle both comma-separated strings and single job names
        job_names = [name.strip() for name in job_names.split(',')]
    
    for seed in seeds: # I iterate over seeds first, because I rather have one seed of multiple jobs than multiple seeds of one job
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
