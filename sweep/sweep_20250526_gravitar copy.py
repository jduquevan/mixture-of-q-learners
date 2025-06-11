from sweep_utils import submit_slurm_job, get_compute_command
### Jobs ###
BASE_RUN_ID = "20250526_v2"
BASE_PYTHON_COMMAND = "python purejaxql/di_atari.py" \
            " RUN_ID={run_id}" \
            " WANDB_TAGS={wandb_tags}" \
            " SEED={seed}" \
            " cluster@_global_={cluster}" \
            " alg.ENV_NAME=Gravitar-v5" \
            " alg.NUM_AGENTS={num_agents}" \
            " alg.NUM_ENVS={num_envs}" \
            " alg.BUFFER_PER_AGENT={buffer_per_agent}" \
            " alg.SHARE_STRATEGY={share_strategy}" \
            " alg.TOTAL_TIMESTEPS_DECAY={total_timesteps_decay}" \
            " alg.TOTAL_TIMESTEPS={total_timesteps}" \
            " alg.RESET_SCHEDULE={reset_schedule}" \
            " alg.MIX_SCHEDULE={mix_schedule}" \
            
def get_gravitar_job(num_agents: int, num_envs: int, buffer_per_agent: int, share_strategy: str, total_timesteps_decay: float, reset_schedule: str, mix_schedule: str, total_timesteps: float):
    python_command = BASE_PYTHON_COMMAND
    python_kwargs = {
        'num_agents': num_agents,
        'num_envs': num_envs,
        'buffer_per_agent': buffer_per_agent,
        'share_strategy': share_strategy,
        'total_timesteps_decay': total_timesteps_decay,
        'reset_schedule': reset_schedule,
        'mix_schedule': mix_schedule,
        'total_timesteps': total_timesteps,
    }
    return python_command, python_kwargs

def get_jobs():
    jobs = {
        'gravitar_baseline': get_gravitar_job(num_agents=1, num_envs=128, buffer_per_agent=128, share_strategy='no-share', total_timesteps=5e7, total_timesteps_decay=5e7, reset_schedule='no-reset', mix_schedule='no-mix'),
    }
    
    return jobs


### Submit Jobs ###
def submit_job(job_names: str | list[str], seed: int, cluster: str = 'milad_mila', fake_submit: bool = True, compute_type: str = "a100l"):
    jobs = get_jobs()
    if isinstance(job_names, str):
        # Handle both comma-separated strings and single job names
        job_names = [name.strip() for name in job_names.split(',')]
        
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
