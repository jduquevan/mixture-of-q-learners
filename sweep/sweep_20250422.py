from sweep_utils import submit_slurm_job, get_compute_command
### Jobs ###
BASE_RUN_ID = "20250422_v0"
BASE_PYTHON_COMMAND = "python purejaxql/di_atari.py" \
            " RUN_ID={run_id}" \
            " WANDB_TAGS={wandb_tags}" \
            " SEED={seed}" \
            " cluster@_global_={cluster}" \
            " alg.ENV_NAME=Breakout-v5" \
            " alg.NUM_AGENTS={num_agents}" \
            " alg.NUM_ENVS={num_envs}" \
            " alg.BUFFER_PER_AGENT={buffer_per_agent}" \
            " alg.SHARE_STRATEGY={share_strategy}" \
            " alg.TOTAL_TIMESTEPS_DECAY={total_timesteps_decay}" \
            " alg.RESET_SCHEDULE={reset_schedule}" \
            " alg.MIX_SCHEDULE={mix_schedule}" \
            
def get_breakout_job(num_agents: int, num_envs: int, buffer_per_agent: int, share_strategy: str, total_timesteps_decay: float, reset_schedule: str, mix_schedule: str):
    python_command = BASE_PYTHON_COMMAND
    python_kwargs = {
        'num_agents': num_agents,
        'num_envs': num_envs,
        'buffer_per_agent': buffer_per_agent,
        'share_strategy': share_strategy,
        'total_timesteps_decay': total_timesteps_decay,
        'reset_schedule': reset_schedule,
        'mix_schedule': mix_schedule,
    }
    return python_command, python_kwargs

def get_jobs():
    jobs = {
        'breakout_baseline': get_breakout_job(num_agents=1, num_envs=128, buffer_per_agent=128, share_strategy='no-share', total_timesteps_decay=5e7, reset_schedule='no-reset', mix_schedule='no-mix'),
        'breakout_share_0.2': get_breakout_job(num_agents=4, num_envs=32, buffer_per_agent=32, share_strategy='share-0.2', total_timesteps_decay=5e7, reset_schedule='no-reset', mix_schedule='no-mix'),
        'breakout_share_all': get_breakout_job(num_agents=4, num_envs=32, buffer_per_agent=32, share_strategy='share-all', total_timesteps_decay=5e7, reset_schedule='no-reset', mix_schedule='no-mix'),
        'breakout_baseline_slow_decay': get_breakout_job(num_agents=1, num_envs=128, buffer_per_agent=128, share_strategy='no-share', total_timesteps_decay=40e7, reset_schedule='no-reset', mix_schedule='no-mix'),
        'breakout_share_all_slow_decay': get_breakout_job(num_agents=4, num_envs=32, buffer_per_agent=32, share_strategy='share-all', total_timesteps_decay=40e7, reset_schedule='no-reset', mix_schedule='no-mix'),
        'breakout_share_0.2_slow_decay': get_breakout_job(num_agents=4, num_envs=32, buffer_per_agent=32, share_strategy='share-0.2', total_timesteps_decay=40e7, reset_schedule='no-reset', mix_schedule='no-mix'),
        'breakout_no_share_two_resets_slow_decay': get_breakout_job(num_agents=1, num_envs=128, buffer_per_agent=128, share_strategy='no-share', total_timesteps_decay=40e7, reset_schedule='two-resets-at-4000-and-6000', mix_schedule='no-mix'),
        'breakout_no_share_two_resets_mix_slow_decay': get_breakout_job(num_agents=1, num_envs=128, buffer_per_agent=128, share_strategy='no-share', total_timesteps_decay=40e7, reset_schedule='two-resets-at-4000-and-6000', mix_schedule='two-mix-for-500-at-4000-and-6000'),
        'breakout_no_share_two_resets_slow_decay_buffer_6400': get_breakout_job(num_agents=1, num_envs=128, buffer_per_agent=6400, share_strategy='no-share', total_timesteps_decay=40e7, reset_schedule='two-resets-at-4000-and-6000', mix_schedule='no-mix'),
        'breakout_no_share_two_resets_four_agents_slow_decay_buffer_1600': get_breakout_job(num_agents=4, num_envs=32, buffer_per_agent=1600, share_strategy='no-share', total_timesteps_decay=40e7, reset_schedule='two-resets-at-4000-and-6000', mix_schedule='no-mix'),
        'breakout_no_share_two_resets_four_agents_slow_decay_buffer_1600_mix_schedule': get_breakout_job(num_agents=4, num_envs=32, buffer_per_agent=1600, share_strategy='no-share', total_timesteps_decay=40e7, reset_schedule='two-resets-at-4000-and-6000', mix_schedule='two-mix-for-500-at-4000-and-6000'),
        'breakout_no_share_two_resets_four_agents_slow_decay': get_breakout_job(num_agents=4, num_envs=32, buffer_per_agent=32, share_strategy='no-share', total_timesteps_decay=40e7, reset_schedule='two-resets-at-4000-and-6000', mix_schedule='no-mix'),
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
