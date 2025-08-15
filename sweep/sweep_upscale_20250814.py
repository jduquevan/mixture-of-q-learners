from sweep_utils import submit_slurm_job, get_compute_command
### Jobs ###
BASE_RUN_ID = "20250814_pqn_upscale"
BASE_PYTHON_COMMAND = "python purejaxql/pqn_atari.py" \
            " RUN_ID={run_id}" \
            " WANDB_TAGS={wandb_tags}" \
            " SEED={seed}" \
            " cluster@_global_={cluster}" \
            " alg=pqn_atari" \
            " alg.ENV_NAME={env_name}" \
            " alg.NUM_ENVS={num_envs}" \
            " +alg.ENV_KWARGS.img_width={img_width}" \
            " +alg.ENV_KWARGS.img_height={img_height}" \
            " alg.TOTAL_TIMESTEPS={total_timesteps}" \
            " alg.EPS_FINISH={eps_finish}" \
            " alg.IS_SARSA=False" \
            " alg.SCALE_FACTOR={scale_factor}" \
                
def get_job(env_name: str,
            num_envs: int,
            img_size: int,
            total_timesteps: int,
            eps_finish: float,
            scale_factor: int):
    
    python_command = BASE_PYTHON_COMMAND
    python_kwargs = {
        'env_name': env_name,
        'num_envs': num_envs,
        'img_width': img_size,
        'img_height': img_size,
        'total_timesteps': total_timesteps,
        'eps_finish': eps_finish,
        'scale_factor': scale_factor,
    }
    return python_command, python_kwargs

def get_jobs():
    jobs = {
        # 'battle_zone_pqn_atari_84x84_512envs': get_job(env_name='BattleZone-v5', num_envs=512, img_size=84, total_timesteps=25e7, eps_finish=0.001),
        # 'battle_zone_pqn_atari_400x400_512envs': get_job(env_name='BattleZone-v5', num_envs=512, img_size=400, total_timesteps=25e7, eps_finish=0.001),
        # 'double_dunk_pqn_atari_84x84_512envs': get_job(env_name='DoubleDunk-v5', num_envs=512, img_size=84, total_timesteps=25e7, eps_finish=0.001),
        # 'double_dunk_pqn_atari_400x400_512envs': get_job(env_name='DoubleDunk-v5', num_envs=512, img_size=400, total_timesteps=25e7, eps_finish=0.001),
        # 'montezuma_pqn_atari_400x400_512envs_0136eps': get_job(env_name='MontezumaRevenge-v5', num_envs=512, img_size=400, total_timesteps=25e7, eps_finish=0.0136),
        'gravitar_pqn_atari_84x84_512envs_0136eps_scaleX5': get_job(env_name='Gravitar-v5', num_envs=512, img_size=84, total_timesteps=25e7, eps_finish=0.0136, scale_factor=5),
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
