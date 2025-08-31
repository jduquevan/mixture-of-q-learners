from sweep_utils import submit_slurm_job, get_compute_command
### Jobs ###
BASE_RUN_ID = "20250818_pitfall_gallery_rnd_augment"
BASE_PYTHON_COMMAND = "python purejaxql/pqn_atari_gallery.py" \
            " RUN_ID={run_id}" \
            " WANDB_TAGS={wandb_tags}" \
            " SEED={seed}" \
            " cluster@_global_={cluster}" \
            " alg=pqn_atari_gallery" \
            " alg.ENV_NAME=Pitfall-v5" \
            " alg.NUM_ENVS={num_envs}" \
            " +alg.ENV_KWARGS.img_width={img_width}" \
            " +alg.ENV_KWARGS.img_height={img_height}" \
            " alg.TOTAL_TIMESTEPS={total_timesteps}" \
            " alg.EPS_FINISH={eps_finish}" \
            " alg.IS_SARSA=True" \
            " alg.REW_SCALE={rew_scale}" \
            " alg.ENV_KWARGS.episodic_life={episodic_life}" \
            " alg.ENV_KWARGS.reward_clip={reward_clip}" \
            " alg.GALLERY_DEPTH={gallery_depth}" \
            " alg.GALLERY_UPDATE_FREQ={gallery_update_freq}" \
            " alg.GALLERY_REWARD_SCALE={gallery_reward_scale}" \
                
def get_job(num_envs: int,
            img_size: int,
            total_timesteps: int,
            eps_finish: float,
            rew_scale: float,
            episodic_life: bool,
            reward_clip: bool,
            gallery_depth: int,
            gallery_update_freq: int,
            gallery_reward_scale: float):
    
    python_command = BASE_PYTHON_COMMAND
    python_kwargs = {
        'num_envs': num_envs,
        'img_width': img_size,
        'img_height': img_size,
        'total_timesteps': total_timesteps,
        'eps_finish': eps_finish,
        'rew_scale': rew_scale,
        'episodic_life': episodic_life,
        'reward_clip': reward_clip,
        'gallery_depth': gallery_depth,
        'gallery_update_freq': gallery_update_freq,
        'gallery_reward_scale': gallery_reward_scale,
    }
    return python_command, python_kwargs

def get_jobs():
    jobs = {
        'gallery_GF20_GDEPTH100_GS10': get_job(num_envs=512, img_size=160, total_timesteps=25e7, eps_finish=0.01,rew_scale=0.001,
                                               episodic_life=False, reward_clip=False, gallery_depth=100, gallery_update_freq=20, gallery_reward_scale=10.0),
        'gallery_GF20_GDEPTH100_GS1': get_job(num_envs=512, img_size=160, total_timesteps=25e7, eps_finish=0.01,rew_scale=0.001,
                                               episodic_life=False, reward_clip=False, gallery_depth=100, gallery_update_freq=20, gallery_reward_scale=1.0),
        'gallery_GF20_GDEPTH100_GS100': get_job(num_envs=512, img_size=160, total_timesteps=25e7, eps_finish=0.01,rew_scale=0.001,
                                               episodic_life=False, reward_clip=False, gallery_depth=100, gallery_update_freq=20, gallery_reward_scale=100.0),
        'gallery_GF20_GDEPTH100_GS0_short': get_job(num_envs=512, img_size=160, total_timesteps=4e7, eps_finish=0.01,rew_scale=0.001,
                                               episodic_life=False, reward_clip=False, gallery_depth=100, gallery_update_freq=20, gallery_reward_scale=0.0),
        'gallery_GF20_GDEPTH100_GS1_short': get_job(num_envs=512, img_size=160, total_timesteps=4e7, eps_finish=0.01,rew_scale=0.001,
                                               episodic_life=False, reward_clip=False, gallery_depth=100, gallery_update_freq=20, gallery_reward_scale=1.0),
        'gallery_GF40_GDEPTH50_GS100_med': get_job(num_envs=512, img_size=160, total_timesteps=25e7, eps_finish=0.01,rew_scale=0.001,
                                                    episodic_life=False, reward_clip=False, gallery_depth=50, gallery_update_freq=40, gallery_reward_scale=100.0),
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
            print(f'Submitting job: {job_name}')
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
