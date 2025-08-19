from sweep_utils import submit_slurm_job, get_compute_command
### Jobs ###
BASE_RUN_ID = "20250816_pitfall"
BASE_PYTHON_COMMAND = "python purejaxql/pqn_atari.py" \
            " RUN_ID={run_id}" \
            " WANDB_TAGS={wandb_tags}" \
            " SEED={seed}" \
            " cluster@_global_={cluster}" \
            " alg=pqn_atari" \
            " alg.ENV_NAME=Pitfall-v5" \
            " alg.NUM_ENVS={num_envs}" \
            " +alg.ENV_KWARGS.img_width={img_width}" \
            " +alg.ENV_KWARGS.img_height={img_height}" \
            " alg.TOTAL_TIMESTEPS={total_timesteps}" \
            " alg.EPS_FINISH={eps_finish}" \
            " alg.IS_SARSA=True" \
            " alg.REW_SCALE={rew_scale}" \
            " alg.CURIOSITY_REWARD={curiosity_reward}" \
            " alg.CURIOSITY_REWARD_SCALE={curiosity_reward_scale}" \
            " alg.ENV_KWARGS.episodic_life={episodic_life}" \
            " alg.ENV_KWARGS.reward_clip={reward_clip}" \
                
def get_job(num_envs: int,
            img_size: int,
            total_timesteps: int,
            eps_finish: float,
            rew_scale: float,
            curiosity_reward: str,
            curiosity_reward_scale: float,
            episodic_life: bool,
            reward_clip: bool):
    
    python_command = BASE_PYTHON_COMMAND
    python_kwargs = {
        'num_envs': num_envs,
        'img_width': img_size,
        'img_height': img_size,
        'total_timesteps': total_timesteps,
        'eps_finish': eps_finish,
        'rew_scale': rew_scale,
        'curiosity_reward': curiosity_reward,
        'curiosity_reward_scale': curiosity_reward_scale,
        'episodic_life': episodic_life,
        'reward_clip': reward_clip,
    }
    return python_command, python_kwargs

def get_jobs():
    jobs = {
        # 'sarsa_atari_84x84_512envs': get_job(num_envs=512, img_size=84, total_timesteps=25e7, eps_finish=0.001),
        # 'sarsa_atari_400x400_512envs': get_job(num_envs=512, img_size=400, total_timesteps=25e7, eps_finish=0.001),
        # 'sarsa_atari_84x84_128envs': get_job(num_envs=128, img_size=84, total_timesteps=25e7, eps_finish=0.001),
        # 'sarsa_atari_400x400_512env_0136eps': get_job(num_envs=512, img_size=400, total_timesteps=25e7, eps_finish=0.0136),
        # 'sarsa_160x160_512envs_05eps': get_job(num_envs=512, img_size=160, total_timesteps=25e7, eps_finish=0.05),
        'sarsa_160x160_512envs_0.01eps_notepisodict_rewscale0.001_noclip': get_job(num_envs=512, img_size=160, total_timesteps=25e7, eps_finish=0.01,
                                                                                rew_scale=0.001, curiosity_reward='DISABLED', curiosity_reward_scale=0.0, 
                                                                                episodic_life=False, reward_clip=False),
        'sarsa_160x160_512envs_0.1eps_notepisodict_rewscale0.001_noclip': get_job(num_envs=512, img_size=160, total_timesteps=25e7, eps_finish=0.1,
                                                                                rew_scale=0.001, curiosity_reward='DISABLED', curiosity_reward_scale=0.0, 
                                                                                episodic_life=False, reward_clip=False),
        'sarsa_160x160_512envs_0.1eps_notepisodict_rewscale0.001_noclip_curious': get_job(num_envs=512, img_size=160, total_timesteps=25e7, eps_finish=0.1,
                                                                                rew_scale=0.001, curiosity_reward='MSE_FRAME', curiosity_reward_scale=1.0, 
                                                                                episodic_life=False, reward_clip=False),
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
