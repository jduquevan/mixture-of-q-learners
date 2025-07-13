from sweep_utils import submit_slurm_job, get_compute_command
### Jobs ###
BASE_RUN_ID = "20250709_v7_reset_optimizer_mix_agents"
BASE_PYTHON_COMMAND = "python purejaxql/mq_atari.py" \
            " RUN_ID={run_id}" \
            " WANDB_TAGS={wandb_tags}" \
            " SEED={seed}" \
            " cluster@_global_={cluster}" \
            " alg.ENV_NAME=Gravitar-v5" \
            " alg.mix.NUM_AGENTS={mix_num_agents}" \
            " alg.NUM_ENVS={num_envs}" \
            " alg.mix.BUFFER_PER_AGENT={buffer_per_agent}" \
            " +alg.ENV_KWARGS.img_width={img_width}" \
            " +alg.ENV_KWARGS.img_height={img_height}" \
            " alg.mix.NUM_UPDATES={mix_num_updates}" \
            " alg.mix.NUM_UPDATES_DECAY={mix_num_updates_decay}" \
            " alg.big.mid_rounds.NUM_UPDATES={big_mid_rounds_num_updates}" \
            " alg.big.mid_rounds.NUM_UPDATES_DECAY={big_mid_rounds_num_updates_decay}" \
            " alg.mq.rounds={mq_rounds}" \
            " alg.big.final_round.NUM_UPDATES={big_final_round_num_updates}" \
            " alg.big.BUFFER_SIZE={big_buffer_size}" \
            " alg.big.final_round.NUM_UPDATES_DECAY={big_final_round_num_updates_decay}" \
            
def get_gravitar_job(mix_num_agents: int, 
                     num_envs: int, 
                     buffer_per_agent: int, 
                     img_size: int,
                     mix_num_updates: int,
                     mix_num_updates_decay: float,
                     big_mid_rounds_num_updates: int,
                     big_mid_rounds_num_updates_decay: float,
                     mq_rounds: int,
                     big_final_round_num_updates: int,
                     big_buffer_size: int,
                     big_final_round_num_updates_decay: int):
    python_command = BASE_PYTHON_COMMAND
    python_kwargs = {
        'mix_num_agents': mix_num_agents,
        'num_envs': num_envs,
        'buffer_per_agent': buffer_per_agent,
        'img_width': img_size,
        'img_height': img_size,
        'mix_num_updates': mix_num_updates,
        'mix_num_updates_decay': mix_num_updates_decay,
        'big_mid_rounds_num_updates': big_mid_rounds_num_updates,
        'big_mid_rounds_num_updates_decay': big_mid_rounds_num_updates_decay,
        'mq_rounds': mq_rounds,
        'big_final_round_num_updates': big_final_round_num_updates,
        'big_buffer_size': big_buffer_size,
        'big_final_round_num_updates_decay': big_final_round_num_updates_decay,
    }
    return python_command, python_kwargs

def get_jobs():
    jobs = {
        'gravitar_mq_debug': get_gravitar_job(mix_num_agents=4,
                                              num_envs=32,
                                              buffer_per_agent=32,
                                              img_size=128,
                                              mix_num_updates=10,
                                              mix_num_updates_decay=10,
                                              big_mid_rounds_num_updates=10, 
                                              big_mid_rounds_num_updates_decay=10, 
                                              mq_rounds=5, 
                                              big_final_round_num_updates=50,
                                              big_buffer_size=256,
                                              big_final_round_num_updates_decay=1),
        
        'gravitar_mq': get_gravitar_job(mix_num_agents=4,
                                              num_envs=32,
                                              buffer_per_agent=32,
                                              img_size=128,
                                              mix_num_updates=1000,
                                              mix_num_updates_decay=1000,
                                              big_mid_rounds_num_updates=1000, 
                                              big_mid_rounds_num_updates_decay=100, 
                                              mq_rounds=5, 
                                              big_final_round_num_updates=5000,
                                              big_buffer_size=256,
                                              big_final_round_num_updates_decay=1),
        
        'gravitar_mq_more_eps': get_gravitar_job(mix_num_agents=4,
                                              num_envs=32,
                                              buffer_per_agent=32,
                                              img_size=128,
                                              mix_num_updates=1000,
                                              mix_num_updates_decay=100,
                                              big_mid_rounds_num_updates=1000, 
                                              big_mid_rounds_num_updates_decay=100, 
                                              mq_rounds=5, 
                                              big_final_round_num_updates=5000,
                                              big_buffer_size=256,
                                              big_final_round_num_updates_decay=1),
        
        'gravitar_mq_longer_mixtrain': get_gravitar_job(mix_num_agents=4,
                                              num_envs=32,
                                              buffer_per_agent=32,
                                              img_size=128,
                                              mix_num_updates=8000,
                                              mix_num_updates_decay=800,
                                              big_mid_rounds_num_updates=2000, 
                                              big_mid_rounds_num_updates_decay=200, 
                                              mq_rounds=1, 
                                              big_final_round_num_updates=5000,
                                              big_buffer_size=256,
                                              big_final_round_num_updates_decay=1),
        
        'gravitar_baseline': get_gravitar_job(mix_num_agents=4,
                                              num_envs=32,
                                              buffer_per_agent=32,
                                              img_size=128,
                                              mix_num_updates=1,
                                              mix_num_updates_decay=1,
                                              big_mid_rounds_num_updates=1, 
                                              big_mid_rounds_num_updates_decay=1, 
                                              mq_rounds=1, 
                                              big_final_round_num_updates=15000,
                                              big_buffer_size=256, 
                                              big_final_round_num_updates_decay=1500),
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
