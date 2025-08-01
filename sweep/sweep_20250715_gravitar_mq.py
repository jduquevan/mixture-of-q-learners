from sweep_utils import submit_slurm_job, get_compute_command
### Jobs ###
BASE_RUN_ID = "20250715_v1_twoday"
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
            " alg.big.mid_rounds.fill.NUM_ENVS={big_mid_rounds_fill_num_envs}" \
            " alg.big.mid_rounds.fill.NUM_TEST_ENVS={big_mid_rounds_fill_num_test_envs}" \
            " alg.big.mid_rounds.NUM_ENVS={big_mid_rounds_num_envs}" \
            " alg.big.mid_rounds.NUM_TEST_ENVS={big_mid_rounds_num_test_envs}" \
            " alg.big.final_round.NUM_ENVS={big_final_round_num_envs}" \
            " alg.big.final_round.NUM_TEST_ENVS={big_final_round_num_test_envs}" \
            " alg.big.mid_rounds.DONT_FILL_BUFFER={big_mid_rounds_dont_fill_buffer}" \
                
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
                     big_final_round_num_updates_decay: int,
                     big_mid_rounds_dont_fill_buffer: bool = False):
    
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
        'big_mid_rounds_fill_num_envs': num_envs // 2,
        'big_mid_rounds_num_envs': num_envs // 2 * mix_num_agents,
        'big_final_round_num_envs': num_envs * mix_num_agents,
        'big_mid_rounds_fill_num_test_envs': 2,
        'big_mid_rounds_num_test_envs': 8,
        'big_final_round_num_test_envs': 16,
        'big_mid_rounds_dont_fill_buffer': big_mid_rounds_dont_fill_buffer,
    }
    return python_command, python_kwargs

def get_jobs():
    jobs = {
        'gravitar_mq_longer_mixtrain': get_gravitar_job(mix_num_agents=2,
                                              num_envs=64,
                                              buffer_per_agent=64,
                                              img_size=128,
                                              mix_num_updates=60000,
                                              mix_num_updates_decay=6000,
                                              big_mid_rounds_num_updates=20000, 
                                              big_mid_rounds_num_updates_decay=2000, 
                                              mq_rounds=1, 
                                              big_final_round_num_updates=120000,
                                              big_buffer_size=256,
                                              big_final_round_num_updates_decay=1),
        
        
        'gravitar_mq_longer_mixtrain_evgeni': get_gravitar_job(mix_num_agents=1,
                                              num_envs=128,
                                              buffer_per_agent=128,
                                              img_size=128,
                                              mix_num_updates=60000,
                                              mix_num_updates_decay=6000,
                                              big_mid_rounds_num_updates=20000, 
                                              big_mid_rounds_num_updates_decay=2000, 
                                              mq_rounds=1, 
                                              big_final_round_num_updates=12e0000,
                                              big_buffer_size=256,
                                              big_final_round_num_updates_decay=1),
        
        'gravitar_baseline': get_gravitar_job(mix_num_agents=2,
                                              num_envs=64,
                                              buffer_per_agent=64,
                                              img_size=128,
                                              mix_num_updates=1,
                                              mix_num_updates_decay=1,
                                              big_mid_rounds_num_updates=1, 
                                              big_mid_rounds_num_updates_decay=1, 
                                              mq_rounds=1, 
                                              big_final_round_num_updates=200000,
                                              big_buffer_size=256, 
                                              big_final_round_num_updates_decay=10000),
        
        'gravitar_debug_buffer': get_gravitar_job(mix_num_agents=2,
                                              num_envs=64,
                                              buffer_per_agent=64,
                                              img_size=128,
                                              mix_num_updates=1,
                                              mix_num_updates_decay=1,
                                              big_mid_rounds_num_updates=20000, 
                                              big_mid_rounds_num_updates_decay=1000, 
                                              mq_rounds=1, 
                                              big_final_round_num_updates=1,
                                              big_buffer_size=256, 
                                              big_final_round_num_updates_decay=1,
                                              big_mid_rounds_dont_fill_buffer=True),
        
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
