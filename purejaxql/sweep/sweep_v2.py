import os
import random

def gen_command(config):
    command = "sbatch sweep/run_ppomix_se.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'entropy_coef': [0.003, 0.005, 0.007, 0.01],
        'diversity_coef': [0.003, 0.005, 0.007, 0.01],
        'mixing_steps': [100, 200, 300, 500],
    }

    # sample a random config
    config = {}
    for key, values in hparams.items():
        config[key] = random.choice(values)

    # submit this job using slurm
    command = gen_command(config)
    if fake_submit:
        print('fake submit')
    else:
        os.system(command)
    print(command)

def main(num_jobs: int, fake_submit: bool = True):
    for i in range(num_jobs):
        run_random_job(fake_submit=fake_submit)

if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()