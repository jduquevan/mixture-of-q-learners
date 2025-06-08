import os
import random

def gen_command(config):
    command = "sbatch sweep/run_ppomix.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'ppo_epochs': [1, 2, 3],
        'lr': [0.0001, 0.0003, 0.0007, 0.001, 0.003],
        'entropy_coef': [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
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