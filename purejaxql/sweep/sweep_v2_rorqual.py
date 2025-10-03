import os
import random

def gen_command(config):
    command = "sbatch sweep/run_rorqual.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'entropy_coef': [0.005],
        'diversity_coef': [0.001, 0.003, 0.005],
        'mixing_steps': [50, 100, 200],
        'tau': [0.001, 0.003, 0.005],
        'lr': [0.00015],
        'accum_steps': [3],
        'cross_coeff': [0.25, 0.5],
        'div_adv_beta': [-0.15, 0.15, 0.2, 0.25, 0.3, 0.35],
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