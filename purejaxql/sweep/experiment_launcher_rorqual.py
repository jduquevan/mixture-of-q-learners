import os
import random

SEEDS = [42, 43, 44]
GAMES = ["BattleZone", "DoubleDunk", "NameThisGame", "Phoenix", "Qbert"]
RESOLUTIONS = [84, 160, 400]
BATCH_SIZES = [128, 512]
SBATCH_SCRIPT = "sweep/run_pqn_rorqual.slurm"
PY_ENTRY = "/project/rrg-bengioy-ad/jduque/mixture-of-q-learners/purejaxql/pqn_atari.py"

def gen_command(cfg: dict) -> str:
    parts = [
        "sbatch",
        SBATCH_SCRIPT,
        "--",
        "python",
        PY_ENTRY,
    ]
    parts += [f"{k}={v}" for k, v in cfg.items()]
    return " ".join(str(p) for p in parts)


def main(fake_submit: bool = True):
    for seed in SEEDS:
        for game in  ["Qbert", "NameThisGame"]:
            for res in RESOLUTIONS:
                for batch_size in BATCH_SIZES:
                    cfg = {
                        "SEED": seed,
                        "alg.ENV_NAME": f"{game}-v5",
                        "alg.NUM_ENVS": batch_size,
                        "alg.ENV_KWARGS.img_width": res,
                        "alg.ENV_KWARGS.img_height": res,
                        "alg.TOTAL_TIMESTEPS": 5e9,
                    }
                    command = gen_command(cfg)
                    if fake_submit:
                        print(command)
                        print('fake submit')
                    else:
                        os.system(command)

if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()