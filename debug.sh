source /home/mila/a/aghajohm/repos/mixture-of-q-learners/load_env.sh

python purejaxql/di_atari.py \
    RUN_ID=test \
    DEBUG=True \
    WANDB_MODE=disabled \
    WANDB_TAGS=[gravitar_baseline] \
    SEED=42 \
    cluster@_global_=milad_mila \
    alg.ENV_NAME=Gravitar-v5 \
    alg.NUM_AGENTS=1 \
    alg.NUM_ENVS=128 \
    alg.BUFFER_PER_AGENT=128 \
    alg.SHARE_STRATEGY=no-share \
    alg.TOTAL_TIMESTEPS_DECAY=50000000.0 \
    alg.TOTAL_TIMESTEPS=50000000.0 \
    alg.RESET_SCHEDULE=no-reset \
    alg.MIX_SCHEDULE=no-mix \
    +alg.ENV_KWARGS.img_width=84 \
    +alg.ENV_KWARGS.img_height=84