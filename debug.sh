#source /home/mila/a/aghajohm/repos/mixture-of-q-learners/load_env.sh

python purejaxql/mq_atari.py \
    RUN_ID=test_1002 \
    DEBUG=False \
    WANDB_MODE=offline \
    WANDB_TAGS=[gravitar_baseline] \
    SEED=42 \
    cluster@_global_=milad_mila \
    alg.ENV_NAME=Gravitar-v5 \
    alg.NUM_AGENTS=1 \
    alg.NUM_ENVS=32 \
    alg.BUFFER_PER_AGENT=128 \
    alg.TOTAL_TIMESTEPS_DECAY=50000000.0 \
    alg.TOTAL_TIMESTEPS=50000.0 \
    +alg.ENV_KWARGS.img_width=84 \
    +alg.ENV_KWARGS.img_height=84
