#source /home/mila/a/aghajohm/repos/mixture-of-q-learners/load_env.sh

python purejaxql/mq_atari.py \
    RUN_ID=test_1003 \
    DEBUG=False \
    WANDB_MODE=online \
    WANDB_TAGS=[gravitar_baseline] \
    SEED=42 \
    cluster@_global_=milad_mila \
    alg.ENV_NAME=Breakout-v5 \
    alg.mix.NUM_AGENTS=1 \
    alg.NUM_ENVS=32 \
    alg.mix.BUFFER_PER_AGENT=128 \
    +alg.ENV_KWARGS.img_width=84 \
    +alg.ENV_KWARGS.img_height=84
