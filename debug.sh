#source /home/mila/a/aghajohm/repos/mixture-of-q-learners/load_env.sh

python purejaxql/mq_atari.py \
    RUN_ID=test_1004 \
    DEBUG=False \
    WANDB_MODE=disabled \
    WANDB_TAGS=[gravitar_baseline] \
    SEED=42 \
    cluster@_global_=milad_mila \
    alg.ENV_NAME=Breakout-v5 \
    alg.mix.NUM_AGENTS=8 \
    alg.NUM_ENVS=3 \
    alg.mix.BUFFER_PER_AGENT=9 \
    +alg.ENV_KWARGS.img_width=84 \
    +alg.ENV_KWARGS.img_height=84
