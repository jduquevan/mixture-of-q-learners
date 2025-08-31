#!/bin/bash

source load_env.sh
python purejaxql/pqn_atari_ngu.py \
    RUN_ID=test_ngu_1 \
    DEBUG=0 \
    WANDB_TAGS=[skiing_pqn_atari_400x400_512envs_higheps] \
    SEED=42 \
    cluster@_global_=milad_mila \
    alg=pqn_atari_ngu \
    alg.ENV_NAME=Skiing-v5 \
    alg.NUM_ENVS=7 \
    +alg.ENV_KWARGS.img_width=160 \
    +alg.ENV_KWARGS.img_height=160 \
    alg.TOTAL_TIMESTEPS=250000000.0 \
    alg.EPS_FINISH=0.01 \
    alg.IS_SARSA=False