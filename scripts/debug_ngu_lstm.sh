#!/bin/bash

source load_env.sh
python purejaxql/pqn_atari_lstm_ngu.py \
    RUN_ID=test_lstm_ngu_11 \
    DEBUG=0\
    WANDB_TAGS=[test_ngu_pitfall] \
    SEED=42 \
    cluster@_global_=milad_mila \
    alg=pqn_atari_ngu \
    alg.ENV_NAME=Pitfall-v5 \
    alg.NUM_ENVS=512 \
    +alg.ENV_KWARGS.img_width=160 \
    +alg.ENV_KWARGS.img_height=160 \
    alg.TOTAL_TIMESTEPS=250000000.0 \
    alg.EPS_FINISH=0.01 \
    alg.IS_SARSA=False \
    +alg.HIDDEN_SIZE=256