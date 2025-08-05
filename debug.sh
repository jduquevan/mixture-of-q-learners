#source /home/mila/a/aghajohm/repos/mixture-of-q-learners/load_env.sh

python purejaxql/ep_atari.py \
    RUN_ID=test_1278 \
    DEBUG=False \
    WANDB_MODE=online \
    WANDB_TAGS=[ep_atari] \
    SEED=42 \
    cluster@_global_=milad_mila \
    alg.ENV_NAME=Breakout-v5 \
    alg.NUM_ENVS=1024 \
    +alg.ENV_KWARGS.img_width=84 \
    +alg.ENV_KWARGS.img_height=84 \
    alg.NUM_UPDATES=100 \
    alg.NUM_UPDATES_DECAY=5 \
    alg.ROUNDS=3 \
    alg.DEPTH_EXPLORATION_EPS_SCHEDULER=depth_exploration_0.1_U0.3 \
    alg.ENV_WISE_EXPLORATION_EPS_SCHEDULER=envwise50D50R_U0.3 \
    alg.BUFFER_SIZE=1024 \
    
