#source /home/mila/a/aghajohm/repos/mixture-of-q-learners/load_env.sh

python purejaxql/mq_atari.py \
    RUN_ID=test_1103 \
    DEBUG=False \
    WANDB_MODE=online \
    WANDB_TAGS=[gravitar_baseline] \
    SEED=42 \
    cluster@_global_=milad_mila \
    alg.ENV_NAME=Breakout-v5 \
    alg.mix.NUM_AGENTS=4 \
    alg.NUM_ENVS=32 \
    alg.mix.BUFFER_PER_AGENT=32 \
    +alg.ENV_KWARGS.img_width=84 \
    +alg.ENV_KWARGS.img_height=84 \
    alg.mix.NUM_UPDATES=10 \
    alg.mix.NUM_UPDATES_DECAY=5 \
    alg.big.mid_rounds.NUM_UPDATES=10 \
    alg.big.mid_rounds.NUM_UPDATES_DECAY=5 \
    alg.mq.rounds=3 \
    alg.big.final_round.NUM_UPDATES=50 \
    alg.big.final_round.NUM_UPDATES_DECAY=10 \
