#!/usr/bin/bash
FILE_NAME="lunar_lander_dqn.pt"
ENV="LunarLander-v2"
python scripts/generate_policy.py $FILE_NAME $ENV -t 5e5
python scripts/record_video.py $FILE_NAME $ENV
python -m polext extract.py $FILE_NAME