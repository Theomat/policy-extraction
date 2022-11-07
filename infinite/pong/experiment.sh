#!/usr/bin/bash
FILE_NAME="pong_dqn.pt"
ENV="ALE/Pong-ram-v5"
python scripts/generate_policy.py $FILE_NAME $ENV -t 5000000
python scripts/record_video.py $FILE_NAME $ENV
python -m polext extract.py $FILE_NAME
