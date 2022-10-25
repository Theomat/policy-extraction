#!/usr/bin/bash
FILE_NAME="pong_dqn.pt"
python generate_policy.py $FILE_NAME --env "ALE/Pong-ram-v5"
python -m polext extract.py $FILE_NAME
