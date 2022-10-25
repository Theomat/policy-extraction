#!usr/bin/bash
FILE_NAME="my_dqn.pt"
python generate_policy.py $FILE_NAME --env "LunarLander-v2"
python -m polext extract.py $FILE_NAME
