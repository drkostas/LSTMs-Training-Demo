#!/bin/bash
cd "/Users/gkos/Insync/delfinas7kostas@gmail.com/Google Drive/Projects/UTK/COSC525-Project4"
conda activate cosc525_project4
which python

python train.py --attr age --task 1
python train.py --attr race --task 1
python train.py --attr age --task 2
python train.py --attr race --task 2
python train.py --attr age --task 3
python train.py --attr race --task 3