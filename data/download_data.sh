#!/bin/bash

gdown --id 10CVazyokzNA0RJYYj4leNJ-dkFExSvFw
gdown --id 1VCM66cwG2ROKR_wP8hEpud_T0GX0Qym2 

unzip wav2folder.zip -d wav2folder
unzip bvh2folder.zip -d bvh2folder

python ./clip_wav.py ./wav2folder --dest ./wav2clips
python ./clip_bvh.py ./bvh2folder --dest ./bvh2clips
