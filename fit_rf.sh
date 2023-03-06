#!/bin/bash


PYTHON="/opt/anaconda3/envs/blurry_vision/bin/python"
# NET='supRN50_conv1_21_g4_30e_g0_30e_e60' #1502
# NET='supRN50_conv1_21_g0_60e_e35' #1503
# NET='supRN50_conv1_21_g0_60e_e60' #1504
# NET='supRN50_conv1_21_g0_30e_g4_30e_e35' #1505
NET='supRN50_conv1_21_g0_30e_g4_30e_e60' #1506

${PYTHON} fit_rfs_23222_loop.py --chosen_net ${NET} 
# /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/fit_receptive_field/fit_rfs_23222_loop.py


#
# TO DO



# supRN50_conv1_21_g0_30e_g4_30e_e60



# DONE (and IN PROGRESS)
# supRN50_conv1_21_g4_60e_e60
# supRN50_conv1_21_g4_60e_e35
# supRN50_conv1_21_g4_30e_g0_30e_e35

# supRN50_conv1_21_g4_30e_g0_30e_e60
# supRN50_conv1_21_g0_60e_e35
# supRN50_conv1_21_g0_60e_e60
# supRN50_conv1_21_g0_30e_g4_30e_e35