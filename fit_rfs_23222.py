# 23rd of Feb 2022
#  source activate blurry_vision
# https://colab.research.google.com/drive/1FG7vegjOjeHpkStBvpJEcSeumfwLXD1w#scrollTo=jmcIN_cdTBa_
# ammended from Copy of gitGabs2_Aine_Edited.ipynb
# Code from Alex Wade - Nov 21


import numpy as np
import matplotlib.pyplot as plt

from gabor_gen import GaborGenerator
from dog_gen import DOGGenerator

from trainer import trainer_fn
from trainer2 import trainer_fn2

from utils import dog_fn,gabor_fn

import matplotlib.image as img  
import cv2

import torch
from torch import nn
from torch.nn import functional as F

# /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/fit_receptive_field

# /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/supRN50_conv1_21_g4_60e_e60_np_arr/supRN50_conv1_21_g4_60e_e60_filter_np_arr_61.npy

# supRN50_conv1_21_g0_60e_e60_np_arr/supRN50_conv1_21_g0_60e_e60_filter_np_arr_13.npy

out_dir = 'test_out'
# {out_dir}
# im_name = 'supRN50_conv1_21_g4_60e_e60_filter_np_arr_61'
im_name = 'supRN50_conv1_21_g0_60e_e60_filter_np_arr_13'

# im=np.load(f'/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/supRN50_conv1_21_g4_60e_e60_np_arr/{im_name}.npy')

im=np.load(f'/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/supRN50_conv1_21_g0_60e_e60_np_arr/{im_name}.npy')


im=im[1,:,:]
im=im-np.mean(im)
im=im/(np.std(im))

class Neuron(nn.Module):
    def __init__(self, rf):
        super().__init__()
        h, w = rf.shape
        self.rf = torch.tensor(rf.reshape(1, 1, h, w).astype(np.float32))
        
    def forward(self, x):
        return F.elu((x * self.rf).sum()) + 1

# COLAB CODE NOT NEEDED IN VS - add fig.savefig('figname.png')
fig, (ax) = plt.subplots(1, 1, figsize=(6, 3), dpi=100)
ax.imshow(im)
ax.set(xticks=[], yticks=[]);
fig.savefig(f'{out_dir}/t2_pre_np_arr_{im_name}_2?.png')

plt.imsave(f'{out_dir}/t2_pre_np_arr_{im_name}.png', im) 

neuron = Neuron(im)

# 
_, _, h, w = neuron.rf.shape
torch.manual_seed(20)

gabor_gen = GaborGenerator(image_size=(h, w))
dog_gen = DOGGenerator(image_size=(h, w))

# Create a DOG and Gabor generator

learned_rf_dog = dog_gen().squeeze().cpu().data.numpy()
learned_rf_gabor = gabor_gen().squeeze().cpu().data.numpy()

true_rf = neuron.rf.squeeze().cpu().data.numpy()

# Gabor vs. true RF before training
# visualise comparison - no utility in this - keep ony for testing phase
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 3), dpi=100)
ax1.imshow(learned_rf_dog);
ax1.set(xticks=[], yticks=[], title="Initial DOG")
ax2.imshow(learned_rf_gabor);
ax2.set(xticks=[], yticks=[], title="Initial gabor")

ax3.imshow(true_rf);
ax3.set(xticks=[], yticks=[], title="True RF");

# https://www.southampton.ac.uk/~feeg1001/notebooks/Matplotlib.html
# save this figure.....
fig.savefig(f'{out_dir}/t2_pre_rf_type_comparison_{im_name}.png')
# imsave(test.png)

# /pre_np_arr_{im_name}.png'

# Train the gabor generator to maximizes the model output
from torch import optim
# def trainer_fn2(dog_gen, model_neuron,
# try using trained_fn2 instead
dog_gen, evolved_rfs_dog, dog_loss = trainer_fn2(dog_gen, neuron,save_rf_every_n_epoch=100,optimizer=optim.Adam,lr=0.001)

gabor_gen, evolved_rfs_gabor, gab_loss = trainer_fn(gabor_gen, neuron,save_rf_every_n_epoch=100,optimizer=optim.Adam,lr=0.001)

# why is trainer_fn and not trainer_fn2 being used?????

# visualisation of training process.....
# n_rows = 4
# n_cols = (len(evolved_rfs_dog) + n_rows - 1) // n_rows
# # How did it get to the DOG
# fig, axes = plt.subplots(n_rows, n_cols, dpi=100, figsize=(20, 12))

# for ind, ax in enumerate(axes.flat):
#     if ind < len(evolved_rfs_dog):
#         ax.imshow(evolved_rfs_dog[ind])
#         ax.set(xticks=[], yticks=[])
#     else:
#         ax.axis('off')
# fig, axes = plt.subplots(n_rows, n_cols, dpi=100, figsize=(20, 12))
# # How did it get ot the Gabor?
# for ind, ax in enumerate(axes.flat):
#     if ind < len(evolved_rfs_gabor):
#         ax.imshow(evolved_rfs_gabor[ind])
#         ax.set(xticks=[], yticks=[])
#     else:
#         ax.axis('off')

# fig.savefig('TRAINING_PROCESS.png')


# # Gabor vs. true RF after training
# learned_rf_dog = dog_gen().squeeze().cpu().data.numpy()
# learned_rf_dog=learned_rf_dog-np.mean(learned_rf_dog)
# learned_rf_dog=learned_rf_dog/np.std(learned_rf_dog)

# learned_rf_gabor = gabor_gen().squeeze().cpu().data.numpy()
# learned_rf_gabor=learned_rf_gabor-np.mean(learned_rf_gabor)
# learned_rf_gabor=learned_rf_gabor/np.std(learned_rf_gabor)

# Gabor vs. true RF after training
learned_rf_dog = dog_gen().squeeze().cpu().data.numpy()
learned_rf_dog_pro=learned_rf_dog-np.mean(learned_rf_dog)
learned_rf_dog_pro=learned_rf_dog_pro/np.std(learned_rf_dog_pro)

learned_rf_gabor = gabor_gen().squeeze().cpu().data.numpy()
learned_rf_gabor_pro=learned_rf_gabor-np.mean(learned_rf_gabor)
learned_rf_gabor_pro=learned_rf_gabor_pro/np.std(learned_rf_gabor_pro)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
ax1.imshow(learned_rf_gabor);
# ax1.set(xticks=[], yticks=[], title="Learned DOG")
ax1.set(xticks=[], yticks=[], title="unprocessed gabor")
ax2.imshow(learned_rf_gabor_pro);
# ax2.set(xticks=[], yticks=[], title="Learned Gabor")
ax2.set(xticks=[], yticks=[], title="processed gabor")

fig.savefig(f'{out_dir}/processed_vs_un_gabor_{im_name}.png')



true_rf = neuron.rf.squeeze().cpu().data.numpy()
fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(6, 3), dpi=100)

ax1.imshow(learned_rf_dog);
# ax1.set(xticks=[], yticks=[], title="Learned DOG")
ax1.set(xticks=[], yticks=[], title=f"DOG model loss:{dog_loss:.2f}")
ax2.imshow(learned_rf_gabor);
# ax2.set(xticks=[], yticks=[], title="Learned Gabor")
ax2.set(xticks=[], yticks=[], title=f"Gabor model loss: {gab_loss:.2f}")

ax3.imshow(true_rf);
ax3.set(xticks=[], yticks=[], title="True RF");

fig.savefig(f'{out_dir}/t2_POST_RF_{im_name}.png')

# /pre_np_arr_{im_name}.png'

# supRN50_conv1_21_g4_60e_e60_filter_np_arr_61

# WHY DOES THIS NOT GO THE WHOLE WAYto 100 %?????????
# Loss: -4.54:  25%|█████████▎                           | 5048/20000 [00:09<00:28, 533.08it/s]
# Loss: -4.23:  37%|█████████████▊                       | 7436/20000 [00:15<00:25, 490.30it/s]

# numbers replicate
# Loss: -4.54:  25%|█████████▎                           | 5048/20000 [00:10<00:30, 488.27it/s]
# Loss: -4.23:  37%|█████████████▊                       | 7436/20000 [00:14<00:24, 504.69it/s]

# is it quitting wen it hits a given loss?????


# supRN50_conv1_21_g0_60e_e60_filter_np_arr_13

# Loss: -3.81:  11%|███████████▎                                                                                               | 2110/20000 [00:04<00:37, 474.97it/s]
# Loss: -5.03:  26%|███████████████████████████▌                                                                               | 5143/20000 [00:12<00:35, 413.41it/s]

# (blurry_vision) ainedineen@cusacklab-lamb00:~/blurry_vision/pytorch_untrained_models/imagenet/fit_receptive_field$ python fit_rfs_23222.py
# Loss: -3.81:  11%|███████                                                            | 2110/20000 [00:04<00:39, 457.67it/s]
# Loss: -5.03:  26%|█████████████████▏                                                 | 5143/20000 [00:11<00:32, 463.51it/s]
# (


    # QUESTION: What dictates after how many epochs to give up - is it a local minimum?????
    # do we need two seperate trainers or will 1 suffice with generator instead of eg. dog_generator 
    # currently no other differences to the script
    # https://github.com/dineenai/fit_receptive_field/blob/main/dog_gen.py


    # print some parameters.... eg. signa
    # study gaussian again!





    #   if lr_change_counter > 3:
    #         break


    #     return loss

    # gabor_generator.eval();
    # return gabor_generator, saved_rfs - called evolved_rfs




# loss stayed te same for 300 epochs
    # epoch: 1900, lr = 0.001
    # Loss: -3.81:  10%|██████▋                                                            | 2000/20000 [00:03<00:31, 578.17it/s]epoch: 2000, lr = 0.001
    # Loss: -3.81:  10%|██████▉                                                            | 2059/20000 [00:03<00:30, 579.91it/s]epoch: 2100, lr = 0.001
    # Loss: -3.81:  11%|███████   


#     epoch: 4900, lr = 0.001
# epoch: 4800, lr = 0.001
# Loss: -5.02:  24%|████████████████▍                                                  | 4899/20000 [00:09<00:27, 541.32it/s]e
# Loss: -5.03:  25%|████████████████▌                                                  | 4954/20000 [00:09<00:27, 542.14it/s]epoch: 5000, lr = 0.001
# Loss: -5.03:  25%|████████████████▉                                                  | 5064/20000 [00:09<00:27, 539.54it/s]epoch: 5100, lr = 0.001
# Loss: -5.03:  26%|█████████████████▏                                                 | 5143/20000 [00:10<00:28, 513.51it/s]
