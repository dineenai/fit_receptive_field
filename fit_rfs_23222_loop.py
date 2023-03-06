# 23rd of Feb 2022
#  source activate blurry_vision
# https://colab.research.google.com/drive/1FG7vegjOjeHpkStBvpJEcSeumfwLXD1w#scrollTo=jmcIN_cdTBa_
# ammended from Copy of gitGabs2_Aine_Edited.ipynb
# Code from Alex Wade - Nov 21

import argparse
import os
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

parser = argparse.ArgumentParser(description='Fit Receptive Fields')
parser.add_argument('--chosen_net', default='', type=str, metavar='NET',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()
chosen_net = args.chosen_net
# chosen_net = 'supRN50_conv1_21_g4_60e_e60'


# /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/fit_receptive_field

# /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/supRN50_conv1_21_g4_60e_e60_np_arr/supRN50_conv1_21_g4_60e_e60_filter_np_arr_61.npy

# supRN50_conv1_21_g0_60e_e60_np_arr/supRN50_conv1_21_g0_60e_e60_filter_np_arr_13.npy


out_dir = f'/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/fit_receptive_field/fit_comp-Mar23/{chosen_net}'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
# {out_dir}
# im_name = 'supRN50_conv1_21_g4_60e_e60_filter_np_arr_61'


# chosen_net = 'supRN50_conv1_21_g0_60e_e60'

# /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/fit_receptive_field

import os

# folder path
# dir_path = "/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/supRN50_conv1_21_g0_60e_e60_np_arr"
dir_path = f"/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/{chosen_net}_np_arr"
size = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        size += 1
print('File count:', size)
print(f"Number of files is {size}")

# size = 12 #get n of files in directory - should actually be fixed number!!!!!


# file_name = [] #<class 'list'>

file_name = [0] * size
dog_sigma1 = [0] * size
dog_sigma2 = [0] * size
dog_amp1 = [0] * size
dog_amp2 = [0] * size
dog_center = [0] * size
dog_image_size = [0] * size
dog_target_std = [0] * size
gabor_sigma = [0] * size

gabor_sigma = [0] * size
gabor_theta = [0] * size
gabor_Lambda = [0] * size
gabor_psi = [0] * size
gabor_gamma = [0] * size
gabor_center = [0] * size
gabor_image_size = [0] * size
gabor_target_std = [0] * size

dog_model_loss = [0] * size
gabor_model_loss = [0] * size

# best_model = ['0'] * size    # gab / dog
best_model = [0] * size 

for i in range (size):
    # im_name = f'supRN50_conv1_21_g0_60e_e60_filter_np_arr_{i}'
    im_name = f'{chosen_net}_filter_np_arr_{i}'
    print(im_name)

    # im=np.load(f'/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/supRN50_conv1_21_g4_60e_e60_np_arr/{im_name}.npy')

    im=np.load(f'/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/{chosen_net}_np_arr/{im_name}.npy')


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

    # # COLAB CODE NOT NEEDED IN VS - add fig.savefig('figname.png')
    # fig, (ax) = plt.subplots(1, 1, figsize=(6, 3), dpi=100)
    # ax.imshow(im)
    # ax.set(xticks=[], yticks=[]);
    # fig.savefig(f'{out_dir}/t2_pre_np_arr_{im_name}_2?.png')

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

    # # Gabor vs. true RF before training
    # # visualise comparison - no utility in this - keep ony for testing phase
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 3), dpi=100)
    # ax1.imshow(learned_rf_dog);
    # ax1.set(xticks=[], yticks=[], title="Initial DOG")
    # ax2.imshow(learned_rf_gabor);
    # ax2.set(xticks=[], yticks=[], title="Initial gabor")

    # ax3.imshow(true_rf);
    # ax3.set(xticks=[], yticks=[], title="True RF");

    # # https://www.southampton.ac.uk/~feeg1001/notebooks/Matplotlib.html
    # # save this figure.....
    # fig.savefig(f'{out_dir}/t2_pre_rf_type_comparison_{im_name}.png')
    # # imsave(test.png)

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




    # super().__init__()


# 
    # AttributeError: 'numpy.ndarray' object has no attribute 'sigma'
    # print(f'attributes of processed gabor: sigma: {learned_rf_gabor_pro.sigma}, gamma: {learned_rf_gabor_pro.gamma}, psi: {learned_rf_gabor_pro.psi}, {learned_rf_gabor_pro.gamma}')
    # print(f'attributes of unprocessed gabor: sigma: {learned_rf_gabor.sigma}, gamma: {learned_rf_gabor.gamma}, psi: {learned_rf_gabor.psi}, {learned_rf_gabor.gamma}')
    print(f'attributes of unprocessed gabor: sigma: {gabor_gen.sigma[0]:.2f}, gamma: {gabor_gen.gamma[0]:.2f}, psi: {gabor_gen.psi[0]:.2f}, {gabor_gen.gamma[0]:.2f}')

    # print(f'attributes of unprocessed gabor: sigma: {gabor_gen().sigma}, gamma: {gabor_gen().gamma}, psi: {gabor_gen().psi}, {gabor_gen().gamma}') #AttributeError: 'Tensor' object has no attribute 'sigma'


# outpur:
# attributes of unprocessed gabor: sigma: Parameter containing:
# tensor([3.], requires_grad=True), gamma: Parameter containing:
# tensor([1.8814], requires_grad=True), psi: Parameter containing:
# tensor([1.5232], requires_grad=True), Parameter containing:
# tensor([1.8814], requires_grad=True)

# attributes of unprocessed dog: sigma1: Parameter containing:
# tensor([0.7251], requires_grad=True), sigma2: Parameter containing:
# tensor([18.0320], requires_grad=True)


    # print(f'attributes of processed dog: sigma1: {learned_rf_dog_pro.sigma1}, sigma2: {learned_rf_dog_pro.sigma2}')
    # print(f'attributes of unprocessed dog: sigma1: {learned_rf_dog.sigma1}, sigma2: {learned_rf_dog.sigma2}')
    print(f'attributes of unprocessed dog: sigma1: {dog_gen.sigma1[0]:.2f}, sigma2: {dog_gen.sigma2[0]:.2f}')

    # learned_rf_gabor_pro.sigma
    # learned_rf_gabor_pro.sigma_x

    # # Comment out graphs Mar 23
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
    # ax1.imshow(learned_rf_gabor);
    # # ax1.set(xticks=[], yticks=[], title="Learned DOG")
    # ax1.set(xticks=[], yticks=[], title="unprocessed gabor")
    # ax2.imshow(learned_rf_gabor_pro);
    # # ax2.set(xticks=[], yticks=[], title="Learned Gabor")
    # ax2.set(xticks=[], yticks=[], title="processed gabor")

    # fig.savefig(f'{out_dir}/pro_vs_unpro_gabor_{im_name}.png')


    # # Comment out graphs Mar 23
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
    # ax1.imshow(learned_rf_dog);
    # ax1.set(xticks=[], yticks=[], title="unprocessed dog")
    # ax2.imshow(learned_rf_dog_pro);
    # ax2.set(xticks=[], yticks=[], title="processed dog")

    # fig.savefig(f'{out_dir}/pro_vs_unpro_dog_{im_name}.png')



    true_rf = neuron.rf.squeeze().cpu().data.numpy()
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(6, 3), dpi=100)

    ax1.imshow(learned_rf_dog);
    # ax1.set(xticks=[], yticks=[], title="Learned DOG")
    ax1.set(xticks=[], yticks=[], title=f"DOG loss:{dog_loss:.2f}")
    ax2.imshow(learned_rf_gabor);
    # ax2.set(xticks=[], yticks=[], title="Learned Gabor")
    ax2.set(xticks=[], yticks=[], title=f"Gabor loss: {gab_loss:.2f}")

    ax3.imshow(true_rf);
    ax3.set(xticks=[], yticks=[], title="True RF");

    fig.savefig(f'{out_dir}/RF_model_comparison_{im_name}.png')

    # fig.savefig(f'{out_dir}/RF_model_comparison_grey_{im_name}.png', cmap='gray')
#     fit_rfs_23222_loop.py:249: MatplotlibDeprecationWarning: savefig() got unexpected keyword argument "cmap" which is no longer supported as of 3.3 and will become an error two minor releases later
#   fig.savefig(f'{out_dir}/RF_model_comparison_grey_{im_name}.png', cmap='gray')

    import pandas as pd

 



    # dog_loss = []
    # gabor_loss = []

    # best_model = []     # gab / dog


    file_name[i] = str(im_name) #<class 'list'> #IndexError: list assignment index out of range
   

#    tensor to float -  x_inp.item().

    print(f"type of dog_gen.sigma1[0] {dog_gen.sigma1[0]} vs after .item() {dog_gen.sigma1[0].item()}")

    dog_sigma1[i] = dog_gen.sigma1[0].item()
    dog_sigma2[i] = dog_gen.sigma2[0].item()
    dog_amp1[i] = dog_gen.amp1[0].item()
    dog_amp2[i] = dog_gen.amp2[0].item()
    dog_center[i] = dog_gen.center[0].item()
    # dog_image_size[i] = dog_gen.image_size[0] #TypeError: 'float' object is not subscriptable
    # dog_image_size[i] = dog_gen.image_size[0].item() #dog_image_size[i] = dog_gen.image_size[0].item()
    dog_image_size[i] = dog_gen.image_size
    dog_target_std[i] = dog_gen.target_std

    gabor_sigma[i] = gabor_gen.sigma[0].item()
    gabor_theta[i] = gabor_gen.theta[0].item()
    gabor_Lambda[i] = gabor_gen.Lambda[0].item()
    gabor_psi[i] = gabor_gen.psi[0].item()
    gabor_gamma[i] = gabor_gen.gamma[0].item()
    gabor_center[i] = gabor_gen.center[0].item()
    gabor_image_size[i] = gabor_gen.image_size
    gabor_target_std[i] = gabor_gen.target_std

    # dog_sigma1[i] = str(round(dog_gen.sigma1[0], 3))
    # dog_sigma2[i] = str(round(dog_gen.sigma2[0], 3))
    # dog_amp1[i] = str(round(dog_gen.amp1[0], 3))
    # dog_amp2[i] = str(round(dog_gen.amp2[0], 3))
    # dog_center[i] = str(round(dog_gen.center[0], 3))
    # dog_image_size[i] = str(round(dog_gen.image_size[0], 3))
    # dog_target_std[i] = str(round(dog_gen.target_std[0], 3))

    # gabor_sigma[i] = str(round(gabor_gen.sigma[0], 3))
    # gabor_theta[i] = str(round(gabor_gen.theta[0], 3))
    # gabor_Lambda[i] = str(round(gabor_gen.Lambda[0], 3))
    # gabor_psi[i] = str(round(gabor_gen.psi[0], 3))
    # gabor_gamma[i] = str(round(gabor_gen.gamma[0], 3))
    # gabor_center[i] = str(round(gabor_gen.center[0], 3))
    # gabor_image_size[i] = str(round(gabor_gen.image_size[0], 3))
    # gabor_target_std[i] = str(round(gabor_gen.target_std[0], 3))

# TypeError: len() of a 0-d tensor
    # print(f"type dof_loss{type(dog_loss)}, len? {len(dog_loss)}")
    # type dof_loss<class 'list'>, len? 0

    dog_model_loss[i] = dog_loss
    gabor_model_loss[i] = gab_loss

    if gab_loss > dog_loss:
        best_model[i] = 'dog' 
    else:
        best_model[i] = 'gab' 

    print(f"for rf {im_name}, the best model is {best_model[i]}, gab_loss {gab_loss:.2f}, dog_odd {dog_loss:.2f}")
    # 
    # /pre_np_arr_{im_name}.png'

    # supRN50_conv1_21_g4_60e_e60_filter_np_arr_61


# attributes of unprocessed gabor: sigma: 3.00, gamma: 2.21, psi: 0.67, 2.21
# attributes of unprocessed dog: sigma1: 0.56, sigma2: 18.48

# supRN50_conv1_21_g0_60e_e60_filter_np_arr_3

# rf_model_params = pd.DataFrame(
# {'DOG': image,
#     'lst2Title': dog_sigma1,
#     'lst3Title': lst3
# })

# rf_model_params = pd.DataFrame(list(zip( file_name, dog_sigma1, dog_sigma2, dog_amp1,dog_amp2, dog_center, dog_image_size,
#                         dog_target_std, gabor_sigma, gabor_sigma, gabor_theta, gabor_Lambda, gabor_psi,
#                         gabor_gamma, gabor_center,
#                         gabor_image_size, gabor_target_std, dog_model_loss, gabor_model_loss, best_model )))
rf_model_params = pd.DataFrame(
    { 'filename': file_name, 'dog_sig1': dog_sigma1, 'dog_sig2': dog_sigma2, 'dog_amp1': dog_amp1, 'dog_amp2': dog_amp2, 'dog_center': dog_center, 'dog_image_size': dog_image_size,
                        'dog_target_std': dog_target_std, 'gabor_sigma': gabor_sigma, 'gabor_sigma': gabor_sigma, 'gabor_theta': gabor_theta, 'gabor_Lambda': gabor_Lambda, 'gabor_psi': gabor_psi,
                        'gabor_gamma': gabor_gamma, 'gabor_center': gabor_center,
                        'gabor_image_size': gabor_image_size, 'gabor_target_std': gabor_target_std, 'dog_model_loss': dog_model_loss, 'gabor_model_loss': gabor_model_loss, 'best_model': best_model})
                        # 'gabor_image_size': gabor_image_size, 'gabor_target_std': gabor_target_std, 'dog_loss': dog_loss, 'gabor_loss': gab_loss, 'best_model': best_model})


# df.to_csv(index=False)
# rf_model_params.to_csv(f'rf_model_params/supRN50_conv1_21_g0_60e_e60_rf_model_params.csv')
rf_model_params.to_csv(f'rf_model_params-Mar23/{chosen_net}_rf_model_params.csv')