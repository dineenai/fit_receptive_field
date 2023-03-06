
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob




def format_loss_columns(df, col=""):
    gabor_model_loss_suffix = len('tensor-')
       
    df[f'{col}'] = df[f'{col}'].str[gabor_model_loss_suffix:]
    df[f'{col}']  = df[f'{col}'].str.split(',').str[0]
    df[f'{col}'] = df[f'{col}'].astype('float')
    
    return df

# PATHS
save_dir = "/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/fit_receptive_field/plot_rf_params-Mar23"
path_to_dfs = "/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/fit_receptive_field/rf_model_params-Mar23"



all_models= []
for file in glob.glob(f"{path_to_dfs}/*.csv"):
    print(f'Filename is {file}')
    len_file_suffix = len('_rf_model_params.csv')
    network_name = file[:-len_file_suffix]
    
    network_name= network_name.split("/")[-1]

    epoch = network_name.split("_e")[-1]

    epoch_suffix = "_e" + epoch
    print(epoch_suffix)
    print(network_name) #still has epocg eg. _e60
    print('network_name')
    
    network_name_lessEpoch = network_name[:-len(epoch_suffix)]
    print(network_name_lessEpoch)

    rf_param_df = pd.read_csv(file)
    rf_param_df.drop(columns=['Unnamed: 0'], inplace=True)

    rf_param_df['network']= network_name_lessEpoch
    rf_param_df['network_epoch'] = network_name
    rf_param_df['epoch']= int(epoch)
    print(rf_param_df.columns)
    
    all_models.append(rf_param_df)

allModels_df = pd.concat(all_models, axis=0, ignore_index=True)
print(allModels_df)



allModels_df = format_loss_columns(allModels_df,'gabor_model_loss')
allModels_df = format_loss_columns(allModels_df,'dog_model_loss')
print(allModels_df.columns)

# Index(['filename', 'dog_sig1', 'dog_sig2', 'dog_amp1', 'dog_amp2',
#        'dog_center', 'dog_image_size', 'dog_target_std', 'gabor_sigma',
#        'gabor_theta', 'gabor_Lambda', 'gabor_psi', 'gabor_gamma',
#        'gabor_center', 'gabor_image_size', 'gabor_target_std',
#        'dog_model_loss', 'gabor_model_loss', 'best_model', 'network', 'epoch'],
#       dtype='object')

# To compare loss for DOG vs GAB - need to melt the df! to have both in their own column
# use filename!

 
# # $\sigma$  is the sigma/standard deviation of the Gaussian envelop
# print(rf_param_df['best_model'])
# print(rf_param_df.columns)
# rf_param_df['Network_epoch'] = 'supRN50_conv1_21_g0_60e_e60'

# rf_param_df_gab = rf_param_df[rf_param_df['best_model']=='gab']
# print(rf_param_df_gab['gabor_sigma'])

# # HUE Gab vs DOG...
# # Plot distribution of RFs as Violin Plots?
# rf_param_df_dog = rf_param_df[rf_param_df['best_model']=='dog']
# print(len(rf_param_df_dog['dog_sig1']))
# # gabor_sigma



# Output loss in readable format:

# print(type(rf_param_df['gabor_model_loss'][0]))

# print(rf_param_df['gabor_model_loss'])
# print(rf_param_df['dog_model_loss'])


# # Plot the gaussians......

# # plot 
# # hue for which was best


# Change aspect ratio!
# hue_order=subset_of_nets,  order=order_of_cat, errorbar=None, palette=sns.color_palette("Paired"), , 

gabor_best_models = allModels_df[allModels_df['best_model']=='gab']

ax = sns.catplot(data=allModels_df, kind="violin",x="gabor_sigma", y="network_epoch" , hue='best_model',aspect=6)
# plt.ylabel("Shape Bias (%)")
# plt.xlabel("Category")
plt.title(f'Plot Gabor Sigma for all RFs (Seperated by ideal model - DOG vs Gabor Patch)')
# fig.set_figwidth(20)
plt.savefig(f'{save_dir}/Gabor_sigma_plot-violin.png', bbox_inches="tight")


print(allModels_df['gabor_sigma'].describe())


print(gabor_best_models['network_epoch'].value_counts())


for chosen_epoch in [35,60]:
# chosen_epoch = 35
    allModels_df_epoch35 = allModels_df[allModels_df['epoch']==chosen_epoch]
    ax = sns.catplot(data=allModels_df_epoch35, kind="violin",x="gabor_sigma", y="network_epoch" ,aspect =2.5)
    # plt.ylabel("Shape Bias (%)")
    # plt.xlabel("Category")
    plt.title(f'Plot Gabor Sigma for all RFs at epoch {chosen_epoch}')
    # fig.set_figwidth(20)
    plt.savefig(f'{save_dir}/gabor_sigma_allRFs_epoch-{chosen_epoch}_plot-violin.png', bbox_inches="tight")


# chosen_epoch = 35

print(allModels_df['network'].unique())

# ['supRN50_conv1_21_g0_30e_g4_30e' 'supRN50_conv1_21_g4_30e_g0_30e'
#  'supRN50_conv1_21_g0_60e' 'supRN50_conv1_21_g4_60e']

rename_chosen_models_exp={
                            'supRN50_conv1_21_g0_30e_g4_30e':'HighRes-Blur',
                            'supRN50_conv1_21_g4_30e_g0_30e':'Blur-HighRes',
                            'supRN50_conv1_21_g0_60e':'HighRes',
                            'supRN50_conv1_21_g4_60e':'Blur',
                          }
subset_of_nets = ['HighRes','Blur-HighRes','Blur']

def rename_and_select_rows_from_df(df, rename_models_dict, subset_of_nets):
    df.replace(regex=rename_chosen_models_exp, inplace=True)  
    df = df.rename(columns={"network": "Network"})
    df = df.loc[df['Network'].isin(subset_of_nets)]
    return df

allModels_df_subset = rename_and_select_rows_from_df(allModels_df, rename_chosen_models_exp, subset_of_nets)

print(allModels_df_subset)


    

ax = sns.catplot(data=allModels_df_subset, kind="violin",y="gabor_sigma", hue="Network", hue_order=subset_of_nets,x='epoch', aspect=1.7, palette=sns.color_palette("Blues_r"))
plt.title(f'Size of Receptive Fields for Networks over Time')
# fig.set_figwidth(20)

# ax.legend(title="Network", loc='center left', bbox_to_anchor=(1, 0.50))
    
plt.ylabel("Size of Receptive Field ($\sigma$ of modelled Gabor Patch)")
plt.xlabel("Epoch")
plt.savefig(f'{save_dir}/compare_gabor_sigma_allRFs_epoch-35-60_plot-violin.png', bbox_inches="tight")





# ax = sns.catplot(data=rf_param_df, kind="violin",y="gabor_model_loss", x="Network_epoch", hue="best_model", )
# # plt.ylabel("Shape Bias (%)")
# # plt.xlabel("Category")
# plt.title(f'Plot loss for Gabor model - depending on whicg model was preferred')
# # fig.set_figwidth(20)
# plt.savefig(f'{save_dir}/Gabor_loss_plot-violin.png', bbox_inches="tight")


# ax = sns.catplot(data=rf_param_df, kind="violin",y="dog_model_loss", x="Network_epoch", hue="best_model", )
# # plt.ylabel("Shape Bias (%)")
# # plt.xlabel("Category")
# plt.title(f'Plot loss for DOG model - depending on whicg model was preferred')
# # fig.set_figwidth(20)
# plt.savefig(f'{save_dir}/DOG_loss_plot-violin.png', bbox_inches="tight")

# # MELT THIS DF to get     # 