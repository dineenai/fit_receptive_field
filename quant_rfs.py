# python
# /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/fit_receptive_fieldls

# Load: supRN50_conv1_21_g0_60e_e60_rf_model_params.csv
# Count no of gaussians/gabors

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("rf_model_params/supRN50_conv1_21_g0_60e_e60_rf_model_params.csv") 
# Preview the first 5 lines of the loaded data 
print(data.head())

for col in data.columns:
    print(col)

print((data.best_model == 'gab').sum())
print((data.best_model == 'dog').sum())
print((data.best_model == 'test').sum())


print()

sigmas = data[["dog_sig1", "dog_sig2", "gabor_sigma","best_model"]]
print(sigmas)
# Note that the sigmas are not directly comparable!
# 

# slow
# data.loc[data['best_model'] == 'dog', 'dog_sig1'].item()

# x = data.query('best_model=='dog'')['dog_sig1']

dog_sig1_dog = data[data['best_model']=='dog']['dog_sig1']
dog_sig2_dog = data[data['best_model']=='dog']['dog_sig2']
dog_model_loss_dog = data[data['best_model']=='dog']['dog_model_loss']
gabor_sigma_gab = data[data['best_model']=='gab']['gabor_sigma']
gabor_model_loss_gab = data[data['best_model']=='gab']['gabor_model_loss']

print(dog_sig1_dog)
print(dog_sig1_dog.mean())
print(dog_sig2_dog.mean())
print(gabor_sigma_gab.mean())

# Note that all losses are tensors ==> Must convertt to scalar BEFORE sending to csv
# print(gabor_model_loss_gab.mean())
# print(data['gabor_model_loss'].mean())
# print(dog_model_loss_dog.mean())
# print(data['dog_model_loss'].mean())

# # print((gabor_model_loss_gab[0]).numpy())
# print(type((gabor_model_loss_gab[0])))
# print(gabor_model_loss_gab[0])

# create csv file with means of the sigmas 
# also add number of gabors
# number of dogs
# assuming that all rfs were one or the other
# could add a loss cut off point to exclude more complex ones 




column_names = ["mean_dog_sig1_dog", "mean_dog_sig2_dog", "mean_dog_amp1", "mean_dog_amp2", "mean_dog_count", "mean_gabor_sigma_gab", "mean_gabor_count"]

df = pd.DataFrame(columns = column_names)


mean_values
rf_model_params = pd.DataFrame(
    { 'filename': file_name, 'dog_sig1': dog_sigma1, 'dog_sig2': dog_sigma2, 'dog_amp1': dog_amp1, 'dog_amp2': dog_amp2, 'dog_center': dog_center, 'dog_image_size': dog_image_size,
                        'dog_target_std': dog_target_std, 'gabor_sigma': gabor_sigma, 'gabor_sigma': gabor_sigma, 'gabor_theta': gabor_theta, 'gabor_Lambda': gabor_Lambda, 'gabor_psi': gabor_psi,
                        'gabor_gamma': gabor_gamma, 'gabor_center': gabor_center,
                        'gabor_image_size': gabor_image_size, 'gabor_target_std': gabor_target_std, 'dog_model_loss': dog_model_loss, 'gabor_model_loss': gabor_model_loss, 'best_model': best_model})


# df.to_csv(index=False)
rf_model_params.to_csv(f'rf_model_params/supRN50_conv1_21_g0_60e_e60_rf_model_params.csv')