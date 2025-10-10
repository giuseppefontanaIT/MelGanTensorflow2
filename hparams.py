import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import warnings

warnings.filterwarnings('ignore')

# Iperparametri del Generatore
n_mels:int=256
audio_duration:int=27
up_factors:list[int]=[8,8,2,2]
dil_rates:list[int]=[1,3,9]
gen_kernel_size=3
rec_field_in_timesteps=27

# Iperparametri del Discriminatore
strided_convolutions=4
strides=4

# Iperparametri del Multiscale Discriminator
downscale_factors:list[int,int]=[2,4]
ms_kernel_size=4
