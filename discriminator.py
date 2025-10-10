from hparams import *
from generator import Generator
import tensorflow as tf
import keras.layers as l
from keras.models import Model,Sequential
from librosa.feature import melspectrogram

class Discriminator(Model):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.layer=[
            Sequential(
                name='Discr_layer1',
                layers=[
                    l.Conv1D(filters=16,kernel_size=15,strides=1,padding='same'),
                    l.LeakyReLU(0.2),
                    ]
                ),
            Sequential(
                name='Discr_layer2',
                layers=[
                    l.Conv1D(filters=64,kernel_size=41,strides=4,padding='same',groups=4),
                    l.LeakyReLU(0.2),
                    ]
                ),
            Sequential(
                name='Discr_layer3',
                layers=[
                    l.Conv1D(filters=256,kernel_size=41,strides=4,padding='same',groups=16),
                    l.LeakyReLU(0.2),
                    ]
                ),
            Sequential(
                name='Discr_layer4',
                layers=[
                    l.Conv1D(filters=1024,kernel_size=41,strides=4,padding='same',groups=64),
                    l.LeakyReLU(0.2),
                    ]
                ),
            Sequential(
                name='Discr_layer5',
                layers=[
                    l.Conv1D(filters=1024,kernel_size=41,strides=4,padding='same',groups=256),
                    l.LeakyReLU(0.2),
                    ]
                ),
            Sequential(
                name='Discr_layer6',
                layers=[
                    l.Conv1D(filters=1024, kernel_size=5, strides=1, padding='same'),
                    l.LeakyReLU(0.2)
                ]),
            Sequential(
                name='Discr_layer7',
                layers=[
                    l.Conv1D(filters=1, kernel_size=3, strides=1,padding='same'),
                ])
            
            ]

    def call(self,data):
        result=[]
        x=data
        for layer in self.layer:
            x=layer(x)
            result.append(x)
        return result


class MultiScaleDiscriminator(Model):
    def __init__(self,kernel_size:int=ms_kernel_size,downscale_factors:list[int]=downscale_factors):
        super(MultiScaleDiscriminator,self).__init__()
        self.layer=[
                    Discriminator() for _ in range(3)
                ]
        self.avg_pool=l.AveragePooling1D(pool_size=kernel_size,strides=downscale_factors[0],padding='same')
        self.avg_pool2=l.AveragePooling1D(pool_size=kernel_size,strides=downscale_factors[1],padding='same') 

    def call(self,x):
        result=[]

        for i,layer in enumerate(self.layer):
            result.append(layer(x))
            if i==0:
                x=self.avg_pool(x)
            else:
                x=self.avg_pool2(x)

        return result




import numpy as np
if __name__=='__main__':
    
    gen=Generator(n_mels=256,audio_duration=216)
    pred=gen(tf.random.normal((1,256,216),dtype=tf.float32))
    discr=MultiScaleDiscriminator()
    pred=discr(pred)
    for p in pred:
        print("p-esimo discriminatore:")
        for h in p:
            print(f"\th-esima shape: {h.shape}")
    
