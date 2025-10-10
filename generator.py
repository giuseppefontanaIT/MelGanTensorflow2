from hparams import *
import tensorflow as tf
import keras.layers as l
from keras.models import Model, Sequential


class ResStack(Model):
    def __init__(self,filters:int=256,dil_rate:int=1):
        super(ResStack,self).__init__()
        self.layer=Sequential(
                name="ResStack",
                layers=[
                    l.LeakyReLU(0.2),
                    l.Conv1D(filters=filters,kernel_size=3,strides=1,padding='same',dilation_rate=dil_rate),
                    l.LeakyReLU(0.2),
                    l.Conv1D(filters=filters,kernel_size=1,strides=1,padding='valid',dilation_rate=1)
                    ]
                )
        self.conv=l.Conv1D(filters=filters,kernel_size=1,strides=1)
        self.add=l.Add()

    def call(self,data):
        skip_conn=tf.identity(data)
        x=self.layer(data)
        x=self.conv(x)
        return self.add([skip_conn,x])



class Generator(Model):
    def __init__(self,n_mels:int=n_mels,audio_duration:int=audio_duration):
        super(Generator,self).__init__()
        self.layer=Sequential(
                name='MelGanGenerator',
                layers=[
                    l.Input((n_mels,audio_duration)),
                    l.Conv1D(filters=512,kernel_size=7,padding='same')
                    ]
                )

        filters=256
        for f in up_factors:
            self.layer.add(l.LeakyReLU(0.2))
            self.layer.add(
                    l.Conv1DTranspose(filters=filters, kernel_size=2 * f, strides=f, padding='same')
            )
            i=0
            for d in dil_rates:
                self.layer.add(ResStack(filters=filters, dil_rate=d))
                i+=1
            filters=filters//2

        self.layer.add(l.LeakyReLU(0.2))
        self.layer.add(l.Conv1D(filters=1, kernel_size=7,strides=1, padding='same'))
        self.layer.add(l.Activation('tanh'))


    def call(self,melspec):
        return self.layer(melspec)




if __name__=='__main__':
    mels=256
    duration=216
    gen=Generator(n_mels=mels,audio_duration=duration)

    pred=gen(tf.random.normal((1,mels,duration),dtype=tf.float32))
    print(pred.shape)
    gen.summary(
            expand_nested=True,
            show_trainable=True
            )
