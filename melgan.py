from hparams import *

from generator import Generator # implementazione del generatore
from discriminator import MultiScaleDiscriminator # implementazione del generatore
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import keras.layers as l
from keras.models import Model,Sequential
from keras.losses import BinaryCrossentropy
from keras.callbacks import History
from pystoi import stoi # la metrica Short Time Intelligibility Measure
from librosa.feature import melspectrogram,mfcc # per il calcolo dello spettrogramma e della Mel Cepstral Distortion (MCD)
from librosa import load
from functools import partial
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError as MAE
import numpy as np
import tqdm # per la Progress Bar delle epoche
from scipy.spatial.distance import euclidean # sempre per l'implementazione della MCD
from fastdtw import fastdtw

# implementazione della metrica Mel Cepstral Distortion (MCD)
def mcd(y_true,y_pred,sr=65536/1.71):
    a=np.reshape(y_true,65536)
    b=np.reshape(y_pred,65536)
    not_aligned_a=mfcc(y=a,sr=sr)
    not_aligned_b=mfcc(y=b,sr=sr)
    distance,path=fastdtw(not_aligned_a.T,not_aligned_b.T,dist=euclidean)
    return distance/len(path)

# suddivisione in batch
def batch(zipped, batch_size):
    lst= list(zipped)
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

# per praticità rinomino la funzione flatten
flatten=tf.nest.flatten


# Implementazione della MelGan
class MelGan(Model):
    def __init__(self,n_mels:int=256,audio_duration:int=5):
        super(MelGan,self).__init__()
        self.gen=Generator(n_mels=n_mels,audio_duration=audio_duration)
        self.discr=MultiScaleDiscriminator()
        self.lambda_scaling=10 # questo parametro è la lambda della feture matching loss che serve a dare più importanza alla feature matching loss che alla adversarial loss. Permette di produre un audio più realistico ma rallenta la convergenza
    
    # Fase di Forward
    def call(self,melspec,training:bool=True):
        return self.gen(melspec,training=True)
    
    # compilazione del modello
    def compile(self,*args,**kwargs):
        super(MelGan,self).compile(*args,**kwargs)
        self.stoi=partial(stoi,fs_sig=22050,extended=False)
        self.mcd=mcd
        # la learning rate del generatore è maggiore in modo tale che il generatore apprenda più velocemente del discriminatore e riesca a ingannarlo
        self.g_opt=Adam(learning_rate=1e-4,beta_1=0.5,beta_2=0.9)
        self.d_opt=Adam(learning_rate=4e-5,beta_1=0.5,beta_2=0.9)
        self.g_loss=BinaryCrossentropy(from_logits=True)
        self.feature_matching_loss=MAE()
        self.d_loss=BinaryCrossentropy(from_logits=True)

    # il train step, che viene computato per ogni batch di un'epoca
    def train_step(self,melspec,audio):
        generated_audio=self.gen(melspec,training=False)
        # Computazione della discesa del gradiente per il discriminatore
        with tf.GradientTape() as d_tape:
            real_labels=self.discr(audio,training=True)
            fake_labels=self.discr(generated_audio,training=True)
            fake_loss=[]
            # Essendo il discriminatore composto da più discriminatori in cascata esso produce 3 label per sample. Queste label non sono di shape omogenea quindi sono obbligato a calcolare la loss in maniera iterativa
            for label in fake_labels:
                for label_i in label:
                    fake_loss.append(self.d_loss(tf.zeros_like(label_i),label_i))
            real_loss=[]
            for label in real_labels:
                for label_i in label:
                    real_loss.append(self.d_loss(tf.ones_like(label_i),label_i))
            total_d_loss=tf.reduce_mean(real_loss)+tf.reduce_mean(fake_loss)
            total_d_loss=total_d_loss

        # applico il gradiente all'ottimizzatore del discriminatore
        d_grad=d_tape.gradient(total_d_loss,self.discr.trainable_weights)
        self.d_opt.apply_gradients(zip(d_grad,self.discr.trainable_weights))
       
        # Computazione della discesa del gradiente per il generatore 
        with tf.GradientTape() as g_tape:
            fake_audio=self.gen(melspec,training=True)
            fake_pred=self.discr(fake_audio,training=False)
            real_pred=self.discr(audio,training=False) 
            # stesso problema del discriminatore. Anche qui si calcola la loss in maniera iterativa
            for i,label in enumerate(fake_pred):
                for j,label_i in enumerate(label):
                    fake_loss.append(self.g_loss(tf.ones_like(label_i),label_i))
            feature_matching_loss=[]
            for i,_ in enumerate(fake_pred):
                for j,_ in enumerate(fake_pred[i]):
                    feature_matching_loss.append(self.feature_matching_loss(real_pred[i][j],fake_pred[i][j]))
            total_g_loss=tf.reduce_mean(fake_loss)+tf.reduce_mean(feature_matching_loss)*self.lambda_scaling
            metric_res=[] # per la stoi
            metric_res2=[] # per la mcd
            for index,a in enumerate(audio):
                metric_res.append(self.stoi(a,fake_audio[index]))
                metric_res2.append(self.mcd(a.numpy(),fake_audio[index].numpy()))
            metric_res=tf.reduce_mean(metric_res)
            metric_res2=tf.reduce_mean(metric_res2)

        # applico il gradiente all'ottimizzatore del generatore
        g_grad=g_tape.gradient(total_g_loss,self.gen.trainable_weights)
        self.g_opt.apply_gradients(zip(g_grad,self.gen.trainable_weights))
        
        # ritorno le informazioni in modo tale che possano essere visualizzate tramite la print
        return {'d_loss':total_d_loss,'g_loss':total_g_loss,'stoi':metric_res,'mcd':metric_res2}

    # funzione di addestramento della MelGan. Qui avviene la Backpropagation
    def fit(self,melspec,audio,batch_size,epochs=100,checkpoints:int=None):
        dataset=zip(melspec,audio)
        dataset=batch(dataset,batch_size)
        self.learning_curves=[] 
        for i in range(epochs):
            for batch_x,batch_y in tqdm.tqdm(dataset[0],ncols=50):
                batch_x=tf.expand_dims(batch_x,axis=0)
                batch_y=tf.expand_dims(batch_y,axis=0)
                data=self.train_step(batch_x,batch_y)
                report=f"Epoch {i+1}/{epochs}: "
                self.learning_curves.append(data)
                for key,value in data.items():
                    report=report+f"{key}:{value:.3e} "
                print(report)
            print(report)
            if checkpoints!=None:
                np.save(f"learning_curves_{i}",self.learning_curves)
                if (i+1)%checkpoints==0 or i==epochs-1:
                    print('Saving weights')
                    self.gen.save_weights(f"weights/generator/generator_epoch_{i}.weights.h5",overwrite=True)
                    print("Generator weights saved")
                    self.discr.save_weights(f"weights/discriminator/multiscalediscriminator_epoch_{i}.weights.h5",overwrite=True)
                    print("Multiscale Discriminator weights saved")

if __name__=='__main__':
    from dataloader import *
    duration=1.71
    mels,audio=load_dataset(n_audios=30,sr_audio=65536/duration)
    mels=(mels-np.min(mels))/(np.max(mels)-np.min(mels))
    audio=(audio-np.min(audio))/(np.max(audio)-np.min(audio))
    
    gan=MelGan(n_mels=mels[0].shape[-2],audio_duration=mels[0].shape[-1])
    pred=gan.gen(tf.expand_dims(mels[0],axis=0))
    pred=gan.discr(pred)
    gan.compile()

    history=gan.fit(mels,audio,batch_size=10,checkpoints=5)
    



    

    
