from librosa import load,power_to_db
from librosa.feature import melspectrogram
from librosa.display import specshow
import os
import numpy as np
import matplotlib.pyplot as plt

   

# deprecated. E' un refuso. Ignorare
def calculate_sr(duration:float,hop_lenght:float,samples_desired:int,n_fft:int):
    return(((samples_desired-1)*hop_lenght)+n_fft)/duration
    


# carica il dataset in memoria e se richiesto salva gli spettrogrammi Mel
def load_dataset(audio_folder:str='audio',sr_audio:int=22050,n_audios:int=20,n_mels:int=256,duration:float|int=1.71,hop_length:int=512,save:bool=False):
    files=os.listdir(audio_folder)
    melspecs=[]
    audios=[]
    counter=0
    for f in files:
        offset=0
        for i in range(n_audios):
            y,sr=load(os.path.join(audio_folder,f),offset=offset,duration=duration)
            mel=melspectrogram(y=y,n_mels=n_mels,hop_length=hop_length)
            mel=power_to_db(mel,ref=np.max)
            if save==False:
                melspecs.append(mel)
                y,_=load(os.path.join(audio_folder,f),sr=sr_audio,offset=offset,duration=duration)
                audios.append(y[:,np.newaxis])
            offset+=duration
            if save==True:
                specshow(mel,sr=sr,x_axis='time',y_axis='mel')
                plt.title('mel spectrogram')
                plt.colorbar(format='%+02.0f dB')
                plt.savefig(str(counter)+'.jpg')
                plt.figure()
                counter+=1
                np.save(f"mel_{counter}",mel)
            
    return melspecs,audios


if __name__=='__main__':
    n_fft=2048
    hop_lenght=512
    duration=5
    samples=65536
    print(calculate_sr(frames=frames,duration=duration,hop_lenght=hop_lenght,n_ftt=n_ftt))



