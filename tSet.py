import math, random
import torch
import torchaudio
import os
import pandas as pd
from torchaudio import transforms
from IPython.display import Audio
from torch.utils.data import DataLoader, Dataset, random_split





directory = '/home/horacio/PycharmProjects/musicGenreClassifier/musicGenreSoundTracks/Converted/'
class AudioUtil():
    def openFile(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    #convert to 2 channels since most models expect all items to have the same dimension
    def rechannel(aud, newChannel):
        sig, sr = aud
        if(sig.shape[0] == newChannel):
            #dont do anything it's fine
            return aud
        if(newChannel == 1):
            #convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            #Convert from mono to steoreo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return((resig, sr))

    #Standardize sampling rate so that all arrays have the same dimensiones.
    def resample(aud, newsr):
        sig, sr = aud

        if(sr == newsr):
            #ok
            return aud
        num_channels = sig.shape[0]
        #Resample the first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if(num_channels > 1):
            #Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return((resig, newsr))

    #Resize to the same length
    #Resize all audio samples to have the same length by either extending its duration by padding
    #It with silence or by truncating it.
    #Pad or truncate the signal to a fiex length in miliseconds. 'max_ms'
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if(sig_len > max_len):
            #Truncate the signal to the given length
            sig = sig[:,:max_len]
        elif(sig_len < max_len):
            #Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            #Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

            return (sig, sr)

    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    #Generate a spectrogram :D
    def spectrogram(aud, n_mels=64, n_fft=1024, hop_len=0):
        sig, sr = aud
        top_db = 80
        #A spectrogram has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)

        #Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec


# custom object to load data
class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    # Number of items in dataset
    def __len__(self):
        return len(self.df)

    # i'th item in dataset
    def __getitem__(self, idx):
        audio_file = self.data_path + self.df.loc[idx, 'path']
        # class id
        class_id = self.df.loc[idx, 'classID']

        aud = AudioUtil.openFile(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectrogram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id

#Create the DataFrame
#Note to self, next time get a dataset with a csv or dataframe bozo
file_paths = []
genres = []

for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith('.wav'):
            genre = os.path.basename(root)
            file_paths.append(os.path.join(root, file))
            genres.append(genre)

data = {'File Path': file_paths, 'Genre': genres}
df = pd.DataFrame(data)


sound_Datasets = []
#iterate through folders and songs
for genreFolder in os.listdir(directory):
    genrePath = os.path.join(directory, genreFolder)
    for album in os.listdir(genrePath):
        albumPath = os.path.join(genrePath, album)
        for file in os.listdir(albumPath):
            filePath = os.path.join(albumPath, file)
            if file.endswith('.wav'):
                dataset = SoundDS(df, filePath)
                sound_Datasets.append(dataset)


print("you've made it far with Fabi")



