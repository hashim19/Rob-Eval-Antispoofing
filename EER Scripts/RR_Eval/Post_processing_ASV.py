#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter, filtfilt
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter, filtfilt
from IPython.display import Audio
import librosa
import soundfile as sf
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter_filtfilt(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# def butter_highpass(cutoff, fs, order=4):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
#     return b, a

# def butter_highpass_filter(data, cutoff, fs, order=4):
#     b, a = butter_highpass(cutoff, fs, order=order)
#     y = filtfilt(b, a, data)
#     return y

def process_audio_file(input_file, output_folder):
    basefile = os.path.basename(input_file)
    file = (os.path.splitext(basefile))[0]

    # Compression (flac to mp3)
    compressed_file = os.path.join(cf, f'{file}_pp_c.mp3')
    os.system(f'ffmpeg -i {input_file} {compressed_file}')

    # Decompression (mp3 to wav)
    dc_file = os.path.join(dc, f'{file}_pp_dc.wav')
    dc_file1 = os.path.join(dc1, f'{file}_pp_dc1.wav')
    dc_file2 = os.path.join(dc2, f'{file}_pp_dc2.wav')
    os.system(f'ffmpeg -i {compressed_file} {dc_file}')
    os.system(f'ffmpeg -i {compressed_file} -b:a 128k {dc_file1}')
    os.system(f'ffmpeg -i {compressed_file} -b:a 256k {dc_file2}')

    # # Load the original audio file
    y, sr = librosa.load(dc_file)



    # Specify the fraction of Nyquist frequency for the cutoff
    nyquist_fraction = 0.1  #adjustable
    lp_nyquist_fraction = 0.2  #adjustable

    # Calculate the cutoff frequency
    cutoff_frequency = nyquist_fraction * (sr / 2)
    lp_cutoff_frequency = lp_nyquist_fraction * (sr / 2)


    # # Apply high-pass filtering
    # y_highpass = butter_highpass_filter(y, cutoff_frequency, sr)

    # sf.write(os.path.join(hf, f'{file}_pp_hf.wav'), y_highpass, sr)

    # Apply the low-pass filter using lfilter
    filtered_data = butter_lowpass_filter(y, lp_cutoff_frequency, sr)

    # Apply the low-pass filter using filtfilt
    filtered_data_filtfilt = butter_lowpass_filter_filtfilt(y, lp_cutoff_frequency, sr)

    # Save the filtered audio as a 16kHz .wav file
    sf.write(os.path.join(lf, f'{file}_pp_lf.wav'),filtered_data, sr)
    sf.write(os.path.join(lf1, f'{file}_pp_lf1.wav'),filtered_data_filtfilt, sr)


    # Up-sample by a factor of 2
    y_upsampled = librosa.resample(y, orig_sr=sr, target_sr=sr*2)

    # Down-sample by a factor of 2
    y_downsampled = librosa.resample(y, orig_sr=sr, target_sr=sr//2)

    # Create an Audio object with the original sample rate
    audio_original = Audio(y, rate=sr)

    # Save the up-sampled and down-sampled audio as .wav files
    sf.write(os.path.join(us, f'{file}_pp_us.wav'),y_upsampled, sr*2)
    sf.write(os.path.join(ds, f'{file}_pp_ds.wav'), y_downsampled, sr//2)

    # Write information to a text file
    with open(os.path.join(output_folder, f'{file}_info.txt'), 'w') as info_file:
        info_file.write(f'SPEAKER_ID: {file[:7]}\n')
        info_file.write(f'AUDIO_FILE_NAME: {file}\n')
        info_file.write(f'SYSTEM_ID: {"-"}\n')  # You can modify this based on your system ID logic
        info_file.write(f'KEY: {"bonafide" if "bonafide" in input_file else "spoof"}\n')
        info_file.write(f'POST_PROCESSING_TYPE: {" ".join(file.split("_")[2:])}\n')

# Folder containing the audio files
folder_path = 'C:/Users/gokar/Downloads/flac-20231226T055550Z-003/flac'
output_folder = 'C:/Users/gokar/Downloads/ASV_output'
os.makedirs(output_folder, exist_ok=True)

# Creating subfolders inside the output folders
subfolders = ['compressed', 'decompressed', 'decompressed1', 'decompressed2', 'lf', 'lf1', 'upsampled', 'downsampled']
for subfolder in subfolders:
    os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)
batch_size = 10
# Get a list of files in the folder
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
for i in range(0, len(files), batch_size):
    batch_files = files[i:i + batch_size]
    for file in files:
        input_file = os.path.join(folder_path, file)
        cf = os.path.join((output_folder), 'compressed')
        dc = os.path.join((output_folder), 'decompressed')
        dc1 = os.path.join((output_folder), 'decompressed1')
        dc2 = os.path.join((output_folder), 'decompressed2')
        lf = os.path.join((output_folder), 'lf')
        lf1 = os.path.join((output_folder), 'lf1')
        us = os.path.join((output_folder), 'upsampled')
        ds = os.path.join((output_folder), 'downsampled')
        process_audio_file(input_file, output_folder)
        # hf = os.path.join((output_folder),'hf')

### !pip install --upgrade paramiko cryptography

# In[2]:


get_ipython().system('pip install librosa')


# In[ ]:




