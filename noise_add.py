from __future__ import print_function
import pandas as pd
import librosa
import soundfile as sf
from audiomentations import AddShortNoises, PolarityInversion
import scipy.io.wavfile as wavf
import numpy as np
import os 

# path to the database
data_dir = '/data/Data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac/'  

# path to protocol file
protocol_file = '/data/Data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

# read the protcol file
evalprotcol_df = pd.read_csv(protocol_file, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])

print(evalprotcol_df)

# save_processed_audio_dir = 'white_noise'
filename_ls = []
snr_range = [0,0.5,10,10.5,20,20.5]
# loop over the protcol file
for i in range(0,len(snr_range),2):
    for index, row in evalprotcol_df.iterrows():
        filename = row["AUDIO_FILE_NAME"]

        # path to the file
        full_file = data_dir + filename + '.flac'

        # read audio file
        audio_data, sr = librosa.load(full_file, sr=None)
        transform = AddShortNoises(
            sounds_path="/data/noise_addition/babble.wav",
            min_snr_in_db=snr_range[i],
            max_snr_in_db=snr_range[i+1],
            noise_rms="relative_to_whole_input",
            min_time_between_sounds=2.0,
            max_time_between_sounds=8.0,
            noise_transform=PolarityInversion(),
            p=1.0
       )
        augmented_white_noise = transform(audio_data, sample_rate=22050)
        ## conver array to wav file 
        fs = 22050
        #output filename
        out_f = ("{filename}_babble_{i}_{j}".format(filename =filename ,i=snr_range[i], j=snr_range[i+1])+'.wav')
        print(out_f)
        #write array to wav file.
        noise_wav = wavf.write(out_f, fs, augmented_white_noise)

        #create a folder to save the audios
        createpath = ('data/AsvSpoofData_2019_babble_{i}_{j}'.format(i=snr_range[i], j=snr_range[i+1]))
        if not os.path.exists(createpath):
            os.makedirs(createpath)
        os.rename(out_f, os.path.join(createpath, out_f)) 

        # pass this audio data to your postprocessing function, for example reverb_3 applies reverberation of T60=0.3 to the audio
        # reverb_3_audio = reverb_3(audio_data) # just an example

        # write the new file to the specified location
        new_filename = (filename + '_babble_{i}_{j}'.format(i=snr_range[i],j=snr_range[i+1] ))
        # sf.write(new_filename + '.flac', augmented_white_noise, sr, format='flac', subtype='PCM_16')

        filename_ls.append(new_filename)

        # change filenames in the pandas dataframe
    evalprotcol_df["AUDIO_FILE_NAME"] = filename_ls

    # save pandas dataframe as txt file
    evalprotcol_df.to_csv("babble_{i}_{j}_protocol.txt".format(i=snr_range[i], j=snr_range[i+1]), sep=" ", index=False, header=False)






    





    