import pandas as pd
import librosa
import soundfile as sf

# path to the database
data_dir = '/home/alhashim/Data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac/'  

# path to protocol file
protocol_file = '/home/alhashim/Data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

# read the protcol file
evalprotcol_df = pd.read_csv(protocol_file, sep=" ", names=["Speaker_Id", "AUDIO_FILE_NAME", "Not_Used_for_LA", "SYSTEM_ID", "KEY"])

print(evalprotcol_df)

save_processed_audio_dir = 'reverb_0_3'

filename_ls = []
# loop over the protcol file
for index, row in evalprotcol_df.iterrows():

    filename = row["AUDIO_FILE_NAME"]

    # path to the file
    full_file = data_dir + filename + '.flac'

    # read audio file
    audio_data, sr = librosa.load(full_file, sr=None)

    # pass this audio data to your postprocessing function, for example reverb_3 applies reverberation of T60=0.3 to the audio
    # reverb_3_audio = reverb_3(audio_data) # just an example

    # write the new file to the specified location
    new_filename = filename + '_RT_0_3'
    sf.write(new_filename + '.flac', reverb_3_audio, sr, format='flac', subtype='PCM_16')

    filename_ls.append(new_filename)

# change filenames in the pandas dataframe
evalprotcol_df["AUDIO_FILE_NAME"] = filename_ls

# save pandas dataframe as txt file
evalprotcol_df.to_csv("reverb_0_3_protocol.txt", sep=" ", index=False, header=False)






    





    
