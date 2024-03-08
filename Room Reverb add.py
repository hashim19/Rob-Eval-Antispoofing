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

def room_reverb(audio, fs, rt60_tgt,source,mic):
    
    room_dim = [10, 7.5, 3.5]  # meters
    source = [2.5, 3.73, 1.76]
    mic = [6.3, 4.87, 1.2]
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )

    room.add_source(source, signal=audio, delay=0.5)

    mic_locs = np.c_[mic]

    room.add_microphone_array(mic_locs)
    room.simulate()
    return room.mic_array

###########################################Change the mods here####################################
mod_name = "_RT_0_3"
rt60 = 0.3
###########################################Change the mods here####################################

filename_ls = []
# loop over the protcol file
for index, row in evalprotcol_df.iterrows():

    filename = row["AUDIO_FILE_NAME"]

    # path to the file
    full_file = data_dir + filename + '.flac'

    # read audio file
    audio_data, sr = librosa.load(full_file)

    # pass this audio data to your postprocessing function, for example reverb_3 applies reverberation of T60=0.3 to the audio
    reverb_3_audio = room_reverb(audio_data, sr,rt)
    # reverb_3_audio = reverb_3(audio_data) # just an example

    # write the new file to the specified location
    output_path = ""
    new_filename = filename + mod
    sf.write(new_filename + '.flac', reverb_3_audio, sr, format='flac', subtype='PCM_16')

    filename_ls.append(new_filename)

# change filenames in the pandas dataframe
evalprotcol_df["AUDIO_FILE_NAME"] = filename_ls

# save pandas dataframe as txt file
evalprotcol_df.to_csv("reverb_0_3_protocol.txt", sep=" ", index=False, header=False)