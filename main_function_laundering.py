from __future__ import print_function
import pandas as pd
import librosa
import soundfile as sf
import pyroomacoustics as pra
import pyroomacoustics
import numpy as np
import os
from audiomentations import AddShortNoises, PolarityInversion
import scipy.io.wavfile as wavf
import warnings
import subprocess
import threading
warnings.filterwarnings("ignore")

def room_reverb(audio, fs, rt60_tgt):
    
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
    #print("exit")
    return room.mic_array

def room_reverb_out(reverb_3_audio,master_reverb_out_path,filename,save_processed_audio_dir):
    output_path = master_reverb_out_path + "{}/{}/".format(save_processed_audio_dir,row["speaker"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    new_filename = output_path + filename + '_{}'.format(save_processed_audio_dir)
    # sf.write(new_filename + '.flac', reverb_3_audio, sr, format='flac', subtype='PCM_16')
    reverb_3_audio.to_wav(
        new_filename + '.wav',
        norm=True,
        bitdepth=np.int16,
    )
    

def noise_add(noise_path,snr_rangei,snr_rangej):
    transform = AddShortNoises(
        sounds_path=noise_path,
        min_snr_in_db=snr_rangei,
        max_snr_in_db=snr_rangej,
        noise_rms="relative_to_whole_input",
        min_time_between_sounds=2.0,
        max_time_between_sounds=8.0,
        noise_transform=PolarityInversion(),
        p=1.0
    )
    
    return transform

def noise_add_out(audio_data,sr,transform,fs,master_noise_out_path,save_processed_audio_dir,row,filename):
    fs = fs
    #output filename
    #print(sr,fs)
    augmented_white_noise = transform(audio_data, sample_rate=sr)
    output_path = master_noise_out_path + "{}/{}/".format(save_processed_audio_dir,row["speaker"])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # print(output_path)
    output_path = output_path + filename + "_" +save_processed_audio_dir+".wav"
    #write array to wav file.
    noise_wav = wavf.write(output_path, sr, augmented_white_noise)

def recompression(filename,input_file, output_folder,output_folder2,bit_rate):
    basefile = os.path.basename(input_file)
    file = os.path.splitext(basefile)[0]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)
    print("t")
    # Compression (wav to mp3)
    c_file = output_folder +"{}_{}.mp3".format(filename,bit_rate)
    subprocess.run(['ffmpeg', '-i', input_file, '-b:a', bit_rate, c_file])

   # Compression (mp3 to wav)
    dc_file = output_folder2 +"{}_{}.wav".format(filename,bit_rate)
    subprocess.run(['ffmpeg', '-i',c_file,dc_file])

def downsample(y,sr, filename, output_folder, factor):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # Up-sample by a factor 
    y_downsampled = librosa.resample(y, orig_sr=sr, target_sr=factor)

    # Save the up-sampled audio as .wav files
    sf.write(os.path.join(output_folder, f'{filename}_resample_{factor}.wav'),y_downsampled, factor,subtype='PCM_16')


# path to the database
speaker = ["Barack_Obama","Donald_Trump","Elon_Musk","Emma_Watson","Hillary_Clinton","Joe_Biden","Kamala_Harris","Mike_Pence","Nikki_Haley","Vivek_Ramaswamy"]
datadf = pd.DataFrame()
for sp in speaker:
    #print(sp)
    data_dir = 'Famous_Figures/Data/two_class_data/{}/test/ElevenLabs/'.format(sp)
    file_paths = pd.DataFrame({"path" :[os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, filename))]})
    #print(file_paths)
    datadf = datadf._append(file_paths,ignore_index=True)

datadf[['folder', 'sub', 'subsub',"speaker","train","type","filename"]] = datadf['path'].str.split('/', expand=True)
datadf[["audioname","format"]] = datadf['filename'].str.split('.', expand=True)

save_processed_audio_dir = 'reverb_0_3'
master_data_dir = 'Famous_Figures/Data/two_class_data/'


street_0_05 = noise_add("/data/noises/STREET-CITY-2.wav",0,0.5)
street_10_105 = noise_add("/data/noises/STREET-CITY-2.wav",10,10.5)
street_20_205 = noise_add("/data/noises/STREET-CITY-2.wav",20,20.5)

volvo_0_05 = noise_add("/data/noises/volvo.wav",0,0.5)
volvo_10_105 = noise_add("/data/noises/volvo.wav",10,10.5)
volvo_20_205 = noise_add("/data/noises/volvo.wav",20,20.5)

babble_0_05 = noise_add("/data/noises/babble.wav",0,0.5)
babble_10_105 = noise_add("/data/noises/babble.wav",10,10.5)
babble_20_205 = noise_add("/data/noises/babble.wav",20,20.5)

white_0_05 = noise_add("/data/noises/white.wav",0,0.5)
white_10_105 = noise_add("/data/noises/white.wav",10,10.5)
white_20_205 = noise_add("/data/noises/white.wav",20,20.5)

cafe_0_05 = noise_add("/data/noises/CAFE-FOODCOURTB-1.wav",0,0.5)
cafe_10_105 = noise_add("/data/noises/CAFE-FOODCOURTB-1.wav",10,10.5)
cafe_20_205 = noise_add("/data/noises/CAFE-FOODCOURTB-1.wav",20,20.5)


for index, row in datadf.iterrows():
    filename = row["audioname"]

    # path to the file
    full_file = master_data_dir + row["speaker"]+"/" +row["train"]+"/" +row["type"]+"/" +filename+'.wav'
    # print(full_file)
    # read audio file
    audio_data, sr = librosa.load(full_file,sr = None)
    # functions
    # reverb_3_audio = room_reverb(audio_data, sr, 0.3)
    # reverb_6_audio = room_reverb(audio_data, sr, 0.6)
    # reverb_9_audio = room_reverb(audio_data, sr, 0.9)
    # room_reverb_out(reverb_3_audio,"Laundered_speaker_audios/launderedreverb/",filename,"rt_03")
    # room_reverb_out(reverb_6_audio,"Laundered_speaker_audios/launderedreverb/",filename,"rt_06")
    # room_reverb_out(reverb_9_audio,"Laundered_speaker_audios/launderedreverb/",filename,"rt_09")
    # bitrate = ["16k","64k","128k","196k","256k","320k"]
    # for bit_rate in bitrate:
    #      recompression(filename,full_file, "Laundered_speaker_audios/launderedcompression/{}/{}/".format(bit_rate, row["speaker"]),"Laundered_speaker_audios/launderedrecompression/{}/{}/".format(bit_rate, row["speaker"]),bit_rate)

    # factors = [    11025 ,
    # 8000,
    # 22050,
    # 44100]
    # for factor in factors:
    #     downsample(audio_data,sr, filename,  "Laundered_speaker_audios/downsample/{}/{}/".format(factor,row["speaker"]), factor)


    t1 = threading.Thread(target = noise_add_out,args=(audio_data,sr,volvo_0_05,16000,"/data/Laundered_speaker_audios/test/launderednoise/","volvo_0",row,filename))
    t2 = threading.Thread(target =noise_add_out,args=(audio_data,sr,volvo_10_105,16000,"/data/Laundered_speaker_audios/test/launderednoise/","volvo_10",row,filename))
    t3 = threading.Thread(target =noise_add_out,args=(audio_data,sr,volvo_20_205,16000,"/data/Laundered_speaker_audios/launderednoise/","volvo_20",row,filename))

    
    t4 = threading.Thread(target =noise_add_out,args=(audio_data,sr,babble_0_05,16000,"/data/Laundered_speaker_audios/test/launderednoise/","babble_0",row,filename))
    t5 = threading.Thread(target =noise_add_out,args=(audio_data,sr,babble_10_105,16000,"/data/Laundered_speaker_audios/test/launderednoise/","babble_10",row,filename))
    t6 = threading.Thread(target =noise_add_out,args=(audio_data,sr,babble_20_205,16000,"/data/Laundered_speaker_audios/test/launderednoise/","babble_20",row,filename))

    t7 = threading.Thread(target =noise_add_out,args=(audio_data,sr,white_0_05,16000,"/data/Laundered_speaker_audios/test/launderednoise/","white_0",row,filename))
    t8 = threading.Thread(target =noise_add_out,args=(audio_data,sr,white_10_105,16000,"/data/Laundered_speaker_audios/test/launderednoise/","white_10",row,filename))
    t9 = threading.Thread(target =noise_add_out,args=(audio_data,sr,white_20_205,16000,"/data/Laundered_speaker_audios/test/launderednoise/","white_20",row,filename))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    t9.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    t7.join()
    t8.join()
    t9.join()

    print(row["audioname"],"done")
    
    if not sr == 44100:
        print(filename,sr)

    # print(filename)
    
    # 

    # 

