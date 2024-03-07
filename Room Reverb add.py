import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
import soundfile as sf
import os

def room_reverb(rt60_tgt,source,mic,path,output_path,name_suff, file_name):
    audio, fs = sf.read(path)

    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )

    room.add_source(source, signal=audio, delay=0.5)

    mic_locs = np.c_[mic]

    room.add_microphone_array(mic_locs)
    room.simulate()
    output_path_name = "{}/{}{}.flac".format(output_path, file_name, name_suff)
    room.mic_array.to_wav(
        output_path_name,
        norm=True,
        bitdepth=np.int16,
    )


if __name__ == "__main__":
    rt60 = [0.3,0.6,0.9]
    name_suff = ['0_3','0_6','0_9']
    room_dim = [10, 7.5, 3.5]  # meters
    source = [2.5, 3.73, 1.76]
    mic = [6.3, 4.87, 1.2]
    
    #change these inputs
    path = r"C:\Users\SS Studios\Desktop\Laundering Attach\audios\sample.flac"
    output = "C:/Users/SS Studios/Desktop/Laundering Attach/Laundered/rr"
    file = "filename_"
    for rt60_tgt in rt60:
        room_reverb(rt60_tgt,source,mic,path, output,name_suff, file)
    