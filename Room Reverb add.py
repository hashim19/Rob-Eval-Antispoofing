import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile
import soundfile as sf
import os
# The desired reverberation time and dimensions of the room
rt60_tgt = 0.6  # seconds
room_dim = [10, 7.5, 3.5]  # meters
master_input_path = "C:/Users/SS Studios/Downloads/LA/LA/ASVspoof2019_LA_eval/flac/"
master_output_path = "D:/RT/0.9/ASV/"

#fs, audio = wavfile.read(r"C:\Users\SS Studios\Downloads\LA_E_1000273.wav")



#file_paths = [os.path.join(master_input_path, filename) for filename in os.listdir(master_input_path) if os.path.isfile(os.path.join(master_input_path, filename))]
#for path in file_paths:
#    print(path[-18:-5])
file_paths = [r"C:\Users\SS Studios\Downloads\LA\LA\ASVspoof2019_LA_eval\flac\LA_E_8076347.flac"]

for path in file_paths:  
      audio, fs = sf.read(path)
    
      e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    
      room = pra.ShoeBox(
          room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
      )
    
      room.add_source([2.5, 3.73, 1.76], signal=audio, delay=0.5)
    
      mic_locs = np.c_[
          [6.3, 4.87, 1.2], [6.3, 4.93, 1.2],  # mic 1  # mic 2
      ]
    
      room.add_microphone_array(mic_locs)
      room.simulate()
    
      room.mic_array.to_wav(
          "D:/RT/m/ASV/{}_RT_0_6.wav".format(path[-17:-5]),
          norm=True,
          bitdepth=np.int16,
      )
    
      # measure the reverberation time
      rt60 = room.measure_rt60()
      print("The desired RT60 was {}".format(rt60_tgt))
      print("The measured RT60 is {}".format(rt60[1, 0]))
  
#audio, fs = sf.read(r"C:\Users\SS Studios\Downloads\LA_E_1000273.wav")

#e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

#room = pra.ShoeBox(
#    room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
#)

#room.add_source([2.5, 3.73, 1.76], signal=audio, delay=0.5)

#mic_locs = np.c_[
#    [6.3, 4.87, 1.2], [6.3, 4.93, 1.2],  # mic 1  # mic 2
#]

#room.add_microphone_array(mic_locs)
#room.simulate()

#room.mic_array.to_wav(
#    r"C:\Users\SS Studios\Downloads\guitar_16k_reverb_{args.method}.wav",
#    norm=True,
#    bitdepth=np.int16,
#)


#rt60 = room.measure_rt60()
#print("The desired RT60 was {}".format(rt60_tgt))
#print("The measured RT60 is {}".format(rt60[1, 0]))
