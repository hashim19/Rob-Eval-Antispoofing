import pandas as pd
import os
import shutil

# Read the txt file into a pandas DataFrame
txt_path = 'C:/Users/aesal/OneDrive/Desktop/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
df = pd.read_csv(txt_path, sep=" ", header=None, names=["session", "audio_file", "label_type", "label", "spoof_type"])

# Filter out rows that are not labeled as "spoof"
spoof_df = df[df['spoof_type'] == 'spoof']

# Randomly select 100 rows from the rows labeled as "spoof"
selected_df = spoof_df.sample(n=100)

# Add "_P1" to the 100 randomly selected file names
selected_df['modified_audio'] = selected_df['audio_file'].apply(lambda x: x + "_P1")

# Create a new txt file with the 100 modified file names
selected_df['output_line'] = selected_df['session'] + " " + selected_df['modified_audio'] + " " + selected_df['label_type'] + " " + selected_df['label'] + " " + selected_df['spoof_type']
selected_df['output_line'].to_csv('100_samples.txt', index=False, header=False)

# Copy the corresponding randomly selected audio files from the txt file to the new directory "100 audio samples" and append "_P1" to the filenames
audio_directory = 'C:/Users/aesal/OneDrive/Desktop/LA/ASVspoof2019_LA_eval/flac'
destination_directory = 'C:/Users/aesal/OneDrive/Desktop/LA/ASVspoof2019_LA_eval/100 audio samples'
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

for audio in selected_df['audio_file']:
    original_file_path = os.path.join(audio_directory, audio + ".flac")
    destination_file_path = os.path.join(destination_directory, audio + "_P1" + ".flac")
    shutil.copy(original_file_path, destination_file_path)

print("Process completed!")