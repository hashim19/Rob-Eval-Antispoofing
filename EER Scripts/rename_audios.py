import os

# Mod_2 and Mod_3 paths
mod_2_folder = 'C:/Users/aesal/Desktop/Room_Rev/Mod_2'
mod_3_folder = 'C:/Users/aesal/Desktop/Room_Rev/Mod_3'

# Added 02 and 03
new_ending_mod_2 = '02'
new_ending_mod_3 = '03'

# Function to rename .flac files in a folder with a new ending
def rename_flac_files(folder_path, new_ending):
    for filename in os.listdir(folder_path):
        if filename.endswith('_P1_PP_RR_01.flac'):
            old_path = os.path.join(folder_path, filename)
            new_filename = filename.replace('_P1_PP_RR_01.flac', f'_P1_PP_RR_{new_ending}.flac')
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)

# Rename .flac files in the "Mod_2" and "Mod_3" folders
rename_flac_files(mod_2_folder, new_ending_mod_2)
rename_flac_files(mod_3_folder, new_ending_mod_3)