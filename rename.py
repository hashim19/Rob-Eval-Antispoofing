# Read from txt
input_file_path = '100_samples.txt'
with open(input_file_path, 'r') as input_file:
    lines = input_file.readlines()

# Replace "_P1" with "_P1_PP_RR_01" "02 for mod 2" and "03 for mod 3"
modified_lines = [line.replace('_P1', '_P1_PP_RR_01') for line in lines]

# Change for mod 2 and mod 3
output_file_path = 'RR_Mod_1.txt'
with open(output_file_path, 'w') as mod_file:
    mod_file.writelines(modified_lines)