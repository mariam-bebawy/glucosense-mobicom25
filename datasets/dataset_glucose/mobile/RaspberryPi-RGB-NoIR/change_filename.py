import os

directory = './'  # Adjust path as needed

for filename in os.listdir(directory):
    if filename.endswith('_RGB_NoIR.mat'):
        # Create new filename by replacing '_RGB_NoIR.mat' with '_RGB.mat'
        new_filename = filename.replace('_RGB_NoIR.mat', '_RGB.mat')
        
        # Construct full file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')