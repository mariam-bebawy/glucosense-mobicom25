import pandas as pd
import glob
import os

# Get all CSV files matching the pattern
csv_files = glob.glob('/local-scratch/GlucoseProject/mobicom23_mobispectral/regression/Experiment/LeaveOneOut/LeaveOneOut_Calibration_ToF*.csv')

# Print the number of files found
print(f"Found {len(csv_files)} CSV files")

if len(csv_files) == 0:
    print("No CSV files found! Check the path and file pattern.")
    exit()

# Create a dictionary to store dataframes and their subject numbers
sheets_dict = {}

# First, read all files and store them in the dictionary
for csv_file in csv_files:
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Get the subject number from the file name
        subject_num = int(csv_file.split('ToF')[-1].split('.')[0])
        
        # Store in dictionary with subject number as key
        sheets_dict[subject_num] = df
        
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")

# Create a new Excel writer object
with pd.ExcelWriter('LeaveOneOut_GooglePixel_Calibration_Results.xlsx', engine='openpyxl') as writer:
    # Write sheets in order of subject number
    for subject_num in sorted(sheets_dict.keys()):
        sheet_name = f'Subject_{subject_num}'
        sheets_dict[subject_num].to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Added sheet: {sheet_name}")

print("Excel file created successfully with all CSV data in order!")