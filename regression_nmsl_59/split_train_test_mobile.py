import csv
hs_test = ['2749','2754','2759', '2771', '2778', '2789', '2768', '2783', '2797', '2803', '2811', '2821', '2906', '2911', '2816', '2882', '3208', '2867', '2899', '2923', '2824', '2830', '2850', '2855', '2842', '2884', '2889', '2894', '2977', '2862', '2875', '2905', '2916', '2970', '3280', '3285', '3598', '3486', '3678', '3683', '3688', '3690', '3696', '3706', '3738', '3743', '3703', '3734', '3712', '3718', '3761', '3861', '3929', '3722', '3781', '3794', '3727', '3778', '3751', '3756', '3749', '3836', '3841', '3771', '3786', '3768', '3959', '3872', '3934', '3956', '3801', '3806', '3811', '3825', '3833', '3844', '3853', '3824', '3850', '3863', '3869', '3924', '3950', '3979', '3944', '3964', '3973', '3984'];
input_file = 'labels_GooglePixel(all).csv'
with open(input_file, 'r') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Read the header row
    selected_rows= [row for row in reader if row[1] in hs_test]
    #selected_rows = [row for row in reader if row[1] not in hs_test]
    #print(selected_rows_train)

with open('pixel4xl_test.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)  # Write the header row
    writer.writerows(selected_rows)  # Write the selected rows

#with open('pixel4xl_test.csv', 'w', newline='') as outfile:
#    writer = csv.writer(outfile)
#    writer.writerow(header)  # Write the header row
#    writer.writerows(selected_rows_test)  # Write the selected rows



