import pickle
import sklearn
import argparse
from torch.utils.data import DataLoader
import csv
from hsi_dataset import TrainDataset, TestDataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn

# parser = argparse.ArgumentParser(description="Spectral Regression Toolbox")
# parser.add_argument('--data_root', type=str, default='../datasets/dataset_skin/regression')
# opt = parser.parse_args()


def ARD(pred, test):
        # Evaluate using ARD
        ard = abs(pred - test) / test
        return ard

def clarke_error_grid(pred_values, ref_values,  title_string):

    #Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))

    #Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    if max(ref_values) > 400 or max(pred_values) > 400:
        print( "Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(max(ref_values), max(pred_values)))
    if min(ref_values) < 0 or min(pred_values) < 0:
        print( "Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(min(ref_values),  min(pred_values)))

    #Clear plot
    plt.clf()

    #Set up plot
    plt.scatter(ref_values, pred_values, marker='o', color='blue', s=8)
    #plt.title("Clarke Error Grid")
    plt.xlabel("Reference Glucose (mg/dL)", fontsize=15)
    plt.ylabel("Prediction Glucose (mg/dL)", fontsize=15)
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400], fontsize=15)
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400], fontsize=15)
    plt.gca().set_facecolor('white')

    #Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect((400)/(400))

    #Plot zone lines
    plt.plot([0,400], [0,400], ':', c='black')                      #Theoretical 45 regression line
    plt.plot([0, 175/3], [70, 70], '-', c='black')
    #plt.plot([175/3, 320], [70, 400], '-', c='black')
    plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')           #Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70], [84, 400],'-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290],[180, 400],'-', c='black')
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')                     #Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    plt.plot([70, 400], [56, 320],'-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180],'-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')

    #Add zone titles
    plt.text(30, 15, "A", fontsize=15)
    plt.text(370, 260, "B", fontsize=15)
    plt.text(280, 370, "B", fontsize=15)
    plt.text(160, 370, "C", fontsize=15)
    plt.text(160, 15, "C", fontsize=15)
    plt.text(30, 140, "D", fontsize=15)
    plt.text(370, 120, "D", fontsize=15)
    plt.text(30, 370, "E", fontsize=15)
    plt.text(370, 15, "E", fontsize=15)

    #Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
            zone[0] += 1    #Zone A

        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
            zone[4] += 1    #Zone E

        elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
            zone[2] += 1    #Zone C
        elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
            zone[3] += 1    #Zone D
        else:
            zone[1] += 1    #Zone B

    return plt, zone

def plot_surveillance_error_grid(pred_values, ref_values, img):
    fig, ax = plt.subplots()
    img = plt.imread(img)
    ax.imshow(img, extent=[0, 600, 0, 600])
    plt.scatter(ref_values, pred_values, color="blue", s=2)
    ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=15)
    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=15)
    plt.xticks([0, 100, 200, 300, 400, 500, 600], fontsize=15)
    plt.yticks([0, 100, 200, 300, 400, 500, 600], fontsize=15)
    # plt.title("Surveillance Error Grid for Patient {} using DRL".format(subject_id))
    # plt.savefig("{}/seg_{}min_{}.pdf".format(save_path, prediction_horizon, subject_id), dpi=600, bbox_inches='tight')
    plt.show()

def calculate_seg_risks(pred_values, ref_values, patient_id):
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))
    data_points = len(ref_values)
    risk_levels = {
        'None': 0,
        'Slight': 0,
        'Moderate': 0,
        'Great': 0,
        'Extreme': 0,
    }
  
    for pred, gt in zip(pred_values, ref_values):
        if 0 <= pred <= 120:
            ground_truth_seg_regions(gt, "A", risk_levels)
        elif 120 < pred <= 240:
            ground_truth_seg_regions(gt, "B", risk_levels)
        elif 240 < pred <= 360:
            ground_truth_seg_regions(gt, "C", risk_levels)
        elif 360 < pred <= 480:
            ground_truth_seg_regions(gt, "D", risk_levels)
        elif 480 < pred <= 600:
            ground_truth_seg_regions(gt, "E", risk_levels)
    risk_levels_percentage = {k: (v / data_points) * 100 for k, v in risk_levels.items()}
    with open("./output/seg_risk_levels_{}.csv".format(patient_id), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=risk_levels_percentage.keys())
        writer.writeheader()
        writer.writerow(risk_levels_percentage)


def ground_truth_seg_regions(val, level, risk_levels):
    if 0 <= val <= 120:
        if level == "A":
            risk_levels["None"] += 1
        elif level == "B":
            risk_levels['Slight'] += 1
        elif level == "C":
            risk_levels['Moderate'] += 1
        elif level == "D":
            risk_levels['Extreme'] += 1
        elif level == "E":
            risk_levels['Extreme'] += 1
    elif 120 < val <= 240:
        if level == "A":
            risk_levels["Slight"] += 1
        elif level == "B":
            risk_levels["None"] += 1
        elif level == "C":
            risk_levels['Slight'] += 1
        elif level == "D":
            risk_levels['Great'] += 1
        elif level == "E":
            risk_levels['Extreme'] += 1
    elif 240 < val <= 360:
        if level == "A":
            risk_levels["Moderate"] += 1
        elif level == "B":
            risk_levels["Slight"] += 1
        elif level == "C":
            risk_levels["None"] += 1
        elif level == "D":
            risk_levels['Moderate'] += 1
        elif level == "E":
            risk_levels['Great'] += 1
    elif 360 < val <= 480:
        if level == "A":
            risk_levels["Great"] += 1
        elif level == "B":
            risk_levels["Moderate"] += 1
        elif level == "C":
            risk_levels["Slight"] += 1
        elif level == "D":
            risk_levels["None"] += 1
        elif level == "E":
            risk_levels['Moderate'] += 1
    elif 480 < val <= 600:
        if level == "A":
            risk_levels["Extreme"] += 1
        elif level == "B":
            risk_levels["Great"] += 1
        elif level == "C":
            risk_levels["Moderate"] += 1
        elif level == "D":
            risk_levels["Slight"] += 1
        elif level == "E":
            risk_levels["None"] += 1

# Get the ARD per image, and compute the CDF
def CDF(ard):
    # Step 1: Sort the ARD values
    sorted_ard = np.sort(ard)

    # Step 3: Compute CDF values
    cdf = np.arange(1, len(sorted_ard) + 1) / len(sorted_ard)

    # Step 4: Plot the CDF
    plt.plot(sorted_ard, cdf, marker='.', linestyle='none')
    plt.xlabel('Absolute Relative Difference (ARD)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Absolute Relative Difference')
    plt.grid(True)
    plt.show()



def plot_overlapped_error_grids(pred_values, ref_values, seg_img, show=True, save_path=None):
    plt.clf()
    fig, ax = plt.subplots()

    # Plot SEG background image
    seg_img = plt.imread(seg_img)
    ax.imshow(seg_img, extent=[0, 600, 0, 600])

    # Plot data points
    ax.scatter(ref_values, pred_values, color="blue", s=1)

    # Overlay CEG
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 600])
    ax.set_aspect(1.0)

    # Plot CEG zone lines
    ax.plot([0, 400], [0, 400], ':', c='black') # Theoretical 45-degree line
    ax.plot([0, 175/3], [70, 70], '-', c='black')
    ax.plot([175/3, 400/1.2], [70, 400], '-', c='black')
    ax.plot([70, 70], [84, 400], '-', c='black')
    ax.plot([0, 70], [180, 180], '-', c='black')
    ax.plot([70, 290], [180, 400], '-', c='black')
    ax.plot([70, 70], [0, 56], '-', c='black')
    ax.plot([70, 400], [56, 320], '-', c='black')
    ax.plot([180, 180], [0, 70], '-', c='black')
    ax.plot([180, 400], [70, 70], '-', c='black')
    ax.plot([240, 240], [70, 180], '-', c='black')
    ax.plot([240, 400], [180, 180], '-', c='black')
    ax.plot([130, 180], [0, 70], '-', c='black')
    plt.hlines(y=400, xmin=0, xmax=400, linestyle='dashed', color='black')
    plt.vlines(x=400, ymin=0, ymax=400, linestyle='dashed', color='black')

    # Add zone titles for CEG
    ax.text(30, 15, "A", fontsize=10)
    ax.text(370, 260, "B", fontsize=10)
    ax.text(280, 370, "B", fontsize=10)
    ax.text(160, 370, "C", fontsize=10)
    ax.text(160, 15, "C", fontsize=10)
    ax.text(30, 140, "D", fontsize=10)
    ax.text(370, 120, "D", fontsize=10)
    ax.text(30, 370, "E", fontsize=10)
    ax.text(370, 15, "E", fontsize=10)

    # Set labels and limits
    ax.set_xlabel("Measured Blood Glucose Values (mg/dL)", fontsize=10)
    ax.set_ylabel("Predicted Blood Glucose Values (mg/dL)", fontsize=10)

    # Show or save the plot
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)


# def main():
#         train_data = TrainDataset(data_root, 'labels_s1_train.txt')
#         test_data = TestDataset(data_root, 'labels_s1_test.txt')
#         # train_data = DatasetFromDirectory(data_root,"", fruit)
#         print("Total Samples:", len(train_data))
#         # data = DataLoader(dataset=train_data)

#         X, y = [],[]

#         for sig, label in test_data:
#                 X.append(sig.squeeze())
#                 y.append(label.ravel())

#         X = numpy.asarray(X)
#         y = numpy.asarray(y)



# if __name__ == "__main__":
#         data_root = opt.data_root
#         main()
