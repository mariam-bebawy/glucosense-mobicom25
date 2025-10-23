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
from shapely.geometry import Point, Polygon

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
    plt.scatter(ref_values, pred_values, marker='o', color='black', s=8)
    plt.title(title_string + " Clarke Error Grid")
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("Prediction Concentration (mg/dl)")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
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

    # Normalize the zone counts
    zone = [x / len(ref_values) for x in zone]

    return plt, zone

def plot_surveillance_error_grid(pred_values, ref_values, img, show=True, save_path=None):
    plt.clf()
    fig, ax = plt.subplots()
    img = plt.imread(img)
    ax.imshow(img, extent=[0, 600, 0, 600])
    plt.scatter(ref_values, pred_values, color="blue", s=1)
    ax.set_xlabel("Measured Blood Glucose Values (mg/dL)")
    ax.set_ylabel("Predicted Blood Glucose Values (mg/dL)")
    # plt.title("Surveillance Error Grid for Patient {} using DRL".format(subject_id))
    # plt.savefig("{}/seg_{}min_{}.pdf".format(save_path, prediction_horizon, subject_id), dpi=600, bbox_inches='tight')
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)

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
    # with open("./output/seg_risk_levels_{}.csv".format(patient_id), 'w') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=risk_levels_percentage.keys())
    #     writer.writeheader()
    #     writer.writerow(risk_levels_percentage)
    return risk_levels_percentage


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

# A overlapped graph of CEG and SEG
def plot_overlapped_CEG_SEG(pred_values, ref_values, seg_img, show=True, save_path=None):
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

    # Add zone titles for CEG
    ax.text(30, 15, "A", fontsize=15)
    ax.text(370, 260, "B", fontsize=15)
    ax.text(280, 370, "B", fontsize=15)
    ax.text(160, 370, "C", fontsize=15)
    ax.text(160, 15, "C", fontsize=15)
    ax.text(30, 140, "D", fontsize=15)
    ax.text(370, 120, "D", fontsize=15)
    ax.text(30, 370, "E", fontsize=15)
    ax.text(370, 15, "E", fontsize=15)

    # Set labels and limits
    ax.set_xlabel("Measured Blood Glucose Values (mg/dL)")
    ax.set_ylabel("Predicted Blood Glucose Values (mg/dL)")

    # Show or save the plot
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)


def plot_overlapped_PEG_SEG(pred_values, ref_values, img, diabetes_type, show=True, save_path=None):
    plt.clf()
    fig, ax = plt.subplots()
    img = plt.imread(img)
    ax.imshow(img, extent=[0, 600, 0, 600])

    # Overlay Parkes Error Grid
    # for act, pred in zip(ref_values, pred_values):
    #     zone = parkes_error_zone_detailed(act, pred, diabetes_type)
    #     # color = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red', 4: 'red', 5: 'purple', 6: 'purple', 7: 'black'}[zone]
    #     ax.scatter(act, pred, color='blue', s=1)

    # Plot points with blue color
    ax.scatter(ref_values, pred_values, color="blue", s=1)
    
    ax.set_xlabel("Measured Blood Glucose Values (mg/dL)")
    ax.set_ylabel("Predicted Blood Glucose Values (mg/dL)")
    
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)

# Get the ARD per image, and compute the CDF
def CDF(ard, show=True, save_path=None):
    # Step 1: Sort the ARD values
    sorted_ard = np.sort(ard)

    # Step 2: Compute CDF values
    cdf = np.arange(1, len(sorted_ard) + 1) / len(sorted_ard)

    plt.clf()
    # Step 3: Plot the CDF
    plt.plot(sorted_ard, cdf, marker='.', linestyle='none')
    plt.xlabel('Absolute Relative Difference (ARD)')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Absolute Relative Difference')
    plt.grid(True)
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.clf()

# Parkes Error Grid
class _Parkes(object):
    """Internal class for drawing a Parkes consensus error grid plot"""

    def __init__(
        self,
        type,
        reference,
        test,
        units,
        x_title,
        y_title,
        graph_title,
        xlim,
        ylim,
        color_grid,
        color_gridlabels,
        color_points,
        grid,
        percentage,
        point_kws,
        grid_kws,
    ):
        # variables assignment
        self.type: int = type
        self.reference: np.array = np.asarray(reference)
        self.test: np.array = np.asarray(test)
        self.units = units
        self.graph_title: str = graph_title
        self.x_title: str = x_title
        self.y_title: str = y_title
        self.xlim: list = xlim
        self.ylim: list = ylim
        self.color_grid: str = color_grid
        self.color_gridlabels: str = color_gridlabels
        self.color_points: str = color_points
        self.grid: bool = grid
        self.percentage: bool = percentage
        self.point_kws = {} if point_kws is None else point_kws.copy()
        self.grid_kws = {} if grid_kws is None else grid_kws.copy()

        self._check_params()
        self._derive_params()

    def _check_params(self):
        if self.type != 1 and self.type != 2:
            raise ValueError("Type of Diabetes should either be 1 or 2.")

        if len(self.reference) != len(self.test):
            raise ValueError("Length of reference and test values are not equal")

        if self.units not in ["mmol", "mg/dl", "mgdl"]:
            raise ValueError(
                "The provided units should be one of the following:"
                " mmol, mgdl or mg/dl."
            )

        if any(
            [
                x is not None and not isinstance(x, str)
                for x in [self.x_title, self.y_title]
            ]
        ):
            raise ValueError("Axes labels arguments should be provided as a str.")

    def _derive_params(self):
        if self.x_title is None:
            _unit = "mmol/L" if "mmol" else "mg/dL"
            self.x_title = "Reference glucose concentration ({})".format(_unit)

        if self.y_title is None:
            _unit = "mmol/L" if "mmol" else "mg/dL"
            self.y_title = "Predicted glucose concentration ({})".format(_unit)

    def _coef(self, x, y, xend, yend):
        if xend == x:
            raise ValueError("Vertical line - function inapplicable")
        return (yend - y) / (xend - x)

    def _endy(self, startx, starty, maxx, coef):
        return (maxx - startx) * coef + starty

    def _endx(self, startx, starty, maxy, coef):
        return (maxy - starty) / coef + startx

    def _calc_error_zone(self):
        # ref, pred
        ref = self.reference
        pred = self.test

        # calculate conversion factor if needed
        n = 18 if self.units == "mmol" else 1

        maxX = max(max(ref) + 20 / n, 550 / n)
        maxY = max([*(np.array(pred) + 20 / n), maxX, 550 / n])

        # we initialize an array with ones
        # this in fact very smart because all the non-matching values will automatically
        # end up in zone A (which is zero)
        _zones = np.zeros(len(ref))

        if self.type == 1:
            ce = self._coef(35, 155, 50, 550)
            cdu = self._coef(80, 215, 125, 550)
            cdl = self._coef(250, 40, 550, 150)
            ccu = self._coef(70, 110, 260, 550)
            ccl = self._coef(260, 130, 550, 250)
            cbu = self._coef(280, 380, 430, 550)
            cbl = self._coef(385, 300, 550, 450)

            limitE1 = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [0, 35 / n, self._endx(35 / n, 155 / n, maxY, ce), 0, 0],
                        [150 / n, 155 / n, maxY, maxY, 150 / n],
                    )
                ]
            )

            limitD1L = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [250 / n, 250 / n, maxX, maxX, 250 / n],
                        [0, 40 / n, self._endy(410 / n, 110 / n, maxX, cdl), 0, 0],
                    )
                ]
            )

            limitD1U = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            25 / n,
                            50 / n,
                            80 / n,
                            self._endx(80 / n, 215 / n, maxY, cdu),
                            0,
                            0,
                        ],
                        [100 / n, 100 / n, 125 / n, 215 / n, maxY, maxY, 100 / n],
                    )
                ]
            )

            limitC1L = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [120 / n, 120 / n, 260 / n, maxX, maxX, 120 / n],
                        [
                            0,
                            30 / n,
                            130 / n,
                            self._endy(260 / n, 130 / n, maxX, ccl),
                            0,
                            0,
                        ],
                    )
                ]
            )

            limitC1U = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            30 / n,
                            50 / n,
                            70 / n,
                            self._endx(70 / n, 110 / n, maxY, ccu),
                            0,
                            0,
                        ],
                        [60 / n, 60 / n, 80 / n, 110 / n, maxY, maxY, 60 / n],
                    )
                ]
            )

            limitB1L = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [50 / n, 50 / n, 170 / n, 385 / n, maxX, maxX, 50 / n],
                        [
                            0,
                            30 / n,
                            145 / n,
                            300 / n,
                            self._endy(385 / n, 300 / n, maxX, cbl),
                            0,
                            0,
                        ],
                    )
                ]
            )

            limitB1U = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            30 / n,
                            140 / n,
                            280 / n,
                            self._endx(280 / n, 380 / n, maxY, cbu),
                            0,
                            0,
                        ],
                        [50 / n, 50 / n, 170 / n, 380 / n, maxY, maxY, 50 / n],
                    )
                ]
            )

            for i, points in enumerate(zip(ref, pred)):
                for f, r in zip(
                    [
                        limitB1L,
                        limitB1U,
                        limitC1L,
                        limitC1U,
                        limitD1L,
                        limitD1U,
                        limitE1,
                    ],
                    [1, 1, 2, 2, 3, 3, 4],
                ):
                    if f.contains(Point(points[0], points[1])):
                        _zones[i] = r

            return [int(i) for i in _zones]

        elif self.type == 2:
            ce = self._coef(35, 200, 50, 550)
            cdu = self._coef(35, 90, 125, 550)
            cdl = self._coef(410, 110, 550, 160)
            ccu = self._coef(30, 60, 280, 550)
            ccl = self._coef(260, 130, 550, 250)
            cbu = self._coef(230, 330, 440, 550)
            cbl = self._coef(330, 230, 550, 450)

            limitE2 = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            35 / n,
                            self._endx(35 / n, 200 / n, maxY, ce),
                            0,
                            0,
                        ],  # x limits E upper
                        [200 / n, 200 / n, maxY, maxY, 200 / n],
                    )
                ]
            )  # y limits E upper

            limitD2L = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            250 / n,
                            250 / n,
                            410 / n,
                            maxX,
                            maxX,
                            250 / n,
                        ],  # x limits D lower
                        [
                            0,
                            40 / n,
                            110 / n,
                            self._endy(410 / n, 110 / n, maxX, cdl),
                            0,
                            0,
                        ],
                    )
                ]
            )  # y limits D lower

            limitD2U = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            25 / n,
                            35 / n,
                            self._endx(35 / n, 90 / n, maxY, cdu),
                            0,
                            0,
                        ],  # x limits D upper
                        [80 / n, 80 / n, 90 / n, maxY, maxY, 80 / n],
                    )
                ]
            )  # y limits D upper

            limitC2L = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [90 / n, 260 / n, maxX, maxX, 90 / n],  # x limits C lower
                        [0, 130 / n, self._endy(260 / n, 130 / n, maxX, ccl), 0, 0],
                    )
                ]
            )  # y limits C lower

            limitC2U = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            30 / n,
                            self._endx(30 / n, 60 / n, maxY, ccu),
                            0,
                            0,
                        ],  # x limits C upper
                        [60 / n, 60 / n, maxY, maxY, 60 / n],
                    )
                ]
            )  # y limits C upper

            limitB2L = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            50 / n,
                            50 / n,
                            90 / n,
                            330 / n,
                            maxX,
                            maxX,
                            50 / n,
                        ],  # x limits B lower
                        [
                            0,
                            30 / n,
                            80 / n,
                            230 / n,
                            self._endy(330 / n, 230 / n, maxX, cbl),
                            0,
                            0,
                        ],
                    )
                ]
            )  # y limits B lower

            limitB2U = Polygon(
                [
                    (x, y)
                    for x, y in zip(
                        [
                            0,
                            30 / n,
                            230 / n,
                            self._endx(230 / n, 330 / n, maxY, cbu),
                            0,
                            0,
                        ],  # x limits B upper
                        [50 / n, 50 / n, 330 / n, maxY, maxY, 50 / n],
                    )
                ]
            )  # y limits B upper

            for i, points in enumerate(zip(ref, pred)):
                for f, r in zip(
                    [
                        limitB2L,
                        limitB2U,
                        limitC2L,
                        limitC2U,
                        limitD2L,
                        limitD2U,
                        limitE2,
                    ],
                    [1, 1, 2, 2, 3, 3, 4],
                ):
                    if f.contains(Point(points[0], points[1])):
                        _zones[i] = r

            return [int(i) for i in _zones]

    def plot(self, ax):
        # ref, pred
        ref = self.reference
        pred = self.test

        # calculate conversion factor if needed
        n = 18 if self.units == "mmol" else 1

        maxX = self.xlim or max(max(ref) + 20 / n, 550 / n)
        maxY = self.ylim or max([*(np.array(pred) + 20 / n), maxX, 550 / n])

        if self.type == 1:
            ce = self._coef(35, 155, 50, 550)
            cdu = self._coef(80, 215, 125, 550)
            cdl = self._coef(250, 40, 550, 150)
            ccu = self._coef(70, 110, 260, 550)
            ccl = self._coef(260, 130, 550, 250)
            cbu = self._coef(280, 380, 430, 550)
            cbl = self._coef(385, 300, 550, 450)

            _gridlines = [
                ([0, min(maxX, maxY)], [0, min(maxX, maxY)], ":"),
                ([0, 30 / n], [50 / n, 50 / n], "-"),
                ([30 / n, 140 / n], [50 / n, 170 / n], "-"),
                ([140 / n, 280 / n], [170 / n, 380 / n], "-"),
                (
                    [280 / n, self._endx(280 / n, 380 / n, maxY, cbu)],
                    [380 / n, maxY],
                    "-",
                ),
                ([50 / n, 50 / n], [0 / n, 30 / n], "-"),
                ([50 / n, 170 / n], [30 / n, 145 / n], "-"),
                ([170 / n, 385 / n], [145 / n, 300 / n], "-"),
                (
                    [385 / n, maxX],
                    [300 / n, self._endy(385 / n, 300 / n, maxX, cbl)],
                    "-",
                ),
                ([0 / n, 30 / n], [60 / n, 60 / n], "-"),
                ([30 / n, 50 / n], [60 / n, 80 / n], "-"),
                ([50 / n, 70 / n], [80 / n, 110 / n], "-"),
                (
                    [70 / n, self._endx(70 / n, 110 / n, maxY, ccu)],
                    [110 / n, maxY],
                    "-",
                ),
                ([120 / n, 120 / n], [0 / n, 30 / n], "-"),
                ([120 / n, 260 / n], [30 / n, 130 / n], "-"),
                (
                    [260 / n, maxX],
                    [130 / n, self._endy(260 / n, 130 / n, maxX, ccl)],
                    "-",
                ),
                ([0 / n, 25 / n], [100 / n, 100 / n], "-"),
                ([25 / n, 50 / n], [100 / n, 125 / n], "-"),
                ([50 / n, 80 / n], [125 / n, 215 / n], "-"),
                (
                    [80 / n, self._endx(80 / n, 215 / n, maxY, cdu)],
                    [215 / n, maxY],
                    "-",
                ),
                ([250 / n, 250 / n], [0 / n, 40 / n], "-"),
                (
                    [250 / n, maxX],
                    [40 / n, self._endy(410 / n, 110 / n, maxX, cdl)],
                    "-",
                ),
                ([0 / n, 35 / n], [150 / n, 155 / n], "-"),
                ([35 / n, self._endx(35 / n, 155 / n, maxY, ce)], [155 / n, maxY], "-"),
            ]

        elif self.type == 2:
            ce = self._coef(35, 200, 50, 550)
            cdu = self._coef(35, 90, 125, 550)
            cdl = self._coef(410, 110, 550, 160)
            ccu = self._coef(30, 60, 280, 550)
            ccl = self._coef(260, 130, 550, 250)
            cbu = self._coef(230, 330, 440, 550)
            cbl = self._coef(330, 230, 550, 450)

            _gridlines = [
                ([0, min(maxX, maxY)], [0, min(maxX, maxY)], ":"),
                ([0, 30 / n], [50 / n, 50 / n], "-"),
                ([30 / n, 230 / n], [50 / n, 330 / n], "-"),
                (
                    [230 / n, self._endx(230 / n, 330 / n, maxY, cbu)],
                    [330 / n, maxY],
                    "-",
                ),
                ([50 / n, 50 / n], [0 / n, 30 / n], "-"),
                ([50 / n, 90 / n], [30 / n, 80 / n], "-"),
                ([90 / n, 330 / n], [80 / n, 230 / n], "-"),
                (
                    [330 / n, maxX],
                    [230 / n, self._endy(330 / n, 230 / n, maxX, cbl)],
                    "-",
                ),
                ([0 / n, 30 / n], [60 / n, 60 / n], "-"),
                ([30 / n, self._endx(30 / n, 60 / n, maxY, ccu)], [60 / n, maxY], "-"),
                ([90 / n, 260 / n], [0 / n, 130 / n], "-"),
                (
                    [260 / n, maxX],
                    [130 / n, self._endy(260 / n, 130 / n, maxX, ccl)],
                    "-",
                ),
                ([0 / n, 25 / n], [80 / n, 80 / n], "-"),
                ([25 / n, 35 / n], [80 / n, 90 / n], "-"),
                ([35 / n, self._endx(35 / n, 90 / n, maxY, cdu)], [90 / n, maxY], "-"),
                ([250 / n, 250 / n], [0 / n, 40 / n], "-"),
                ([250 / n, 410 / n], [40 / n, 110 / n], "-"),
                (
                    [410 / n, maxX],
                    [110 / n, self._endy(410 / n, 110 / n, maxX, cdl)],
                    "-",
                ),
                ([0 / n, 35 / n], [200 / n, 200 / n], "-"),
                ([35 / n, self._endx(35 / n, 200 / n, maxY, ce)], [200 / n, maxY], "-"),
            ]

        colors = ["#196600", "#7FFF00", "#FF7B00", "#FF5700", "#FF0000"]

        _gridlabels = [
            (600, 600, "A", colors[0]),
            (360, 600, "B", colors[1]),
            (600, 355, "B", colors[1]),
            (165, 600, "C", colors[2]),
            (600, 215, "C", colors[2]),
            (600, 50, "D", colors[3]),
            (75, 600, "D", colors[3]),
            (5, 600, "E", colors[4]),
        ]

        # plot individual points
        if self.color_points == "auto":
            ax.scatter(
                self.reference,
                self.test,
                marker="o",
                alpha=0.6,
                c=[colors[i] for i in self._calc_error_zone()],
                s=8,
                **self.point_kws
            )
        else:
            ax.scatter(
                self.reference,
                self.test,
                marker="o",
                color=self.color_points,
                alpha=0.6,
                s=8,
                **self.point_kws
            )

        # plot grid lines
        if self.grid:
            for g in _gridlines:
                ax.plot(
                    np.array(g[0]),
                    np.array(g[1]),
                    g[2],
                    color=self.color_grid,
                    **self.grid_kws
                )

            if self.percentage:
                zones = [["A", "B", "C", "D", "E"][i] for i in self._calc_error_zone()]

                for label in _gridlabels:
                    ax.text(
                        label[0] / n,
                        label[1] / n,
                        label[2],
                        fontsize=12,
                        fontweight="bold",
                        color=label[3]
                        if self.color_gridlabels == "auto"
                        else self.color_gridlabels,
                    )
                    ax.text(
                        label[0] / n + (18 / n),
                        label[1] / n + (18 / n),
                        "{:.1f}".format((zones.count(label[2]) / len(zones)) * 100),
                        fontsize=9,
                        fontweight="bold",
                        color=label[3]
                        if self.color_gridlabels == "auto"
                        else self.color_gridlabels,
                    )

            else:
                for label in _gridlabels:
                    ax.text(
                        label[0] / n,
                        label[1] / n,
                        label[2],
                        fontsize=12,
                        fontweight="bold",
                        color=label[3]
                        if self.color_gridlabels == "auto"
                        else self.color_gridlabels,
                    )

        # limits and ticks
        _ticks = [
            70,
            100,
            150,
            180,
            240,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700,
            750,
            800,
            850,
            900,
            950,
            1000,
        ]

        ax.set_xticks([round(x / n, 1) for x in _ticks])
        ax.set_yticks([round(x / n, 1) for x in _ticks])
        ax.set_xlim(0, maxX)
        ax.set_ylim(0, maxY)

        # graph labels
        ax.set_ylabel(self.y_title)
        ax.set_xlabel(self.x_title)
        if self.graph_title is not None:
            ax.set_title(self.graph_title)


def parkes(
    type,
    reference,
    test,
    units,
    x_label=None,
    y_label=None,
    title=None,
    xlim=None,
    ylim=None,
    color_grid="#000000",
    color_gridlabels="auto",
    color_points="auto",
    grid=True,
    percentage=False,
    point_kws=None,
    grid_kws=None,
    square=False,
    ax=None,
):
    """Provide a glucose error grid analyses as designed by Parkes.

    This is an Axis-level function which will draw the Parke-error grid plot.
    onto the current active Axis object unless ``ax`` is provided.


    Parameters
    ----------
    type : int
        Parkes error grid differ for each type of diabetes. This should be either
        1 or 2 corresponding to the type of diabetes.
    reference, test : array, or list
        Glucose values obtained from the reference and predicted methods, preferably
        provided in a np.array.
    units : str
        The SI units which the glucose values are provided in.
        Options: 'mmol', 'mgdl' or 'mg/dl'.
    x_label : str, optional
        The label which is added to the X-axis. If None is provided, a standard
        label will be added.
    y_label : str, optional
        The label which is added to the Y-axis. If None is provided, a standard
        label will be added.
    title : str, optional
        Title of the Parkes-error grid plot. If None is provided, no title will be
        plotted.
    xlim : list, optional
        Minimum and maximum limits for X-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    ylim : list, optional
        Minimum and maximum limits for Y-axis. Should be provided as list or tuple.
        If not set, matplotlib will decide its own bounds.
    color_grid : str, optional
        Color of the Clarke error grid lines. Defaults to #000000 which represents
        the black color.
    color_gridlabels : str, optional
        Color of the grid labels (A, B, C, ..) that will be plotted.
        Defaults to 'auto' which colors the points according to their relative zones.
    color_points : str, optional
        Color of the individual differences that will be plotted. Defaults to 'auto'
        which colors the points according to their relative zones.
    grid : bool, optional
        Enable the grid lines of the Parkes error. Defaults to True.
    percentage : bool, optional
        If True, percentage of the zones will be depicted in the plot.
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be square-shaped.
    point_kws : dict of key, value mappings, optional
        Additional keyword arguments for `plt.scatter`.
    grid_kws : dict of key, value mappings, optional
        Additional keyword arguments for the grid with `plt.plot`.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the Parkes error grid plot.

    References
    ----------
    [parkes_2000] Parkes, J. L., Slatin S. L. et al.
                  Diabetes Care, vol. 23, no. 8, 2000, pp. 1143-1148.
    [pfutzner_2013] Pfutzner, A., Klonoff D. C., et al.
                    J Diabetes Sci Technol, vol. 7, no. 5, 2013, pp. 1275-1281.
    """

    plotter: _Parkes = _Parkes(
        type,
        reference,
        test,
        units,
        x_label,
        y_label,
        title,
        xlim,
        ylim,
        color_grid,
        color_gridlabels,
        color_points,
        grid,
        percentage,
        point_kws,
        grid_kws,
    )

    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()

    if square:
        ax.set_aspect("equal")

    plotter.plot(ax)

    return ax


def parkeszones(type, reference, test, units, numeric=False):
    """Provides the error zones as depicted by the
    Parkes error grid analysis for each point in the reference and test datasets.


    Parameters
    ----------
    type : int
        Parkes error grid differ for each type of diabetes. This should be either
        1 or 2 corresponding to the type of diabetes.
    reference, test : array, or list
        Glucose values obtained from the reference and predicted methods,
        preferably provided in a np.array.
    units : str
        The SI units which the glucose values are provided in.
        Options: 'mmol', 'mgdl' or 'mg/dl'.
    numeric : bool, optional
        If this is set to true, returns integers (0 to 4) instead of characters
        for each of the zones.

    Returns
    -------
    parkeszones : list
        Returns a list depicting the zones for each of the reference and test values.

    """

    # obtain zones from a Clarke reference object
    _zones = _Parkes(
        type,
        reference,
        test,
        units,
        None,
        None,
        None,
        None,
        None,
        True,
        False,
        "#000000",
        "auto",
        "auto",
        None,
        None,
    )._calc_error_zone()

    if numeric:
        return _zones
    else:
        labels = ["A", "B", "C", "D", "E"]
        return [labels[i] for i in _zones]
    
def parkes_error_zone_detailed(act, pred, diabetes_type):
    """
    This function outputs the Parkes Error Grid region (encoded as integer)
    for a combination of actual and predicted value
    for type 1 and type 2 diabetic patients

    Based on the article 'Technical Aspects of the Parkes Error Grid':
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3876371/
    """
    def above_line(x_1, y_1, x_2, y_2, strict=False):
        if x_1 == x_2:
            return False

        y_line = ((y_1 - y_2) * act + y_2 * x_1 - y_1 * x_2) / (x_1 - x_2)
        return pred > y_line if strict else pred >= y_line

    def below_line(x_1, y_1, x_2, y_2, strict=False):
        return not above_line(x_1, y_1, x_2, y_2, not strict)

    def parkes_type_1(act, pred):
        # Zone E
        if above_line(0, 150, 35, 155) and above_line(35, 155, 50, 550):
            return 7
        # Zone D - left upper
        if (pred > 100 and above_line(25, 100, 50, 125) and
                above_line(50, 125, 80, 215) and above_line(80, 215, 125, 550)):
            return 6
        # Zone D - right lower
        if (act > 250 and below_line(250, 40, 550, 150)):
            return 5
        # Zone C - left upper
        if (pred > 60 and above_line(30, 60, 50, 80) and
                above_line(50, 80, 70, 110) and above_line(70, 110, 260, 550)):
            return 4
        # Zone C - right lower
        if (act > 120 and below_line(120, 30, 260, 130) and below_line(260, 130, 550, 250)):
            return 3
        # Zone B - left upper
        if (pred > 50 and above_line(30, 50, 140, 170) and
                above_line(140, 170, 280, 380) and (act < 280 or above_line(280, 380, 430, 550))):
            return 2
        # Zone B - right lower
        if (act > 50 and below_line(50, 30, 170, 145) and
                below_line(170, 145, 385, 300) and (act < 385 or below_line(385, 300, 550, 450))):
            return 1
        # Zone A
        return 0

    def parkes_type_2(act, pred):
        # Zone E
        if (pred > 200 and above_line(35, 200, 50, 550)):
            return 7
        # Zone D - left upper
        if (pred > 80 and above_line(25, 80, 35, 90) and above_line(35, 90, 125, 550)):
            return 6
        # Zone D - right lower
        if (act > 250 and below_line(250, 40, 410, 110) and below_line(410, 110, 550, 160)):
            return 5
        # Zone C - left upper
        if (pred > 60 and above_line(30, 60, 280, 550)):
            return 4
        # Zone C - right lower
        if (below_line(90, 0, 260, 130) and below_line(260, 130, 550, 250)):
            return 3
        # Zone B - left upper
        if (pred > 50 and above_line(30, 50, 230, 330) and
                (act < 230 or above_line(230, 330, 440, 550))):
            return 2
        # Zone B - right lower
        if (act > 50 and below_line(50, 30, 90, 80) and below_line(90, 80, 330, 230) and
                (act < 330 or below_line(330, 230, 550, 450))):
            return 1
        # Zone A
        return 0

    if diabetes_type == 1:
        return parkes_type_1(act, pred)

    if diabetes_type == 2:
        return parkes_type_2(act, pred)

    raise Exception('Unsupported diabetes type')

def calculate_zone_percentages_PEG(pred_values, ref_values, diabetes_type):
    """
    Calculates the percentage of data points that fall into each Parkes Error Grid zone.

    :param ref_values: Array of actual glucose values.
    :param pred_values: Array of predicted glucose values.
    :param diabetes_type: Type of diabetes (1 or 2), which determines the zone boundaries.
    :return: A dictionary with percentages of data points in each zone.
    """
    
    # Count the number of data points in each zone
    zone_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    total_points = len(ref_values)

    for a, p in zip(ref_values, pred_values):
        zone = parkes_error_zone_detailed(a, p, diabetes_type)
        # 0: Zone A, 1, 2: Zone B, 3, 4: Zone C, 5, 6: Zone D, 7: Zone E
        # Map the zone number to the corresponding letter
        zone_mapping = {
            0: 'A',
            1: 'B',
            2: 'B',
            3: 'C',
            4: 'C',
            5: 'D',
            6: 'D',
            7: 'E'
        }

        # Function to get the letter for a given zone number
        def get_zone_letter(zone_number):
            return zone_mapping.get(zone_number, 'Unknown')
        
        zone = get_zone_letter(zone)
        if zone in zone_counts:
            zone_counts[zone] += 1

    # Calculate percentages
    zone_percentages = {zone: (count / total_points) * 100 for zone, count in zone_counts.items()}

    return zone_percentages

# def plot_parkes_error_grid(predicted, actual, diabetes_type, show=True, save_path=None):
#     """
#     Plots the Parkes Error Grid including zone lines and data points.
    
#     :param actual: Array of actual glucose values.
#     :param predicted: Array of predicted glucose values.
#     :param diabetes_type: Type of diabetes (1 or 2).
#     """
#     # Define the figure and axis
#     fig, ax = plt.subplots(figsize=(10, 8))

#     # Plot zone lines
#     if diabetes_type == 1:
#         # Define the lines for Type 1 Diabetes
#         ax.plot([0, 35], [150, 155], 'k-', lw=1)  # Zone E upper left
#         ax.plot([35, 50], [155, 550], 'k-', lw=1)  # Zone E lower right
#         ax.plot([25, 50], [100, 125], 'k-', lw=1)  # Zone D upper left (upper bound)
#         ax.plot([50, 80], [125, 215], 'k-', lw=1)  # Zone D upper left (middle bound)
#         ax.plot([80, 125], [215, 550], 'k-', lw=1)  # Zone D upper left (lower bound)
#         ax.plot([250, 550], [40, 150], 'k-', lw=1)  # Zone D lower right
#         ax.plot([30, 50], [60, 80], 'k-', lw=1)  # Zone C upper left (upper bound)
#         ax.plot([50, 70], [80, 110], 'k-', lw=1)  # Zone C upper left (middle bound)
#         ax.plot([70, 260], [110, 550], 'k-', lw=1)  # Zone C upper left (lower bound)
#         ax.plot([120, 260], [30, 130], 'k-', lw=1)  # Zone C lower right (upper bound)
#         ax.plot([260, 550], [130, 250], 'k-', lw=1)  # Zone C lower right (lower bound)
#         ax.plot([30, 140], [50, 170], 'k-', lw=1)  # Zone B upper left (upper bound)
#         ax.plot([140, 280], [170, 380], 'k-', lw=1)  # Zone B upper left (middle bound)
#         ax.plot([280, 430], [380, 550], 'k-', lw=1)  # Zone B upper left (lower bound)
#         ax.plot([50, 170], [30, 145], 'k-', lw=1)  # Zone B lower right (upper bound)
#         ax.plot([170, 385], [145, 300], 'k-', lw=1)  # Zone B lower right (middle bound)
#         ax.plot([385, 550], [300, 450], 'k-', lw=1)  # Zone B lower right (lower bound)

#     elif diabetes_type == 2:
#         # Define the lines for Type 2 Diabetes
#         ax.plot([35, 50], [200, 550], 'k-', lw=1)  # Zone E
#         ax.plot([25, 35], [80, 90], 'k-', lw=1)  # Zone D upper left (upper bound)
#         ax.plot([35, 125], [90, 550], 'k-', lw=1)  # Zone D upper left (lower bound)
#         ax.plot([250, 550], [40, 110], 'k-', lw=1)  # Zone D lower right
#         ax.plot([30, 280], [60, 550], 'k-', lw=1)  # Zone C upper left
#         ax.plot([90, 260], [0, 130], 'k-', lw=1)  # Zone C lower right
#         ax.plot([30, 230], [50, 330], 'k-', lw=1)  # Zone B upper left (upper bound)
#         ax.plot([230, 440], [330, 550], 'k-', lw=1)  # Zone B upper left (lower bound)
#         ax.plot([50, 90], [30, 80], 'k-', lw=1)  # Zone B lower right (upper bound)
#         ax.plot([90, 330], [80, 230], 'k-', lw=1)  # Zone B lower right (middle bound)
#         ax.plot([330, 550], [230, 450], 'k-', lw=1)  # Zone B lower right (lower bound)

#     # Scatter plot of actual vs predicted values
#     scatter = ax.scatter(actual, predicted, c=[parkes_error_zone_detailed(a, p, diabetes_type) for a, p in zip(actual, predicted)], cmap='coolwarm', edgecolor='k', s=50)
#     ax.plot([0, 600], [0, 600], 'k--', lw=1)  # 1:1 line

#     # Labels and title
#     ax.set_xlabel('Actual Glucose Value (mg/dL)')
#     ax.set_ylabel('Predicted Glucose Value (mg/dL)')
#     ax.set_title('Parkes Error Grid')
#     ax.set_xlim(0, 600)
#     ax.set_ylim(0, 600)
#     ax.set_aspect('equal')

#     # Create a legend
#     import matplotlib.colors as mcolors
#     norm = mcolors.Normalize(vmin=0, vmax=7)
#     sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
#     cbar.set_label('Parkes Error Grid Zone')

#     if show:
#         plt.show()
#     if save_path:
#         plt.savefig(save_path)

def plot_parkes_error_grid2(predicted, actual, diabetes_type, show=True, save_path=None):
    """
    Plots the Parkes Error Grid including zone lines and data points.
    
    :param actual: Array of actual glucose values.
    :param predicted: Array of predicted glucose values.
    :param diabetes_type: Type of diabetes (1 or 2).
    :param show: Whether to display the plot.
    :param save_path: Path to save the plot image.
    """
    # Define the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot zones using the pre-defined coordinates
    regionA_x = [0,50,50,170,385,550,550,430,280,140,30,0]
    regionA_y = [0,0,30,145,300,450,550,550,380,170,50,50]
    regionB_x = [0,120,120,260,550,550,260,70,50,30,0]
    regionB_y = [0,0,30,130,250,550,550,110,80,60,60]
    regionC_x = [0,250,250,550,550,125,80,50,25,0]
    regionC_y = [0,0,40,150,550,550,215,125,100,100]
    regionD_x = [0,550,550,50,35,0]
    regionD_y = [0,0,550,550,155,150]
    regionE_x = [0,0,550,550]
    regionE_y = [0,550,550,0]

    # Plot the regions with black lines only
    ax.plot(regionA_x, regionA_y, 'k-', lw=1)  # Region A
    ax.plot(regionB_x, regionB_y, 'k-', lw=1)  # Region B
    ax.plot(regionC_x, regionC_y, 'k-', lw=1)  # Region C
    ax.plot(regionD_x, regionD_y, 'k-', lw=1)  # Region D
    ax.plot(regionE_x, regionE_y, 'k-', lw=1)  # Region E

    # Scatter plot of actual vs predicted values with blue dots
    ax.scatter(actual, predicted, color='blue', edgecolor='k', s=50)

    # Plot the 1:1 line
    ax.plot([0, 600], [0, 600], 'k--', lw=1)

    # Labels and title
    ax.set_xlabel('Actual Glucose Value (mg/dL)')
    ax.set_ylabel('Predicted Glucose Value (mg/dL)')
    ax.set_title('Parkes Error Grid')
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 600)
    ax.set_aspect('equal')

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)



def plot_overlapped_PEG_SEG2(pred_values, ref_values, img, show=True, save_path=None):
    """
    Plots the Surveillance Error Grid (SEG) with the Parkes Error Grid overlay and zone labels.

    :param pred_values: Array of predicted glucose values.
    :param ref_values: Array of actual glucose values.
    :param img: Path to the image file of the SEG background.
    :param show: Whether to display the plot.
    :param save_path: Path to save the plot image.
    """
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Load and display the SEG image
    img = plt.imread(img)
    ax.imshow(img, extent=[0, 600, 0, 600])

    # Scatter plot of actual vs predicted values with color based on Parkes Error Grid zones
    scatter = ax.scatter(ref_values, pred_values, 
                         c=[parkes_error_zone_detailed(a, p, 1) for a, p in zip(ref_values, pred_values)], 
                         cmap='tap10', edgecolor='k', s=25)

    # Parkes Error Grid zones
    regionA_x = [0,50,50,170,385,550,550,430,280,140,30,0]
    regionA_y = [0,0,30,145,300,450,550,550,380,170,50,50]
    regionB_x = [0,120,120,260,550,550,260,70,50,30,0]
    regionB_y = [0,0,30,130,250,550,550,110,80,60,60]
    regionC_x = [0,250,250,550,550,125,80,50,25,0]
    regionC_y = [0,0,40,150,550,550,215,125,100,100]
    regionD_x = [0,550,550,50,35,0]
    regionD_y = [0,0,550,550,155,150]
    regionE_x = [0,0,550,550]
    regionE_y = [0,550,550,0]

    # Plot Parkes Error Grid zones on top of SEG
    ax.plot(regionA_x, regionA_y, 'k-', lw=1)  # Region A
    ax.plot(regionB_x, regionB_y, 'k-', lw=1)  # Region B
    ax.plot(regionC_x, regionC_y, 'k-', lw=1)  # Region C
    ax.plot(regionD_x, regionD_y, 'k-', lw=1)  # Region D
    ax.plot(regionE_x, regionE_y, 'k-', lw=1)  # Region E

    # Add labels for each region with corrected coordinates
    ax.text(500, 500, 'A', fontsize=20, ha='center', va='center', color='black')  # Adjusted position for Region A
    ax.text(500, 300, 'B', fontsize=20, ha='center', va='center', color='black')  # Adjusted position for Region B
    ax.text(300, 500, 'B', fontsize=20, ha='center', va='center', color='black')
    ax.text(500, 170, 'C', fontsize=20, ha='center', va='center', color='black')  # Adjusted position for Region C
    ax.text(170, 500, 'C', fontsize=20, ha='center', va='center', color='black')
    ax.text(500, 80, 'D', fontsize=20, ha='center', va='center', color='black')  # Adjusted position for Region D
    ax.text(80, 500, 'D', fontsize=20, ha='center', va='center', color='black')
    ax.text(20, 500, 'E', fontsize=20, ha='center', va='center', color='black')  # Adjusted position for Region E
    
    # Set labels with increased font size
    ax.set_xlabel("Measured Blood Glucose Values (mg/dL)", fontsize=18)  # Increase fontsize here
    ax.set_ylabel("Predicted Blood Glucose Values (mg/dL)", fontsize=18)  # Increase fontsize here

    # Display or save the plot
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
