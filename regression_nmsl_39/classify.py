import time
import argparse
from mobicom23_mobispectral.regression.hsi_dataset import TrainDataset, TestDataset
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description="Spectral Classification Toolbox")
parser.add_argument('--data_root', type=str, default='../datasets/dataset_skin/regression/')
parser.add_argument('--fruit', type=str, default='kiwi')
opt = parser.parse_args()

def fit_model(model, X_train, X_test, y_train, y_test, model_name):
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("\n", model_name, ":")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Time taken:", time.time() - start)

def plot_losses(model, filename):
        plt.plot(model.loss_curve_)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.show()

def main():
        train_data = TrainDataset(data_root)
        test_data = TestDataset(data_root)
        #train_data = DatasetFromDirectory(data_root,"", fruit)
        print("Total Samples:", len(train_data))
        #data = DataLoader(dataset=train_data)

        X, X_test, y, y_test = [],[],[],[]

        for label, sig in train_data:
                X.append(sig)
                #plt.plot(sig)
                #plt.show()
                y.append(label)

        X = numpy.asarray(X)
        y = numpy.asarray(y)
        print("X, y",X.shape, y.shape)

        #X_test = X[890:900,:]
        #y_test = y[890:900]

        for label, sig in test_data:
                X_test.append(sig)
                y_test.append(label)

        X_test = numpy.asarray(X_test)
        y_test = numpy.asarray(y_test)

        print("X_test, y_test",X_test.shape, y_test.shape)

        #X = X[1:290,:]
        #y = y[1:290]

        print("X, y",X.shape, y.shape)

        #kf = StratifiedKFold(n_splits=4, random_state = 0, shuffle=True)

        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X_test = scaler.transform(X_test)
        #mlp = MLPClassifier(hidden_layer_sizes=(200,150,100), max_iter=300, activation='relu', solver='adam', alpha=0.0001)
        #tree = DecisionTreeRegressor()
        tree = GradientBoostingRegressor(random_state=0)
        tree = tree.fit(X,y)
        y_pred = tree.predict(X_test)
        y_test = numpy.squeeze(y_test)

        err = numpy.abs(numpy.subtract(y_pred,y_test))
        #print(err)
        err = err/y_test
        #print(err)
        print(err.mean())
        print(err.std())
        print(y_test[1:10])
        print(y_pred[1:10])
        plt.scatter(y_test, y_pred)
        plt.show()
        #print()
        #plt.scatter(y, y_pred)
        #plt.show()
        #pipeline = Pipeline([('scaler', scaler), ('mlp', mlp)])
        #for k, (train_index, val_index) in enumerate(kf.split(X,y)):
        #    print("Fold:",k)
        #    tree.fit(X[train_index], y[train_index])
        #    pred_y = tree.predict(X[val_index])
        #    plt.figure()
        #    plt.scatter(y[valid_index], pred_y)
        #    plt.show()
            #fit_model(pipeline, X[train_index], X[val_index], y[train_index], y[val_index], "MLP" + fruit + str(k))
            #plot_losses(mlp, "MLP")
            #pickle.dump(pipeline, open(os.path.join("Models", "MLP_" + fruit + "_k" + str(k) + ".pkl"), "wb"))



if __name__ == "__main__":
        data_root = opt.data_root
        fruit = opt.fruit
        main()
