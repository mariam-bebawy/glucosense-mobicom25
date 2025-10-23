import numpy as np

# Multilayer Perceptron Parameters
rgr_mlp_params = dict(
    hidden_layer_sizes=[(x, y) for x in range(100, 600, 50) for y in range(1, 5)],
    activation=['logistic', 'tanh', 'relu'],
    solver=['sgd', 'adam'],
    alpha=np.linspace(0.0001, 0.001, num=10).tolist(),
    batch_size=[32, 64, 100, 150],
    learning_rate=['constant', 'invscaling', 'adaptive'],
    max_iter=[20, 30, 40, 50, 60]
)

# Random Forest Parameters
rgr_random_forest_params = dict(
    n_estimators=[int(i) for i in range(50, 400, 10)],
    max_features=["sqrt", "log2"],
    bootstrap=[True, False],
    max_depth=list(np.arange(1, 50)),
    min_samples_split=np.linspace(0.1, 1.0, 9, endpoint=True).tolist()
)

# Linear Regression Parameters
rgr_linear_regression_params = dict(
    fit_intercept=[True, False],
    positive=[True, False]
)

#Support Vector Machine Parameters
rgr_svm_params = dict(
    kernel = ["poly","rbf","sigmoid",'sigmoid'],
    C = np.linspace(0.1,50,100),
    gamma = [10**x for x in np.arange(-5,5,dtype = float)],
    max_iter = np.arange(50,200)
)

# Partial Least Squares Regression Parameters
rgr_pls_params = dict(
    n_components=[int(i) for i in np.linspace(1, 20, num=20)],  # Number of PLS components
    scale=[True, False],  # Whether to scale the data or not
    max_iter=np.arange(100, 1000, 100).tolist(),  # Maximum number of iterations
    tol=np.linspace(1e-6, 1e-2, num=10).tolist()  # Tolerance for convergence
)

# XGBoost Parameters
rgr_xgb_params = dict(
    n_estimators=[50, 100, 150, 180, 200, 220],  # Number of boosting rounds
    learning_rate=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.2],  # Learning rate (shrinkage)
    max_depth=[3, 5, 7, 9, 12, 15],  # Maximum tree depth for base learners
    min_child_weight=[1, 3, 5, 7, 9],  # Minimum sum of instance weight (hessian) needed in a child
    subsample=[0.6, 0.7, 0.8, 1.0],  # Subsample ratio of the training instances
    colsample_bytree=[0.6, 0.7, 0.8, 1.0],  # Subsample ratio of columns when constructing each tree
    gamma=[0, 0.1, 0.2, 0.3],  # Minimum loss reduction required to make a further partition on a leaf node
    reg_alpha=[0, 0.1, 0.3, 0.5, 0.7, 1.0],  # L1 regularization term on weights
    reg_lambda=[0, 0.1, 0.3, 0.5, 0.7, 1.0],  # L2 regularization term on weights
    scale_pos_weight=[1, 3, 5],  # Balancing of positive and negative weights
    objective=['reg:squarederror'],  # Objective function for regression
    booster=['gbtree', 'gblinear', 'dart']  # Which booster to use
)

# Ridge Regression Parameters
rgr_rr_params = dict(
    alpha=[0.1, 0.3, 0.5, 1.0, 5.0, 10.0],  # Regularization strength
    max_iter=[1000, 1500, 2000],  # Maximum number of iterations
    tol=[0.001, 0.0001, 0.00001]  # Tolerance for convergence
)


# Lasso Regression Parameters
rgr_las_params = dict(
    alpha=[0.1, 0.3, 0.5, 1.0, 5.0, 10.0],  # Regularization strength
    max_iter=[1000, 1500, 2000],  # Maximum number of iterations
    tol=[0.001, 0.0001, 0.00001],  # Tolerance for convergence
    selection=['cyclic', 'random']  # Method used to select features
)


# Elastic Net Parameters
rgr_en_params = dict(
    alpha=[0.1, 0.3, 0.5, 1.0, 5.0, 10.0],  # Regularization strength
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],  # Mixing parameter
    max_iter=[1000, 1500, 2000],  # Maximum number of iterations
    tol=[0.001, 0.0001, 0.00001],  # Tolerance for convergence
    selection=['cyclic', 'random']  # Method used to select features
)
