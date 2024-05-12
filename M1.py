import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

np.int = np.int32
np.float = np.float64
np.bool = np.bool_

# Define the directory where your CSV files are located
directory = 'E:/ASU/Senior 2/2nd Term/Selected topics in Industrial Mechatronics/S_ind project/tool_data/'

# Read the CSV file to obtain the general data from 18 different experiments
train_df = pd.read_csv(directory + 'train.csv')
exp_numbers_with_dev_feed_rate = train_df['feedrate'].tolist()

# Define a list to store the data for each experiment
experiment_data = []

# Iterate through each experiment file
for i in range(1, 19):
    # Construct the file path for the current experiment
    file_name = f'experiment_{i:02d}.csv'
    file_path = directory + file_name
    
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Extract the required features
    features = df[[
        'X1_ActualPosition', 'X1_ActualVelocity', 'X1_ActualAcceleration', 'X1_CommandPosition',
        'X1_CommandVelocity', 'X1_CommandAcceleration', 'X1_CurrentFeedback', 'X1_DCBusVoltage',
        'X1_OutputCurrent', 'X1_OutputVoltage', 'X1_OutputPower', 'Y1_ActualPosition', 
        'Y1_ActualVelocity', 'Y1_ActualAcceleration', 'Y1_CommandPosition', 'Y1_CommandVelocity', 
        'Y1_CommandAcceleration', 'Y1_CurrentFeedback', 'Y1_DCBusVoltage', 'Y1_OutputCurrent', 
        'Y1_OutputVoltage', 'Y1_OutputPower', 'Z1_ActualPosition', 'Z1_ActualVelocity', 
        'Z1_ActualAcceleration', 'Z1_CommandPosition', 'Z1_CommandVelocity', 'Z1_CommandAcceleration',
        'Z1_CurrentFeedback', 'Z1_DCBusVoltage', 'Z1_OutputCurrent', 'Z1_OutputVoltage', 
        'S1_ActualPosition', 'S1_ActualVelocity', 'S1_ActualAcceleration', 'S1_CommandPosition', 
        'S1_CommandVelocity', 'S1_CommandAcceleration', 'S1_CurrentFeedback', 'S1_DCBusVoltage', 
        'S1_OutputCurrent', 'S1_OutputVoltage', 'S1_OutputPower', 'S1_SystemInertia',
        'M1_CURRENT_PROGRAM_NUMBER', 'M1_sequence_number', 'M1_CURRENT_FEEDRATE',
        'Machining_Process'
        
    ]]

    # Store the experiment number and features separately
    experiment_data.append({'experiment_number': i, 'features': features})

print('Data Imported')

#### PreProcessing Step ####

# Preprocessing function
def preprocess_data(df, exp_number, dev_feed_rate):

    s_shape_dimensions = (25, 35)
    s_shape_center = (140+s_shape_dimensions[0]/2, 72+s_shape_dimensions[0]/2)
    s_shape_left = s_shape_center[0] - s_shape_dimensions[0] / 2
    s_shape_right = s_shape_center[0] + s_shape_dimensions[0] / 2
    s_shape_bottom = s_shape_center[1] - s_shape_dimensions[1] / 2
    s_shape_top = s_shape_center[1] + s_shape_dimensions[1] / 2

    # Preprocessing 1: Filter out rows where Machining_Process is not in specified processes
    valid_processes = ['Layer 1 Up','Layer 2 Up','Layer 3 Up', 'Layer 1 Down', 'Layer 2 Down','Layer 3 Down']
    df = df[df['Machining_Process'].isin(valid_processes)]
    
    # Convert dev_feed_rate to a list if it's not already
    if not isinstance(dev_feed_rate, list):
        dev_feed_rate = [dev_feed_rate]
    
    # Preprocessing 2: Filter out rows where Machining_Process is not in specified processes
    dev_feed_rate = [val for val in dev_feed_rate if not pd.isna(val)]
    
    # Preprocessing 2: Filter the data based on the specified condition
    df = df[df['M1_CURRENT_FEEDRATE'].isin([int(val) for val in dev_feed_rate])]

    # Presprocessing 3: Function to eliminate deviated points from S shape
    df = df[(df['X1_ActualPosition'] >= s_shape_left) &
            (df['X1_ActualPosition'] <= s_shape_right) &
            (df['Y1_ActualPosition'] >= s_shape_bottom) &
            (df['Y1_ActualPosition'] <= s_shape_top)]
    
    return df

# Preprocess each experiment data
preprocessed_experiment_data = []

for experiment, feed_rate in zip(experiment_data, exp_numbers_with_dev_feed_rate):
    experiment_number = experiment['experiment_number']
    features = experiment['features']
    preprocessed_features = preprocess_data(features, experiment_number, feed_rate)
    preprocessed_experiment_data.append({'experiment_number': experiment_number, 'features': preprocessed_features})

print('Preprocessing Done')


#### Collect All Preprocessed Points in one Array ####

# Create an array of preprocessed data
preprocessed_array = []

# Iterate through each preprocessed experiment
for experiment in preprocessed_experiment_data:
    features = experiment['features']
    for _, row in features.iterrows():
        data_point = row[:-1].tolist()  # Exclude 'Machining_Process'
        feed_rate = train_df.loc[train_df['feedrate'] == row['M1_CURRENT_FEEDRATE'], 'feedrate'].values[0]
        clamp_pressure = train_df.loc[train_df['feedrate'] == row['M1_CURRENT_FEEDRATE'], 'clamp_pressure'].values[0]
        label = train_df.loc[train_df['feedrate'] == row['M1_CURRENT_FEEDRATE'], 'tool_condition'].values[0]
        label = np.where(label == 'worn', 1, 0)
        data_point.extend([feed_rate, clamp_pressure, label])
        preprocessed_array.append(data_point)

preprocessed_array = np.array(preprocessed_array)

print("Shape of preprocessed array:", preprocessed_array.shape)


#### Data Spliting ####
# Splitting preprocessed_array into X and Y
X = preprocessed_array[:, :-1]  # Features (all columns except the last one)
Y = preprocessed_array[:, -1]   # Labels (last column)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print('Data Spliting Done')
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


#### Feature Extraction by Applying Boruta Algorithm ####

# Define a RandomForestClassifier
forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced',random_state=42, n_estimators=100, max_depth=5)

# Apply Boruta
boruta_selector = BorutaPy(forest, n_estimators=10, verbose=2, random_state=42)

# Fit the Boruta selector
boruta_selector.fit(X_train, y_train)

# Check selected features
selected_features = [f for f, s in zip(range(X_train.shape[1]), boruta_selector.support_) if s]
print("Selected Features:", selected_features)

# Check ranking of features
print("Feature ranking:", boruta_selector.ranking_)

# Get important features
important_features = [f for f, r in zip(range(X_train.shape[1]), boruta_selector.ranking_) if r < 3]
print("Important Features:", important_features)


# Plot the feature importances
plt.figure(figsize=(14,6))

# Bar plot for feature ranking
plt.bar(range(X_train.shape[1]), boruta_selector.ranking_, color=np.where(boruta_selector.ranking_ < 3, 'green', 'red'))

# Highlight important features
for f in important_features:
    plt.bar(f, boruta_selector.ranking_[f], color='green')

plt.xlabel("Features")
plt.ylabel("Ranking")
plt.title("Feature Ranking (Green: Important, Red: Not Important)")
plt.xticks(range(X_train.shape[1]), ['F{}'.format(i+1) for i in range(X_train.shape[1])], rotation=90, ha='right')
plt.tight_layout()
plt.show()


print('Feature Extraction Done')

# Traing and testing set after feature extraction
X_train_selc = boruta_selector.transform(X_train)
X_test_selc = boruta_selector.transform(X_test)

##### Model Develpment ############
models = {
    "Model 1": LogisticRegression(max_iter=1000, C=1.0),
    "Model 2": LogisticRegression(max_iter=1000, C=0.5),
    "Model 3": LogisticRegression(max_iter=1000, C=0.1),
    "Model 4": LogisticRegression(max_iter=1000, C=0.01),
    "Model 5": LogisticRegression(max_iter=1000, C=0.001)
}

# AIC values and residuals
aic_values = {}
residuals = {}
auc_values = {}

for name, model in models.items():
    model.fit(X_train_selc, y_train)
    y_prob = model.predict_proba(X_test_selc)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_val = auc(fpr, tpr)
    auc_values[name] = auc_val
    aic_values[name] = 2 * len(important_features) - 2 * np.log(np.maximum(1e-10, auc_val))
    residuals[name] = np.sum(np.square(y_test - model.predict(X_test_selc)))

# Select the best initial model
best_initial_model = min(aic_values, key=lambda x: (aic_values[x], residuals[x]))
print("Best Initial Model by AIC values:", best_initial_model)

# K-fold Cross-Validation

# Define the number of splits for K-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Dictionary to store the p-values for each model
p_values = {}

for name, model in models.items():
    p_values[name] = []
    for train_index, val_index in kf.split(X_train_selc):
        X_train_kf, X_val_kf = X_train_selc[train_index], X_train_selc[val_index]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]
        model.fit(X_train_kf, y_train_kf)
        # Get the p-value for the current validation set
        y_prob = model.predict_proba(X_val_kf)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val_kf, y_prob)
        p_value = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        # Store the p-value
        p_values[name].append(p_value)

# Print p-values
for name, p_value in p_values.items():
    print(f"P-values for {name}: {p_value}")

# Select the model based on improvement in p-value
best_model_kfold = best_initial_model
for name, p_value in p_values.items():
    if name == best_initial_model:
        continue
    if np.mean(p_value) < np.mean(p_values[best_model_kfold]):
        best_model_kfold = name

print("Best Model after K-fold Cross-Validation:", best_model_kfold)

# Dictionary to store deviance residual values for each model and fold
deviance_residuals = {model_name: [] for model_name in models.keys()}

# Perform K-fold cross-validation
for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train_selc)):
    X_train_fold, X_val_fold = X_train_selc[train_index], X_train_selc[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Train each model on the current fold
    for model_name, model in models.items():
        model.fit(X_train_fold, y_train_fold)
        # Calculate deviance residuals for the current fold and model
        deviance_residual = np.sum(np.square(y_val_fold - model.predict(X_val_fold)))
        deviance_residuals[model_name].append(deviance_residual)

# Plot the deviance residuals for each fold and model
plt.figure(figsize=(12, 8))
for model_name, residuals_list in deviance_residuals.items():
    plt.plot(range(1, k + 1), residuals_list, marker='o', label=model_name)

plt.title('Deviance Residuals for Each Fold and Model')
plt.xlabel('Fold')
plt.ylabel('Deviance Residual')
plt.xticks(range(1, k + 1))
plt.legend()
plt.grid(True)
plt.show()

print("Best Model Selected")

# Finding Optimal Threshold

# Train the best model using the whole training dataset
best_model = models[best_model_kfold]
best_model.fit(X_train_selc, y_train)
y_proba = best_model.predict_proba(X_train_selc)[:, 1]

# Find the optimal threshold
fpr, tpr, thresholds = roc_curve(y_train, y_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Evaluate using k-fold cross-validation
conf_matrices = []
for train_index, test_index in kf.split(X_train_selc):
    X_train_fold, X_test_fold = X_train_selc[train_index], X_train_selc[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    best_model.fit(X_train_fold, y_train_fold)
    y_proba_fold = best_model.predict_proba(X_test_fold)[:, 1]
    conf_matrices.append(confusion_matrix(y_test_fold, y_proba_fold > optimal_threshold))

# Calculate TBN, Sensitivity, Specificity for each fold
TBNs = []
sensitivities = []
specificities = []

for conf_matrix in conf_matrices:
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]

    TBN = TN / (FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    TBNs.append(TBN)
    sensitivities.append(sensitivity)
    specificities.append(specificity)


# Plot TBN of all training set for the chosen model
plt.figure(figsize=(10, 6))
for i, conf_matrix in enumerate(conf_matrices):
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]

    TBN = TN / (FP + FN)
    plt.plot(i+1, TBN, 'ro')
    plt.text(i+1, TBN, f'{TBN:.2f}', fontsize=9, va='bottom')

best_k = np.argmax(TBNs) + 1
best_TBN = max(TBNs)
plt.plot(best_k, best_TBN, 'b^', markersize=10)
plt.text(best_k, best_TBN, f'{best_TBN:.2f}', fontsize=9, va='bottom')

plt.title('TBN of all training set for the chosen model')
plt.xlabel('Fold')
plt.ylabel('TBN')
plt.xticks(range(1, k + 1))
plt.grid(True)
plt.show()

print("Best Training Set (K):", best_k)
print("Optimum Threshold:", optimal_threshold)


# Visualizing Model Performance

# Train the best model using the whole training dataset
best_model = models[best_model_kfold]
best_model.fit(X_train_selc, y_train)

# Predict probabilities for both train and test sets
y_proba_train = best_model.predict_proba(X_train_selc)[:, 1]
y_proba_test = best_model.predict_proba(X_test_selc)[:, 1]

# Prepare color for Tool Condition
colors_train = np.where(y_train == 0, 'blue', 'red')
colors_test = np.where(y_test == 0, 'blue', 'red')

# Prepare marker for Train and Test sets
marker_train = 'o'
marker_test = '^'

# Plot the predicted probabilities
plt.figure(figsize=(12, 8))
scatter_train = plt.scatter(range(len(y_train)), y_proba_train, color=colors_train, label='Train', marker=marker_train)
scatter_test = plt.scatter(range(len(y_train), len(y_train) + len(y_test)), y_proba_test, color=colors_test, label='Test', marker=marker_test)

plt.axhline(y=optimal_threshold, color='gray', linestyle='--', label='Optimal Threshold')

# Legend for color
blue_patch = mpatches.Patch(color='blue', label='Tool Condition 0')
red_patch = mpatches.Patch(color='red', label='Tool Condition 1')

# Creating the legend
legend1 = plt.legend(handles=[blue_patch, red_patch], loc="upper right")

# Adding the marker color to the legend
legend2 = plt.legend([scatter_train, scatter_test], ['Train', 'Test'], loc="upper left")
plt.gca().add_artist(legend1)

plt.xlabel('Index')
plt.ylabel('Predicted Probability')
plt.title('Model Performance by Predicted Probability')
plt.grid(True)
plt.show()

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_proba_test > optimal_threshold)
print("Accuracy of the model:", accuracy)