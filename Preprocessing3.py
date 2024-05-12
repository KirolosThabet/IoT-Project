import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the directory where your CSV files are located
directory = 'E:/ASU/Senior 2/2nd Term/Selected topics in Industrial Mechatronics/S_ind project/tool_data/'

# Read the CSV file containing experiment numbers and feed rates
feed_rate_df = pd.read_csv(directory + 'train.csv')

# Extract the experiment numbers and feed rates from the DataFrame
exp_numbers_with_dev_feed_rate = feed_rate_df.iloc[:, [0, 2]].values.tolist()

# Function to preprocess the data
def preprocess_data(df):
    # Preprocessing 1: Filter out rows where Machining_Process is not in specified processes
    valid_processes = ['Layer 1 Up','Layer 2 Up','Layer 3 Up', 'Layer 1 Down', 'Layer 2 Down','Layer 3 Down']
    df = df[df['Machining_Process'].isin(valid_processes)]
    return df

# Presprocessing 3: Function to eliminate deviated points from S shape
def eliminate_deviated_points(df, s_shape_center, s_shape_dimensions):
    s_shape_left = s_shape_center[0] - s_shape_dimensions[0] / 2
    s_shape_right = s_shape_center[0] + s_shape_dimensions[0] / 2
    s_shape_bottom = s_shape_center[1] - s_shape_dimensions[1] / 2
    s_shape_top = s_shape_center[1] + s_shape_dimensions[1] / 2
    
    # Filter out data points outside the bounding box around the S shape
    df = df[(df['X1_ActualPosition'] >= s_shape_left) &
            (df['X1_ActualPosition'] <= s_shape_right) &
            (df['Y1_ActualPosition'] >= s_shape_bottom) &
            (df['Y1_ActualPosition'] <= s_shape_top)]
    
    return df

# Iterate through each experiment file in the range specified
for i in range(1, 19):
    # Construct the file path for the current experiment
    file_name = f'experiment_{i:02d}.csv'
    file_path = directory + file_name

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Preprocessing 1: Filter out rows where Machining_Process is not in specified processes
    df_1 = preprocess_data(df)
    
    # Preprocessing 2: Filter the data based on the specified condition
    true_feed_rate = exp_numbers_with_dev_feed_rate[i - 1][1]
    filtered_data = df_1[df_1['M1_CURRENT_FEEDRATE'] == true_feed_rate]
    

    # Plot before eliminating deviated points
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(df['X1_ActualPosition'], df['Y1_ActualPosition'], label='Before Presprocessing')
    plt.title(f'Experiment {i} - Before Presprocessing')
    plt.xlabel('X1_ActualPosition (mm)')
    plt.ylabel('Y1_ActualPosition (mm)')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio

    # Set fixed scale for the plot
    plt.xlim(100, 250)  # Adjust the limits as needed
    plt.ylim(50, 200)   # Adjust the limits as needed

    # Presprocessing 3: Eliminate deviated points from S shape
    s_shape_dimensions = (25, 35)
    s_shape_center = (140+s_shape_dimensions[0]/2, 75+s_shape_dimensions[0]/2)
    filtered_data = eliminate_deviated_points(filtered_data, s_shape_center, s_shape_dimensions)
    
    # Plot after eliminating deviated points
    plt.subplot(1, 2, 2)
    plt.scatter(filtered_data['X1_ActualPosition'], filtered_data['Y1_ActualPosition'], label='After Presprocessing')
    plt.title(f'Experiment {i} - After Presprocessing')
    plt.xlabel('X1_ActualPosition (mm)')
    plt.ylabel('Y1_ActualPosition (mm)')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio

    # Set fixed scale for the plot
    plt.xlim(100, 250)  # Adjust the limits as needed
    plt.ylim(50, 200)   # Adjust the limits as needed

    plt.tight_layout()
    plt.show()

