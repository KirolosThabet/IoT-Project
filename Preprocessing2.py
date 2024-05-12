import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the directory where your CSV files are located
directory = 'E:/ASU/Senior 2/2nd Term/Selected topics in Industrial Mechatronics/S_ind project/tool_data/'

# Read the CSV file containing experiment numbers and feed rates
feed_rate_df = pd.read_csv(directory + 'train.csv')

# Extract the experiment numbers and feed rates from the DataFrame
exp_numbers_with_dev_feed_rate = feed_rate_df.iloc[:, [0, 2]].values.tolist()

# Function to plot histograms for a group of experiments
def plot_histograms(start_exp, end_exp):
    # Iterate through each experiment file in the range specified
    for i in range(start_exp, end_exp + 1):
        # Construct the file path for the current experiment
        file_name = f'experiment_{i:02d}.csv'
        file_path = directory + file_name

        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path)

        # Preprocessing 1: Filter out rows where Machining_Process is not in specified processes
        valid_processes = ['Layer 1 Up', 'Layer 2 Up', 'Layer 3 Up', 'Layer 1 Down', 'Layer 2 Down', 'Layer 3 Down']
        df_1 = df[df['Machining_Process'].isin(valid_processes)]

        # Plot histogram for M1_CURRENT_FEEDRATE before Preprocessing
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(df['M1_CURRENT_FEEDRATE'], bins=30, color='blue', alpha=0.7, label='Before Preprocessing')
        plt.axvline(x=exp_numbers_with_dev_feed_rate[i - 1][1], color='r', linestyle='--', label='True Feed Rate')
        plt.xlabel('M1_CURRENT_FEEDRATE')
        plt.ylabel('Frequency')
        plt.title(f'Experiment {i} - Before Preprocessing')
        plt.grid(True)
        plt.legend()

        # Preprocessing 2:Filter the data based on the specified condition
        true_feed_rate = exp_numbers_with_dev_feed_rate[i - 1][1]
        filtered_data = df_1[df_1['M1_CURRENT_FEEDRATE'] == true_feed_rate]

        # Plot histogram for M1_CURRENT_FEEDRATE after Preprocessing
        plt.subplot(1, 2, 2)
        plt.hist(filtered_data['M1_CURRENT_FEEDRATE'], bins=30, color='green', alpha=0.7, label='After Preprocessing')
        plt.axvline(x=exp_numbers_with_dev_feed_rate[i - 1][1], color='r', linestyle='--', label='True Feed Rate')
        plt.xlabel('M1_CURRENT_FEEDRATE')
        plt.ylabel('Frequency')
        plt.title(f'Experiment {i} - After Preprocessing')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

# Plot histograms for experiments grouped by six
for start_exp in range(1, 19, 6):
    end_exp = min(start_exp + 5, 18)  # Ensure the last group does not exceed the maximum experiment number
    plot_histograms(start_exp, end_exp)
