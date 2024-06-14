import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(input_path, metric='Accuracy', output_path=None):
    """
    Plots the specified metric with respect to epoch from a pickle file or all pickle files in a directory.
    
    Parameters:
        input_path (str): Path to the pickle file or directory containing pickle files.
        metric (str): The metric to plot against epoch (default is 'accuracy').
        output_path (str): Path to save the output plot image (default is None, which shows the plot).
    """
    # Function to plot the data
    def plot_data(df, label):
        plt.plot(df['Round'], df[metric], label=label)
        plt.xlabel('Round')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs Round')
        plt.legend()

    # Check if input_path is a directory or a file
    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.endswith('.pkl'):
                file_path = os.path.join(input_path, file)
                with open(file_path, 'rb') as f:
                    df = pickle.load(f)
                    truncated_name = os.path.splitext(file)[0][:10]  # Truncate file name to 10 characters
                    plot_data(df, truncated_name)
    else:
        with open(input_path, 'rb') as f:
            df = pickle.load(f)
            truncated_name = os.path.splitext(os.path.basename(input_path))[0][:10]  # Truncate file name to 10 characters
            plot_data(df, truncated_name)

    if output_path:
        plt.savefig(output_path)
        print(f'Plot saved to {output_path}')
    else:
        plt.show()
    plt.clf()  # Clear the current figure

__all__ = ['plot_metric']