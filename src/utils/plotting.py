import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(path, output_path, name='Accuracy'):
    metrics = pd.read_pickle(path)
    plt.figure(figsize=(6, 5))
    plt.plot(np.arange(metrics['Round']), metrics['Accuracy'][:,1], label='FedAVG')
    plt.ylabel('Test Accuracy', fontsize=16)
    plt.xlabel('Communication Rounds', fontsize=16)
    plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
    plt.grid()
    plt.xlim([0, metrics['Round']])
    plt.title(name, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f'{output_path}/plot.pdf', dpi=1000, bbox_inches='tight')
    # plt.show()
