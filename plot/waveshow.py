import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def waveplot(y=None, sr=None, file_path=None, plot_fig=True, save_fig=False):
    if file_path is not None:
        y, sr = sf.read(file_path)
        filename = file_path.split('/')[-1]
    elif y is None or sr is None:
        raise ValueError("Either 'filepath' or both 'y' and 'sr' must be provided.")

    if plot_fig:
        plt.figure(figsize=(15, 10))
        line = plt.plot(y, linewidth=1.2, color='#9467bd')
        plt.setp(line, linewidth=5)
        plt.xlabel('Time (sec)', fontsize=14)
        plt.ylabel('Amplitude', fontsize=14)
        xlabel = [i for i in range(int(y.shape[0] / sr + 1)) if i % 5 == 0]
        xticks = [i * sr for i in xlabel]
        plt.xticks(xticks, xlabel)
        plt.xlim([0, int(y.shape[0])])
        # plt.tight_layout()
        # plt.grid()
        if save_fig:
            plt.savefig(f'{file_path.split(".")[0]}.svg', format='svg', dpi=300)
        plt.show()


if __name__ == '__main__':
    file_path = 'BD.wav'
    waveplot(file_path=file_path)
