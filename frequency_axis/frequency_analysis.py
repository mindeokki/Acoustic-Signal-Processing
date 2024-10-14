import numpy as np
import soundfile as sf
from scipy.fft import fft, fftfreq
from scipy.signal import welch, coherence, csd

from plot.general import *


def do_fft(y=None, sr=None, file_path=None, plot_fig=True, save_fig=True):
    if file_path is not None:
        y, sr = sf.read(file_path)
        filename = file_path.split('/')[-1]
    elif y is None or sr is None:
        raise ValueError("Either 'filepath' or both 'y' and 'sr' must be provided.")

    N = y.shape[0]  # Signal length
    yf = fft(y)
    print(
        f"Check Parseval's theorem, Sum in time domain {np.sum(y ** 2)}, Sum in freq domain {np.sum(np.abs(yf[:N // 2]) ** 2 * 2 / N)}")
    xf = fftfreq(N, 1 / sr)[:N // 2]
    if plot_fig:
        plt.semilogy(xf, 2.0 / N * np.abs(yf[:N // 2]))  # 2/N (2 from symmetry, N from parseval's theorem)
        set_fig_name(f'FFT plot', 'Frequency [Hz]', 'Amplitude')
        plt.grid()
        if save_fig:
            if file_path is not None:
                plt.savefig(f'FFT_{filename}_plot.pdf', format='pdf', dpi=300)
            else:
                plt.savefig(f'FFT_plot.pdf', format='pdf', dpi=300)
        plt.show()
    return xf, yf


def do_welch(y=None, sr=None, file_path=None, plot_fig=True, save_fig=True):           # Auto spectral density
    if file_path is not None:
        y, sr = sf.read(file_path)
        filename = file_path.split('/')[-1]
    elif y is None or sr is None:
        raise ValueError("Either 'filepath' or both 'y' and 'sr' must be provided.")

    ref_pressure = 20e-6
    f, Pxx = welch(y, sr, nperseg=1024)
    Pxx_db = 10 * np.log10(Pxx / ref_pressure**2)

    if plot_fig:
        plt.semilogx(f, Pxx_db)
        set_fig_name(f'Welch plot', 'Frequency [Hz]', r'Power/Frequency [dB SPL]')
        plt.grid()
        if save_fig:
            if file_path is not None:
                plt.savefig(f'Welch_{filename}_plot.pdf', format='pdf', dpi=300)
            else:
                plt.savefig(f'Welch_plot.pdf', format='pdf', dpi=300)
        plt.show()
    return f, Pxx


def do_csd(y1, y2, sr, plot_fig=True, save_fig=True):
    f, Pxy = csd(y1, y2, sr, nperseg=1024)
    if plot_fig:
        plt.semilogy(f, Pxy)
        set_fig_name(f'CSD plot', 'Frequency [Hz]', r'CSD [$V^2$/Hz]')
        plt.grid()
        if save_fig:
            plt.savefig(f'CSD plot.pdf', format='pdf', dpi=300)
        plt.show()
    return f, Pxy


def do_coherence(y1, y2, sr, plot_fig=True, save_fig=True):
    f, Cxy = coherence(y1, y2, sr)
    if plot_fig:
        plt.semilogy(f, Cxy)
        set_fig_name(f'Coherence plot', 'Frequency [Hz]', r'Coherence')
        plt.grid()
        if save_fig:
            plt.savefig(f'Coherence plot.pdf', format='pdf', dpi=300)
        plt.show()
    return f, Cxy


if __name__ == '__main__':
    y, sr = sf.read('../data_augmentation/BD.wav')
    # y2, sr2 = sf.read('L1.wav')
    do_fft(y, sr)
    do_welch(y, sr)
    # do_csd(y, y2, sr)
    # do_coherence(y, y2, sr)
