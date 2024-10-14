import librosa
import matplotlib.pyplot as plt
import numpy as np
import pywt
import soundfile as sf
import torchaudio

## spectrogram
def do_spectrogram(y=None, sr=None, file_path=None, n_fft=2048, hop_length=512, plot_fig=True, save_fig=True):
    if file_path is not None:
        y, sr = sf.read(file_path)
        filename = file_path.split('/')[-1]
    elif y is None or sr is None:
        raise ValueError("Either 'filepath' or both 'y' and 'sr' must be provided.")

    spec = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    print(spec.shape)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    if plot_fig:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='log', n_fft=n_fft, hop_length=hop_length,)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram (Librosa)')
        plt.tight_layout()
        if save_fig:
            if file_path is not None:
                plt.savefig(f'Spectrogram_librosa_{filename}_plot.pdf', format='pdf', dpi=300)
            else:
                plt.savefig(f'Spectrogram_librosa_plot.pdf', format='pdf', dpi=300)
        plt.show()

    return spec_db


## melspectrogram
def do_melspectrogram(y=None, sr=None, file_path=None, n_fft=2048, hop_length=512, n_mels=128, fmin=50, fmax=8000, plot_fig=True,
                      save_fig=True):
    if file_path is not None:
        y, sr = sf.read(file_path)
        filename = file_path.split('/')[-1]
    elif y is None or sr is None:
        raise ValueError("Either 'filepath' or both 'y' and 'sr' must be provided.")

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if plot_fig:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram (Librosa)')
        plt.tight_layout()
        if save_fig:
            if file_path is not None:
                plt.savefig(f'Melspectrogram_librosa_{filename}_plot.pdf', format='pdf', dpi=300)
            else:
                plt.savefig(f'Melspectrogram_librosa_plot.pdf', format='pdf', dpi=300)
        plt.show()

    return mel_spec_db

def do_melspectrogram_torch(y=None, sr=None, file_path=None, n_fft=2048, hop_length=512, n_mels=128, fmin=50, fmax=8000, plot_fig=True,
                      save_fig=True):
    if file_path is not None:
        y, sr = torchaudio.load(file_path)
        print(sr)
        filename = file_path.split('/')[-1]
    elif y is None or sr is None:
        raise ValueError("Either 'filepath' or both 'y' and 'sr' must be provided.")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, f_min=fmin, f_max=fmax)
    mel_spec = mel_spectrogram(y)

    # Convert to log scale (dB)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    if plot_fig:
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec_db[0].detach().numpy(), aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram (Torchaudio)')
        plt.tight_layout()
        if save_fig:
            if file_path is not None:
                plt.savefig(f'Melspectrogram_torch_{filename}_plot.pdf', format='pdf', dpi=300)
            else:
                plt.savefig(f'Melspectrogram_torch_plot.pdf', format='pdf', dpi=300)
        plt.show()

    return mel_spec_db


## MFCC
def do_mfcc(y=None, sr=None, file_path=None, n_fft=2048, hop_length=512, n_mels=128, fmin=50, fmax=8000, n_mfcc=20, plot_fig=True, save_fig=True):
    if file_path is not None:
        y, sr = torchaudio.load(file_path)
        print(sr)
        filename = file_path.split('/')[-1]
    elif y is None or sr is None:
        raise ValueError("Either 'filepath' or both 'y' and 'sr' must be provided.")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=128, fmin=fmin, fmax=fmax)
    if plot_fig:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, sr=sr, x_axis='time', n_fft=n_fft, hop_length=hop_length,
                                 fmin=fmin, fmax=fmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC (Librosa)')
        plt.tight_layout()
        if save_fig:
            if file_path is not None:
                plt.savefig(f'MFCC_librosa_{filename}_plot.pdf', format='pdf', dpi=300)
            else:
                plt.savefig(f'MFCC_librosa_plot.pdf', format='pdf', dpi=300)
        plt.show()

    return mfcc


## scalogram (wavelet)
def do_scalogram(y=None, sr=None, file_path=None, plot_fig=True, save_fig=True):
    if file_path is not None:
        y, sr = sf.read(file_path)
        filename = file_path.split('/')[-1]
    elif y is None or sr is None:
        raise ValueError("Either 'filepath' or both 'y' and 'sr' must be provided.")

    # Define the wavelet and scales
    wavelet = 'cmor2.5-1.0'  # Change the wavelet
    scales = np.arange(1, 128)  # Scale (frequency bins)

    # Compute the CWT
    cwtmatr, freqs = pywt.cwt(y, scales, wavelet, sampling_period=1 / sr)
    if plot_fig:
        cwtmatr = np.abs(cwtmatr)
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(np.arange(cwtmatr.shape[1]), freqs, cwtmatr)
        plt.colorbar(label='Magnitude')
        plt.title('Scalogram (Wavelet Transform)')
        plt.xlabel('Time [s]')
        plt.ylabel('Scale')
        if save_fig:
            if file_path is not None:
                plt.savefig(f'Scalogram_{filename}_plot.pdf', format='pdf', dpi=300)
            else:
                plt.savefig(f'Scalogram_plot.pdf', format='pdf', dpi=300)
        plt.show()

    return cwtmatr



if __name__ == '__main__':
    y, sr = sf.read('BD.wav')
    do_melspectrogram(y=y, sr=sr)
    # do_melspectrogram(file_path='BD.wav')
    # do_melspectrogram_torch(file_path='BD.wav')
    # do_scalogram(y=y, sr=sr)
    # do_scalogram(file_path='BD.wav')
    # do_mfcc(y, sr)
    print(sr)
    print(y.shape)
    do_spectrogram(y=y, sr=sr)
    # data_path = '2024-06-27_1stSWBD_3m_mea1.mat'
    # f = scipy.io.loadmat(data_path)
    # signals = np.array(f['signal'])
    # y = signals[:, 0]
    # print(signals.shape)
    # sr = 48000
    # cut_off = 6000
    # import soundfile as sf
    # # sf.write('./Arc_discharge.wav', signals[:, 0], samplerate=sr)
    #
    # from plot.waveshow import waveplot
    # waveplot(signals[:, 0], sr)
    # from temporal_tools.temporal_tools import do_amplify
    # y = do_amplify(y, 60)
    # sf.write('./Arc_discharge.wav', y, samplerate=sr)
    #
    # y_highpass = highpass_filter(y, cut_off, sr)
    # sf.write('./Arc_highpass.wav', y_highpass, samplerate=sr)
    # waveplot(y_highpass, sr)
    #
    # do_melspectrogram(y, sr)
    # do_melspectrogram(y_highpass, sr)





