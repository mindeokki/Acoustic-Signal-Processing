import numpy as np
import soundfile as sf



def do_amplify(y, db):
    amplify_ratio = 10 ** (db / 20)
    y = y * amplify_ratio
    return y


def do_cut(y, sr, start_end_list):
    start_time = start_end_list[0]
    end_time = start_end_list[1]
    y = y[start_time * sr:end_time * sr]
    return y


def do_overlap(y1, y2):
    overlapped_sig = y1 + y2
    return overlapped_sig


def make_noisy(y_sig, y_ns, snr):  # original signal's rms remain same
    rms_sig = np.sqrt(np.mean(y_sig ** 2))
    rms_ns = np.sqrt(np.mean(y_ns ** 2))
    y_ns = y_ns * (rms_sig / rms_ns)
    amplify_ratio = 10 ** (-(snr / 20))
    y_ns = y_ns * amplify_ratio
    noisy_sig = y_sig + y_ns
    return noisy_sig


def make_noisy_gaussian(y_sig, snr):
    rms_sig = np.sqrt(np.mean(y_sig ** 2))
    snr_linear = 10 ** (snr / 20)
    rms_ns = rms_sig / snr_linear
    gauss = rms_ns * np.random.normal(size=y_sig.shape)
    noisy_signal = y_sig + gauss
    return noisy_signal


def do_cross_corr(y1, y2):
    cr_corr = np.correlate(y1, y2, mode='full')
    cr_corr /= np.max(cr_corr)
    return cr_corr


def do_auto_corr(y):
    auto_corr = np.correlate(y, y, mode='full')
    auto_corr /= np.max(auto_corr)
    return auto_corr


def do_gcc_phat(y1, y2, sr, max_time=None):
    n = y1.shape[0] + y2.shape[0]

    Y1 = np.fft.rfft(y1, n)  # n : odd, shape : (n+1)/2
    Y2 = np.fft.rfft(y2, n)  # n : even, shape : (n/2)+1

    R = Y1 * np.conj(Y2)
    R /= np.abs(R)

    cc = np.fft.ifft(R, n=n)
    max_shift = int(n / 2)
    if max_time:
        max_shift = np.minimum(int(sr * max_time), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    estimated_delay = shift / sr  # (-) : y2 is more delayed, (+) : y1 is more delayed
    return estimated_delay




if __name__ == '__main__':
    print('hello')

