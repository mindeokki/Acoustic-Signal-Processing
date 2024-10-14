import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
import mtrf
from mtrf.model import load_sample_data, TRF
import soundfile as sf
import mne

def do_TRF(s, r, sr, tau_min, tau_max):
    # s : stimulus, r : response, sr : sampling rate, tau_min : minimum delay (in ms), tau_max : maximum delay (in ms)
    # Generalized reverse correlation
    # Paper from mTRF
    T = len(s)
    tau_min, tau_max = int(tau_min / 1000 * sr), int(tau_max / 1000 * sr)
    tau_window = tau_max - tau_min
    S = np.zeros([T, tau_window])

    for i in range(T):
        for j in range(i+tau_max):
            col_idx = i+tau_max-j-1
            if not (col_idx < 0 or col_idx >= tau_window or j >= T):
                # print(j, col_idx)
                S[j, col_idx] = s[i]
    S = np.flip(S, axis=1)
    w = inv(S.T @ S) @ S.T @ r

    return w

def do_TRF_convolve(s, w, sr, tau_min, tau_max):    # tau_min, tau_max (in ms)
    tau_min, tau_max = int(tau_min / 1000 * sr), int(tau_max / 1000 * sr)
    r_recon = np.convolve(s, w, mode='full')
    r_recon = r_recon[-tau_min:-tau_min + len(s)]
    return r_recon

# def do_STRF(s, r, sr, tau_min, tau_max):


if __name__ == '__main__':
    # r, r_sr = sf.read('L1.wav')
    # s, s_sr = sf.read('L1.wav')
    # r = r[22050:22050+4410]
    # s = s[22050:22050+4410]
    # plt.plot(s)
    # plt.show()
    # T = len(s)
    # print(r.shape)
    # R = np.fft.rfft(r)  # This is a fft with only positive frequency (shape : n/2 + 1)
    # S = np.fft.rfft(s)
    # S_conj = np.conj(S)
    #
    # tau_min, tau_max = [-10, 50]  # in ms
    # tau_min, tau_max = int(tau_min / 1000 * s_sr), int(tau_max / 1000 * s_sr)
    # # print(tau_min, tau_max)
    # # tau_window = tau_max - tau_min
    # # S = np.zeros([T, tau_window])
    # # for i in range(T):
    # #     for j in range(i+tau_max):
    # #         col_idx = i+tau_max-j-1
    # #         if not (col_idx < 0 or col_idx >= tau_window or j >= T):
    # #             # print(j, col_idx)
    # #             S[j, col_idx] = s[i]
    # #
    # # S = np.flip(S, axis=1)
    # #
    # # w = inv(S.T @ S) @ S.T @ r
    # # print(w.shape)
    # # np.save('./w.npy', w)
    # w = np.load('./w.npy')
    # print(s.shape)
    # print(w.shape)
    # r_recon = np.convolve(s, w, mode='full')
    # print(r_recon.shape)
    # print(tau_max)
    # r_recon = r_recon[-tau_min:-tau_min+len(s)]
    # print(np.argmax(r_recon), np.argmax(r))
    # # plt.imshow(S, cmap='hot')
    # # plt.colorbar()
    # # plt.show()
    # plt.plot(r, label='r')
    # # plt.show()
    # plt.plot(r_recon, label='r_recon')
    # plt.legend()
    # plt.show()
    stimulus, response, fs = load_sample_data(n_segments=10, normalize=True) # stimulus given as list [10(epoch), 1536(time, 128Hz, 12seconds), 16(freq)]
    # response given as list [10(epoch), 1536(time), 128(EEG channels)]
    print(stimulus[0].shape)
    print(response[0].shape)

    # mtrf
    fwd_trf = TRF(direction=1)
    tmin, tmax= 0, 0.4
    regularization = 1000
    fwd_trf.train(stimulus, response, fs, tmin, tmax, regularization)

    prediction, r_fwd = fwd_trf.predict(stimulus, response)
    print(f"correlation between actual and predicted response: {r_fwd.round(3)}")

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(2, figsize=(15, 10))
    fwd_trf.plot(feature=13, axes=ax[0], show=False)   # Weight shape : [53, 128], feature is frequency band
    fwd_trf.plot(channel='gfp', axes=ax[1], kind='image', show=False)
    plt.tight_layout()
    plt.show()

