import numpy as np
from math import ceil
from scipy.signal import hilbert
import matplotlib.pyplot as plt

class STRF_mod:
    def __init__(self, t, f, omega_t_list, omega_f_list, gamma_order=3, gamma_bandwidth=0.5):
        self.t = t
        self.f = f
        self.omega_t_list = omega_t_list    # rate
        self.omega_f_list = omega_f_list    # scale
        self.gamma_order = gamma_order
        self.gamma_bandwidth = gamma_bandwidth

    def pair_STRF(self, omega_t, omega_f):
        ot = omega_t * self.t
        of = omega_f * self.f

        ht = omega_t * ot ** (self.gamma_order - 1) * np.exp(-2 * np.pi * self.gamma_bandwidth * ot) * np.sin(2 * np.pi * ot)
        hf = omega_f * np.exp(-(2 * np.pi * of) ** 2 / 2) * np.cos(2 * np.pi * of)

        ht_a = hilbert(ht)
        hf_a = hilbert(hf)

        self.strf_u = ht_a.reshape(1, len(t)) * hf_a.reshape(len(f), 1)
        self.strf_d = np.conj(ht_a).reshape(1, len(t)) * hf_a.reshape(len(f), 1)

        return [self.strf_u.real, self.strf_d.real]

    def pair_STRF_plot(self):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.strf_u.real, extent=[self.t[0], self.t[-1], self.f[0], self.f[-1]], cmap='viridis')
        f_label = ['0 oct']
        f_ticks = [-0]
        for f_idx in range(1, oct + 1):
            f_label = [f'{-f_idx} oct'] + f_label + [f'{f_idx} oct']
            f_ticks = [-f_idx] + f_ticks + [f_idx]
        ax[0].set_yticks(f_ticks, f_label)
        ax[0].set_ylabel('Frequency')
        ax[0].set_title('Upward STRF')
        t_label = []
        t_ticks = []
        for t_idx in np.arange(0, tau_max + 0.5, 0.5):
            t_label.append(t_idx)
            t_ticks.append(t_idx)
        ax[0].set_xticks(t_ticks, t_label)
        ax[0].set_xlabel('Time (sec)')
        ax[1].imshow(self.strf_d.real, extent=[self.t[0], self.t[-1], self.f[0], self.f[-1]], cmap='viridis')
        f_label = ['0 oct']
        f_ticks = [-0]
        for f_idx in range(1, oct + 1):
            f_label = [f'{-f_idx} oct'] + f_label + [f'{f_idx} oct']
            f_ticks = [-f_idx] + f_ticks + [f_idx]
        ax[1].set_yticks(f_ticks, f_label)
        ax[1].set_ylabel('Frequency')
        ax[1].set_title('Downward STRF')
        t_label = []
        t_ticks = []
        for t_idx in np.arange(0, tau_max + 0.5, 0.5):
            t_label.append(t_idx)
            t_ticks.append(t_idx)
        ax[1].set_xticks(t_ticks, t_label)
        ax[1].set_xlabel('Time (sec)')
        plt.show()

    def multi_STRF(self, upward=True):
        STRF_num = len(self.omega_f_list) * len(self.omega_t_list)
        for tt, omega_t in enumerate(self.omega_t_list):
            for ff, omega_f in enumerate(self.omega_f_list):
                strf_u, strf_d = self.pair_STRF(omega_t, omega_f)
                if upward:
                    if (tt == 0 and ff == 0):
                        self.strf_tot = strf_u.reshape(-1, strf_u.shape[0], strf_u.shape[1])
                    else:
                        self.strf_tot = np.concatenate([self.strf_tot, strf_u.reshape(-1, strf_u.shape[0], strf_u.shape[1])], axis=0)
                else:
                    if (tt == 0 and ff == 0):
                        self.strf_tot = strf_d.reshape(-1, strf_u.shape[0], strf_u.shape[1])
                    else:
                        self.strf_tot = np.concatenate([self.strf_tot, strf_d.reshape(-1, strf_d.shape[0], strf_d.shape[1])], axis=0)

        return self.strf_tot

    def multi_STRF_plot(self):
        fig, ax = plt.subplots(len(omega_f_list), len(omega_t_list), figsize=(20, 10))
        for tt, omega_t in enumerate(self.omega_t_list):
            for ff, omega_f in enumerate(self.omega_f_list):
                print(tt, ff)
                idx = tt * len(self.omega_f_list) + ff
                print(idx)
                ax[ff, tt].imshow(self.strf_tot[idx], extent=[self.t[0], self.t[-1], self.f[0], self.f[-1]])
                ax[ff, tt].set_yticks([],[])
                ax[ff, tt].set_xticks([],[])
        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Spectrogram specification
    hop_length = 256
    sr = 25600
    tau_max = 1          # in second
    mel_bins = 128
    t = np.linspace(0, tau_max, ceil(tau_max * sr / hop_length))
    # t_c = t[ceil(tau_max * sr / hop_length / 2)]
    oct = 1
    bin_per_oct = 16 # 128 bin for 8 oct
    f = np.linspace(-oct, oct, 2 * oct * bin_per_oct)
    print(f)
    # f = np.arange(mel_bins)
    # f_c = f[ceil(mel_bins / 2)]

    omega_t_list = [0.5, 1, 2, 4, 6, 8, 10, 12]     # Hz
    omega_f_list = [0.5, 1, 2, 4]     # cyc/oct
    STRF_mod = STRF_mod(t=t, f=f, omega_t_list=omega_t_list, omega_f_list=omega_f_list)
    # strf1, strf2 = STRF_mod.pair_STRF()
    # STRF_mod.pair_STRF_plot()
    STRF_mod.multi_STRF()
    STRF_mod.multi_STRF_plot()

