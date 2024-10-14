import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import random

def freq_masking(tf_feature, F=40, num_masks=1):
    tf_cloned = tf_feature.clone()
    mask_value = tf_cloned.mean().item()

    b, f, t = tf_feature.shape
    for batch in range(b):
        for _ in range(num_masks):
            f_mask = random.randrange(10, F)
            f0 = random.randrange(0, f - f_mask)
            tf_cloned[batch, f0:f0 + f_mask, :] = mask_value

    return tf_cloned

def time_masking(tf_feature, T=50, num_masks=1):
    tf_cloned = tf_feature.clone()
    mask_value = tf_cloned.mean().item()

    b, f, t = tf_feature.shape
    for batch in range(b):
        for _ in range(num_masks):
            t_mask = random.randrange(0, T)
            t0 = random.randrange(0, t - t_mask)
            tf_cloned[batch, :, t0:t0 + t_mask] = mask_value

    return tf_cloned


def time_warping(tf_feature, W=40):
    tf_cloned = tf_feature.clone()
    b, f, t = tf_cloned.shape

    if t - W <= W:
        return tf_cloned
    for b_idx in range(b):
        center = random.randrange(W, t - W)
        warped = random.randrange(center - W, center + W) + 1
        left = torch.empty([b, f, warped])
        right = torch.empty([b, f, t - warped])
        for f_idx in range(f):
            # plt.imshow(F.interpolate(tf_cloned[b_idx, f_idx, :center].unsqueeze(0).unsqueeze(0).unsqueeze(0), size=warped, mode='bicubic', align_corners=False)[0, 0, :, :].detach().numpy())
            # plt.show()
            print(F.interpolate(tf_cloned[b_idx, f_idx, :center].unsqueeze(0).unsqueeze(0).unsqueeze(0), size=warped, mode='bicubic', align_corners=False)[0, 0, :, 0].shape)
            left[b_idx, f_idx, :] = F.interpolate(tf_cloned[b_idx, f_idx, :center].unsqueeze(0).unsqueeze(0).unsqueeze(0), size=warped, mode='bicubic', align_corners=False)[0, 0, 0, :]
            right[b_idx, f_idx, :] = F.interpolate(tf_cloned[b_idx, f_idx, center:].unsqueeze(0).unsqueeze(0).unsqueeze(0), size=t - warped, mode='bicubic', align_corners=False)[0, 0, 0, :]
        tf_cloned[b_idx, :, :] = torch.cat([left, right], dim=2)
    return tf_cloned


def time_shift(tf_feature, samples):
    if isinstance(tf_feature, np.ndarray):
        tf_cloned = np.copy(tf_feature)
        tf_cloned = np.roll(tf_cloned, samples, axis=-1)

    elif isinstance(tf_feature, torch.Tensor):
        tf_cloned = tf_feature.clone()
        tf_cloned = torch.roll(tf_cloned, samples, dims=-1)

    return tf_cloned


def filt_augment(tf_feature, db_range=[-6, 6], n_band=[3, 6], min_bw=6, filter_type="linear"):
    if not isinstance(filter_type, str):
        if torch.rand(1).item() < filter_type:
            filter_type = "step"
            n_band = [2, 5]
            min_bw = 4
        else:
            filter_type = "linear"
            n_band = [3, 6]
            min_bw = 6

    b, f, _ = tf_feature.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()   # [low, high)
    if n_freq_band > 1:
        while f - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1
        band_bndry_freqs = torch.sort(torch.randint(0, f - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([f])))

        if filter_type == "step":
            band_factors = torch.rand((b, n_freq_band)).to(tf_feature) * (db_range[1] - db_range[0]) + db_range[0]
            band_factors = 10 ** (band_factors / 20)

            freq_filt = torch.ones((b, f, 1)).to(tf_feature)
            for i in range(n_freq_band):
                freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

        elif filter_type == "linear":
            band_factors = torch.rand((b, n_freq_band + 1)).to(tf_feature) * (db_range[1] - db_range[0]) + db_range[0]
            freq_filt = torch.ones((b, f, 1)).to(tf_feature)
            for i in range(n_freq_band):
                for j in range(b):
                    freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                        torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                       band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
            freq_filt = 10 ** (freq_filt / 20)
        return tf_feature * freq_filt

    else:
        return tf_feature



def spec_augment(tf_feature, time_warp=True, freq_mask=True, time_mask=True, W=5, F=30, T=40, num_freq_masks=1, num_time_masks=2):
    if time_warp:
        tf_feature = time_warping(tf_feature, W=W)

    if freq_mask:
        tf_feature = freq_masking(tf_feature, F=F, num_masks=num_freq_masks)

    if time_mask:
        tf_feature = time_masking(tf_feature, T=T, num_masks=num_time_masks)

    return tf_feature


def mixup(tf_feature1, tf_feature2, target1=0, target2=0, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    b, f, t = tf_feature1.shape
    index = torch.randperm(b)

    mixed_data = lam * tf_feature1 + (1 - lam) * tf_feature2
    mixed_targets = lam * target1 + (1 - lam) * target2

    return mixed_data, mixed_targets

from time_frequency_axis.time_frequency_analysis import do_melspectrogram
import matplotlib.pyplot as plt

if __name__ == '__main__':
    y, sr = sf.read('Arc_discharge.wav')
    mel = do_melspectrogram(y=y, sr=sr, hop_length=512, plot_fig=True, save_fig=False)
    y1, sr1 = sf.read('L1.wav')
    y1 = y1[:y.shape[0]]
    mel2 = do_melspectrogram(y=y1, sr=sr1, hop_length=512, plot_fig=True, save_fig=False)
    mel2 = torch.tensor(mel2).unsqueeze(0)
    mel = torch.tensor(mel)
    mel = mel.unsqueeze(0)
    print(mel.shape)
    time_masked = time_masking(mel)
    freq_masked = freq_masking(mel)
    time_shifted = time_shift(mel, 15)
    time_warped = time_warping(mel, 30)
    spec_augmented = spec_augment(mel, True, True, True)
    filt_augmented = filt_augment(mel)
    mixed, _ = mixup(mel, mel2, alpha=0.8)

    color = 'magma'
    fig, ax = plt.subplots(6, 1)
    ax[0].imshow(time_masked.squeeze().numpy(), origin='lower', cmap=color)
    ax[0].set_xticks([], [])
    ax[0].set_yticks([], [])
    # plt.show()
    ax[1].imshow(freq_masked.squeeze().numpy(), origin='lower', cmap=color)
    ax[1].set_xticks([], [])
    ax[1].set_yticks([], [])
    # plt.show()
    ax[2].imshow(time_shifted.squeeze().numpy(), origin='lower', cmap=color)
    ax[2].set_xticks([], [])
    ax[2].set_yticks([], [])
    # plt.show()
    ax[3].imshow(time_warped.squeeze().numpy(), origin='lower', cmap=color)
    ax[3].set_xticks([], [])
    ax[3].set_yticks([], [])
    # plt.show()
    ax[4].imshow(spec_augmented.squeeze().numpy(), origin='lower', cmap=color)
    ax[4].set_xticks([], [])
    ax[4].set_yticks([], [])
    # plt.show()
    ax[5].imshow(filt_augmented.squeeze().numpy(), origin='lower', cmap=color)
    ax[5].set_xticks([], [])
    ax[5].set_yticks([], [])
    plt.savefig('data_aug.pdf', format='pdf', dpi=300)
    plt.show()

    fig, ax = plt.subplots(3, 1)
    ax[0].imshow(mel.squeeze().numpy(), origin='lower', cmap=color)
    ax[0].set_xticks([], [])
    ax[0].set_yticks([], [])
    ax[1].imshow(mel2.squeeze().numpy(), origin='lower', cmap=color)
    ax[1].set_xticks([], [])
    ax[1].set_yticks([], [])
    ax[2].imshow(mixed.squeeze().numpy(), origin='lower', cmap=color)
    ax[2].set_xticks([], [])
    ax[2].set_yticks([], [])
    plt.savefig('mixed.pdf', format='pdf', dpi=300)
    plt.show()