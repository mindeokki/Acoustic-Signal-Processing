import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from numpy.fft import fft2, fftshift, fftfreq

from time_frequency_axis.time_frequency_analysis import do_spectrogram


'''
Github homepage : https://github.com/theunissenlab/soundsig/blob/master/soundsig/sound.py
'''

def mtfft(spectrogram, df, dt, Log=False):
    """
        Compute the 2d modulation power and phase for a given time frequency slice.
        return temporal_freq,spectral_freq,mps_pow,mps_phase
    """

    # take the 2D FFT and center it
    smps = fft2(spectrogram)
    smps = fftshift(smps)

    # compute the log amplitude
    mps_pow = np.abs(smps) ** 2
    if Log:
        mps_pow = 10 * np.log10(mps_pow)

    # compute the phase
    mps_phase = np.angle(smps)

    # compute the axes
    nf = mps_pow.shape[0]
    nt = mps_pow.shape[1]

    spectral_freq = fftshift(fftfreq(nf, d=df[1] - df[0]))
    temporal_freq = fftshift(fftfreq(nt, d=dt[1] - dt[0]))
    """
    nb = sdata.shape[1]
    dwf = np.zeros(nb)
    for ib in range(int(np.ceil((nb+1)/2.0))+1):
        posindx = ib
        negindx = nb-ib+2
        print 'ib=%d, posindx=%d, negindx=%d'% (ib, posindx, negindx )
        dwf[ib]= (ib-1)*(1.0/(df*nb))
        if ib > 1:
            dwf[negindx] =- dwf[ib]

    nt = sdata.shape[0]
    dwt = np.zeros(nt)
    for it in range(0, int(np.ceil((nt+1)/2.0))+1):
        posindx = it
        negindx = nt-it+2
        print 'it=%d, posindx=%d, negindx=%d' % (it, posindx, negindx)
        dwt[it] = (it-1)*(1.0/(nt*dt))
        if it > 1 :
            dwt[negindx] = -dwt[it]

    spectral_freq = dwf
    temporal_freq = dwt
    """

    return spectral_freq, temporal_freq, mps_pow, mps_phase


def mps(spectrogram, df, dt, window=None, Norm=True):
    """
    Calculates the modulation power spectrum using overlapp and add method with a gaussian window of length window in s
    Assumes that spectrogram is in dB.  df and dt are the axis of spectrogram.
    """

    # Debugging paramenter
    debugPlot = False

    # Resolution of spectrogram in DB
    dbRES = 50

    # Check the size of the spectrogram vs dt
    nt = dt.size
    nf = df.size
    if spectrogram.shape[1] != nt and spectrogram.shape[0] != nf:
        print('Error in mps. Expected  %d bands in frequency and %d points in time' % (nf, nt))
        print('Spectrogram had shape %d, %d' % spectrogram.shape)
        return 0, 0, 0

    # Z-score the flattened spectrogram is Norm is True
    sdata = deepcopy(spectrogram)

    if Norm:
        maxdata = sdata.max()
        mindata = maxdata - dbRES
        sdata[sdata < mindata] = mindata
        sdata -= sdata.mean()
        sdata /= sdata.std()

    if window == None:
        window = dt[-1] / 10.0

    # Find the number of spectrogram points in the gaussian window
    if dt[-1] < window:
        print('Warning in mps: Requested MPS window size is greater than spectrogram temporal extent.')
        print('mps will be calculate with windows of %d points or %s s' % (nt - 1, dt[-1]))
        nWindow = nt - 1
    else:
        nWindow = np.where(dt >= window)[0][0]
    if nWindow % 2 == 0:
        nWindow += 1  # Make it odd size so that we have a symmetric window

    # if nWindow < 64:
    #    print('Error in mps: window size %d pts (%.3f s) is two small for reasonable estimates' % (nWindow, window))
    #    return np.asarray([]), np.asarray([]), np.asarray([])

    # Generate the Gaussian window
    gt, wg = gaussian_window(nWindow, 6)
    tShift = int(gt[-1] / 3)
    nchunks = 0

    # Pad the spectrogram with zeros.
    minSdata = sdata.min()
    sdataZeros = np.ones((sdata.shape[0], int((nWindow - 1) / 2))) * minSdata
    sdata = np.concatenate((sdataZeros, sdata, sdataZeros), axis=1)

    if debugPlot:
        plt.figure(1)
        plt.clf()
        plt.subplot()
        plt.imshow(sdata, origin='lower')
        plt.title('Scaled and Padded Spectrogram')
        plt.show()
        plt.pause(1)

    for tmid in range(tShift, nt, tShift):

        # t mid is in the original coordinates while tstart and tend
        # are shifted to deal with the zero padding.
        tstart = tmid - (nWindow - 1) // 2 - 1
        tstart += (nWindow - 1) // 2
        if tstart < 0:
            print('Error in mps. tstart negative')
            break;

        tend = tmid + (nWindow - 1) // 2
        tend += (nWindow - 1) // 2
        if tend > sdata.shape[1]:
            print('Error in mps. tend too large')
            break
        nchunks += 1

        # Multiply the spectrogram by the window
        wSpect = deepcopy(sdata[:, tstart:tend])

        # Debugging code
        if debugPlot:
            plt.figure(nchunks + 1)
            plt.clf()
            plt.subplot(121)
            plt.imshow(wSpect, origin='lower')
            plt.title('%d Middle (%d %d)' % (tmid, tstart, tend))

        for fInd in range(nf):
            wSpect[fInd, :] = wSpect[fInd, :] * wg

        # Debugging code
        if debugPlot:
            plt.figure(nchunks + 1)
            plt.subplot(122)
            plt.imshow(wSpect, origin='lower')
            plt.title('After')
            plt.show()
            plt.pause(1)
            input("Press Enter to continue...")

        # Get the 2d FFT
        wf, wt, mps_pow, mps_phase = mtfft(wSpect, df, dt[0:tend - tstart])
        if nchunks == 1:
            mps_powAvg = mps_pow
        else:
            mps_powAvg += mps_pow

    mps_powAvg /= nchunks

    return wf, wt, mps_powAvg


def plot_mps(spectral_freq, temporal_freq, amp, phase=None):
    plt.figure()

    # plot the amplitude
    if phase is not None:
        plt.subplot(2, 1, 1)

    # ex = (spectral_freq.min(), spectral_freq.max(), temporal_freq.min(), temporal_freq.max())
    ex = (temporal_freq.min(), temporal_freq.max(), spectral_freq.min() * 1e3, spectral_freq.max() * 1e3)
    plt.imshow(amp, interpolation='nearest', aspect='auto', origin='lower', cmap='jet', extent=ex)
    plt.ylabel('Spectral Frequency (Cycles/KHz)')
    plt.xlabel('Temporal Frequency (Hz)')
    plt.colorbar()
    plt.ylim((0, spectral_freq.max() * 1e3))
    plt.title('Power')

    # plot the phase
    if phase is not None:
        plt.subplot(2, 1, 2)
        plt.imshow(phase, interpolation='nearest', aspect='auto', origin='lower', cmap='jet', extent=ex)
        plt.ylabel('Spectral Frequency (Cycles/KHz)')
        plt.xlabel('Temporal Frequency (Hz)')
        plt.ylim((0, spectral_freq.max() * 1e3))
        plt.title('Phase')
        plt.colorbar()


if __name__ == '__main__':
    y, sr = sf.read('./BD.wav')
    sep = 5
    part = int(len(y) / sep)
    do_spectrogram(y=y, sr=sr, plot_fig=True, save_fig=False)
    for i in range(sep):
        spectrogram = do_spectrogram(y=y[i*part : (i+1)*part], sr=sr, plot_fig=True, save_fig=False)
        df = np.linspace(0, 44100, spectrogram.shape[0])
        dt = np.linspace(0, 10/sep, spectrogram.shape[1])
        print(df)
        print(dt)
        spectral_freq, temporal_freq, mps_pow, mps_phase = mtfft(spectrogram, dt=dt, df=df, Log=True)
        print(spectral_freq)
        print(temporal_freq)
        plot_mps(spectral_freq, temporal_freq, mps_pow, mps_phase)
        plt.tight_layout()
        plt.show()
