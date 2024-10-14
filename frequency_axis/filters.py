from scipy.signal import butter, filtfilt

def butter_highpass(cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


# Apply the filter
def highpass_filter(y, sr, cutoff, order=5):
    b, a = butter_highpass(cutoff, sr, order=order)
    y = filtfilt(b, a, y)
    return y


if __name__ == '__main__':
    y_highpass = highpass_filter(y, sr, cutoff)