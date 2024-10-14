import numpy as np
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from scipy.io import loadmat


def inspect_matfile(file_path):
    mat_contents = loadmat(file_path)

    print("Keys in the .mat file:")
    for key in mat_contents:
        print(key)

    # Inspect the contents
    for key in mat_contents:
        print(f"\nKey: {key}")
        content = mat_contents[key]
        if isinstance(content, np.ndarray):
            print(f" - Type: {type(content)}")
            print(f" - Shape: {content.shape}")
            print(f" - Data Type: {content.dtype}")
            # Display additional details for arrays
            if content.size > 0 and content.ndim > 0:
                print(f" - Sample Data: {content.flat[:min(content.size, 10)]}")
        else:
            print(f" - Value: {content}")


def read_mp3(file_path):
    audio = AudioSegment.from_mp3(file_path)
    samples = np.array(audio.get_array_of_samples())

    # If stereo, take one channel
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples[:, 0]

    return samples, audio.frame_rate


def write_wav(y, sr, file_path):
    sf.write(file_path, y, sr)


def listen_wav(y, sr):
    sd.play(y, sr)
    sd.wait()  # Wait until the file is done playing


if __name__ == "__main__":
    mat_file_path = './2024-06-27_1stSWBD_3m_mea1.mat'
    inspect_matfile(mat_file_path)
    y, sr = sf.read('./L1.wav')
    listen_wav(y, sr)
