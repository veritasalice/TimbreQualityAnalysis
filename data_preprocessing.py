import librosa
import numpy as np
import pandas as pd
import scipy
import warnings
warnings.filterwarnings('ignore')


input_length = 76799  # =96000*0.8 #16000 * 30
sr = 96000
n_mels = 80


def pre_process_audio_melspec(audio, sample_rate=96000):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=2048, hop_length=960, n_mels=n_mels)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    # 80*80
    return mel_db.T


def load_audio_file(file_path, input_length=input_length):
    try:
        data, _ = librosa.load(str(file_path), sr=sr, mono=True)
    except ZeroDivisionError:
        data = []

    if len(data) > input_length:

        max_offset = len(data) - input_length

        offset = np.random.randint(max_offset)

        data = data[offset: (input_length + offset)]

    else:
        if input_length > len(data):
            max_offset = input_length - len(data)

            offset = np.random.randint(max_offset)
        else:
            offset = 0

        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    return data


def LPC_estimation(audio, p):
    a = librosa.lpc(audio, p)
    b = np.hstack([[0], -1 * a[1:]])
    s_hat = scipy.signal.lfilter(b, [1], audio)
    return s_hat


def save_features(filename):
    file_path = Path(args.wav_path+filename)
    audio = load_audio_file(file_path)
    # 1. pre-emphasis
    audio = scipy.signal.lfilter([1.0, -0.97], 1, audio)
    # 2. calculate features:

    # change pitch
    # Shift up by a major third
    pitch_up = librosa.effects.pitch_shift(audio, sr, n_steps=4)
    # Shift down by a major third
    pitch_down = librosa.effects.pitch_shift(audio, sr, n_steps=-4)
    # change loudness
    loudness_up = audio*1.5
    loudness_down = audio*0.5
    # 3. source estimation
    source = LPC_estimation(audio, p=16)
    source_pu = LPC_estimation(pitch_up, p=16)
    source_pd = LPC_estimation(pitch_down, p=16)
    source_lu = LPC_estimation(loudness_up, p=16)
    source_ld = LPC_estimation(loudness_down, p=16)
    # 4. calculate melspec
    melspec = pre_process_audio_melspec(source)
    melspec_pu = pre_process_audio_melspec(source_pu)
    melspec_pd = pre_process_audio_melspec(source_pd)
    melspec_lu = pre_process_audio_melspec(source_lu)
    melspec_ld = pre_process_audio_melspec(source_ld)
    # print(melspec.shape, melspec_pu.shape, melspec_pd.shape, melspec_lu.shape, melspec_ld.shape)
    # 5. save featuremap
    melspec_path = Path(args.feature_path+filename[:-4] + ".npy")
    melspec_pu_path = Path(args.feature_path+filename[:-4] + "_pu.npy")
    melspec_pd_path = Path(args.feature_path+filename[:-4] + "_pd.npy")
    melspec_lu_path = Path(args.feature_path+filename[:-4] + "_lu.npy")
    melspec_ld_path = Path(args.feature_path+filename[:-4] + "_ld.npy")
    # print(melspec_path)

    np.save(melspec_path, melspec)
    np.save(melspec_pu_path, melspec_pu)
    np.save(melspec_pd_path, melspec_pd)
    np.save(melspec_lu_path, melspec_lu)
    np.save(melspec_ld_path, melspec_ld)
    return True


if __name__ == "__main__":
    from tqdm import tqdm
    # from glob import glob
    from multiprocessing import Pool
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", default="../clarinetData/audio/")
    parser.add_argument("--feature_path", default="../clarinetData/feature/")
    parser.add_argument("--train_g_path", default="../clarinetData/evaluation_setup/train_g.csv")
    parser.add_argument("--train_b_path", default="../clarinetData/evaluation_setup/train_b.csv")
    parser.add_argument("--test_g_path", default="../clarinetData/evaluation_setup/test_g.csv")
    parser.add_argument("--test_b_path", default="../clarinetData/evaluation_setup/test_b.csv")
    args = parser.parse_args()

    base_path = Path(args.wav_path)
    cols = ['filename', 'label']
    train_g = pd.read_csv(args.train_g_path, delimiter=',', header=None, names=cols)
    train_b = pd.read_csv(args.train_b_path, delimiter=',', header=None, names=cols)
    test_g = pd.read_csv(args.test_g_path, delimiter=',', header=None, names=cols)
    test_b = pd.read_csv(args.test_b_path, delimiter=',', header=None, names=cols)
    files_csv = pd.concat([train_g, train_b, test_g, test_b])
    files = files_csv.filename.tolist()
    # files = sorted(list(glob(str(base_path / names ".wav")))
    # print(len(files))

    p = Pool(8)
    for i, _ in tqdm(enumerate(p.imap(save_features, files)), total=len(files)):
        if i % 500 == 0:
            print(i)
