import os
import glob
import torch
import librosa
import torchaudio
import numpy as np
import soundfile as sf


class AudioParser(object):
    def __init__(self, n_fft=511, win_length=None, hop_length=None, sample_rate=22050,
                 feature="spectrogram", top_db=150):
        super(AudioParser, self).__init__()
        self.n_fft = n_fft
        self.win_length = self.n_fft if win_length is None else win_length
        self.hop_length = self.n_fft//2 if hop_length is None else hop_length
        assert feature in ['melspectrogram', 'spectrogram']
        self.feature = feature
        self.top_db = top_db
        if feature == "melspectrogram":
            self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=96 * 20,
                                                                win_length=int(sample_rate * 0.03),
                                                                hop_length=int(sample_rate * 0.01),
                                                                n_mels=96)
        else:
            self.melspec = None

    def __call__(self, audio):
        if self.feature == 'spectrogram':
            # TOP_DB = 150
            comp = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                                win_length=self.win_length)
            real = np.abs(comp)
            real = librosa.amplitude_to_db(real, top_db=self.top_db)
            real += self.top_db / 2

            mean = real.mean()
            real -= mean        # per sample Zero Centering
            return real, comp

        elif self.feature == 'melspectrogram':
            # melspectrogram features, as per FSD50k paper
            x = torch.from_numpy(audio).unsqueeze(0)
            specgram = self.melspec(x)[0].numpy()
            specgram = librosa.power_to_db(specgram)
            specgram = specgram.astype('float32')
            specgram += self.top_db / 2
            mean = specgram.mean()
            specgram -= mean
            return specgram, specgram
