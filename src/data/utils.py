import numpy as np
import torch
import soundfile as sf


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    inputs_complex = torch.zeros((minibatch_size, 1, freq_size, max_seqlength), dtype=torch.complex64)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        real_tensor = sample[0]
        # complex_tensor = sample[1]
        target = sample[1]
        seq_length = real_tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(real_tensor)
        # inputs_complex[x][0].narrow(1, 0, seq_length).copy_(complex_tensor)
        targets.append(target.unsqueeze(0))
    targets = torch.cat(targets)
    return inputs, inputs_complex, targets


def _collate_fn_multiclass(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    inputs_complex = torch.zeros((minibatch_size, 1, freq_size, max_seqlength), dtype=torch.complex64)
    targets = torch.LongTensor(minibatch_size)
    # targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        real_tensor = sample[0]
        # complex_tensor = sample[1]
        target = sample[1]
        seq_length = real_tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(real_tensor)
        # inputs_complex[x][0].narrow(1, 0, seq_length).copy_(complex_tensor)
        # targets.append(target)
        targets[x] = target
    # targets = torch.cat(targets)
    return inputs, inputs_complex, targets


def load_audio(f, sr, min_duration: float = 5.):
    if min_duration is not None:
        min_samples = int(sr * min_duration)
    else:
        min_samples = None
    # x, clip_sr = torchaudio.load(f)
    x, clip_sr = sf.read(f)
    x = x.astype('float32')
    assert clip_sr == sr
    # print(x.shape)
    # pre_emphasis = 0.97
    # x = librosa.util.normalize(x)
    # x = np.append(x[0], x[1:] - pre_emphasis * x[:-1])

    # min filtering and padding if needed
    if min_samples is not None:
        if len(x) < min_samples:
            tile_size = (min_samples // x.shape[0]) + 1
            # print("[__get_audio__] tile size:", tile_size)
            x = np.tile(x, tile_size)[:min_samples]
    # preprocessed_audio = torch.from_numpy(x).unsqueeze(0)
    return x
