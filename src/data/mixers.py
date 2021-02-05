# originally taken from https://github.com/lRomul/argus-freesound/blob/adad5d11e78237de0f38de6cdaf0f9063f8913db/src/mixers.py
# MIT License
# Copyright (c) 2019 Ruslan Baikulov


# Modifications
# I (https://github.com/SarthakYadav)
#   1. added BackgroundAddMixer,
#   2. Improved SigmoidConcatMixer to use torch only.

# Modifications and additions are covered under
# MIT Licence
# Copyright (c) 2021 Sarthak Yadav

import torch
import random
import numpy as np


def get_random_sample(dataset):
    rnd_idx = random.randint(0, len(dataset) - 1)
    rnd_image, _, rnd_target = dataset.__get_item_helper__(rnd_idx)
    return rnd_image, rnd_target


class BackgroundAddMixer:
    def __init__(self, alpha_dist='uniform'):
        assert alpha_dist in ['uniform', 'beta']
        self.alpha_dist = alpha_dist

    def sample_alpha(self):
        if self.alpha_dist == 'uniform':
            return random.uniform(0, 0.5)
        elif self.alpha_dist == 'beta':
            return np.random.beta(0.4, 0.4)

    def __call__(self, dataset, image, target):
        rnd_idx = random.randint(0, dataset.get_bg_len() - 1)
        rnd_image = dataset.get_bg_feature(rnd_idx)

        alpha = self.sample_alpha()
        image = (1 - alpha) * image + alpha * rnd_image
        return image, target


class AddMixer:
    def __init__(self, alpha_dist='uniform'):
        assert alpha_dist in ['uniform', 'beta']
        self.alpha_dist = alpha_dist

    def sample_alpha(self):
        if self.alpha_dist == 'uniform':
            return random.uniform(0, 0.5)
        elif self.alpha_dist == 'beta':
            return np.random.beta(0.4, 0.4)

    def __call__(self, dataset, image, target):
        rnd_image, rnd_target = get_random_sample(dataset)

        alpha = self.sample_alpha()
        image = (1 - alpha) * image + alpha * rnd_image
        target = (1 - alpha) * target + alpha * rnd_target
        target = torch.clip(target, 0.0, 1.0)
        return image, target


class SigmoidConcatMixer:
    def __init__(self, sigmoid_range=(3, 12)):
        self.sigmoid_range = sigmoid_range

    def sample_mask(self, size):
        x_radius = random.randint(*self.sigmoid_range)

        step = (x_radius * 2) / size[1]
        x = torch.arange(-x_radius, x_radius, step=step).float()
        y = torch.sigmoid(x)
        mix_mask = y.repeat(size[0], 1)
        return mix_mask

    def __call__(self, dataset, image, target):
        rnd_image, rnd_target = get_random_sample(dataset)

        mix_mask = self.sample_mask(image.shape[-2:])
        rnd_mix_mask = 1 - mix_mask

        image = mix_mask * image + rnd_mix_mask * rnd_image
        target = target + rnd_target
        target = torch.clip(target, 0.0, 1.0)
        return image, target


class RandomMixer:
    def __init__(self, mixers, p=None):
        self.mixers = mixers
        self.p = p

    def __call__(self, dataset, image, target):
        mixer = np.random.choice(self.mixers, p=self.p)
        image, target = mixer(dataset, image, target)
        return image, target


class UseMixerWithProb:
    def __init__(self, mixer, prob=.5):
        self.mixer = mixer
        self.prob = prob

    def __call__(self, dataset, image, target):
        if random.random() < self.prob:
            return self.mixer(dataset, image, target)
        return image, target
