import numpy as np
import pandas as pd
from torchvision import transforms


class Jittering():
	def __init__(self, sigma):
		self.sigma = sigma

	def __call__(self, x):
		noise = np.random.normal(loc=0, scale=self.sigma, size=x.shape)
		x = x + noise
		return x


class Scaling():
	def __init__(self, sigma):
		self.sigma = sigma

	def __call__(self, x):
		factor = np.random.normal(loc=1., scale=self.sigma, size=(x.shape))
		x = x * factor
		return x


class Rotation():
    def __init__(self):
        pass

    def __call__(self, x):
        flip = np.random.choice([-1, 1], size=(x.shape))
        return flip * x


class ChannelShuffle():
	def __init__(self):
		pass

	def __call__(self, x):
		rotate_axis = np.arange(x.shape[1])
		np.random.shuffle(rotate_axis) 
		return x[:, rotate_axis]


class Permutation():
	def __init__(self, max_segments=5):
		self.max_segments = max_segments

	def __call__(self, x):
		orig_steps = np.arange(x.shape[0])
		
		num_segs = np.random.randint(1, self.max_segments)
		ret = np.zeros_like(x)

		if num_segs > 1:
			splits = np.array_split(orig_steps, num_segs)
			warp = np.concatenate(np.random.permutation(splits)).ravel()
			ret = x[warp]
		else:
			ret = x
		return ret

augmentations_dict = {
	'jittering': Jittering,
	'scaling': Scaling,
	'rotation': Rotation,
	'permutation': Permutation,
	'channel_shuffle': ChannelShuffle
}

def compose_random_augmentations(config_dict, prob=0.5):
	transforms_list = []
	for key in config_dict:
		if config_dict[key]['apply']:
			augmentation = augmentations_dict[key](**config_dict[key]['parameters'])
			if key == ' jittering':
				transforms_list.append(augmentation)
			else:
				transforms_list.append(transforms.RandomApply([augmentation], p=prob))
	return transforms_list
