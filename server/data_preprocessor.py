from skimage.util import random_noise
from scipy.ndimage import rotate
from skimage.transform import swirl, AffineTransform, warp
import numpy as np
from skimage.filters import gaussian

def flip_vertical_np(im):
    ''' Utility for performing vertical image flip '''
    return np.flipud(im)

def flip_horizontal_np(im):
    ''' Utility for performing horizontal image flip '''
    return np.fliplr(im)

def rotate_np(im, severity=2, angle=None):
    ''' Utility for performing random image rotation - Supports 7 severity levels'''
    angle = [15, 28, 40, 50, 65, 80, 90][severity-1]
    angle *= (-1 if np.random.uniform(low=0, high=1) < 0.5 else 1)
    return rotate(im, angle, reshape=False, mode='nearest')

def flip_rotate(im, choice=None, severity=2):
    ''' Utility for performing flip rotate + random image rotation - Supports 7 severity levels'''
    if choice is None: 
        choice = 'vertical' if np.random.uniform(low=0, high=1) < 0.5 else 'horizontal'
    im = flip_vertical_np(im) if choice == 'vertical' else flip_horizontal_np(im)
    return rotate_np(im, severity)

def perform_swirl_transformation(im, severity=2, angle=None):
    ''' Utility for performing swirl transformations - Supports 7 severity levels'''
    strengths = [.05, .15, .20, .25, .35, .40, .5][severity-1]
    angle = [15, 28, 40, 50, 65, 80, 90][severity-1] * (-1 if np.random.uniform(low=0, high=1) < 0.5 else 1)
    return swirl(im, strength=strengths, rotation=angle)

def perform_random_affine_transform(im, severity=2):
    ''' Utility for performing random affine transformations - Supports 7 severity levels'''
    scale = [.005, .075, .1, .105, .11, .12, .13][severity-1]
    # populating random values
    tf_x, tf_y = np.random.uniform(0.0, 0.05), np.random.uniform(0.0, 0.1)
    tf_inv_matrix = AffineTransform( shear=scale, translation=(tf_x, tf_y)).inverse
    
    if np.random.uniform(low=0, high=1) < 0.15:
        im = perform_swirl_transformation(im)
    return warp(im, tf_inv_matrix, order=0)

def add_multiplicative_noise(im, extent=2):
    ''' Utility for adding multiplicative noises - Supports 7 severity levels'''
    scale = [.08, .12, .15, .2, .3, .4, .45][extent-1] 
    im = im / 255.
    return np.uint8((np.clip(im + im * random_noise(im, mode='speckle', var=scale), 0, 1)) * 255)

def add_shot_noise(im, extent=2):
    ''' Utility for adding shot noises - Supports 7 severity levels'''
    # Supports 6 level of severity
    scale = [95, 80, 65, 50, 40, 30, 20][extent-1]
    return np.uint8((np.clip(np.random.poisson((im/255)*scale)/float(scale), 0, 1)) * 255)

def add_gaussian_noise(im, extent=2):
    ''' Utility for adding gaussian noises - Supports 7 severity levels'''
    scale = [.04, .06, .08, .1, .12, .15, .18][extent-1]
    return np.uint8(np.clip(im/255 + np.random.normal(scale=scale, size=im.shape), 0, 1) * 255)

def add_impulse_noise(im, extent=2):
    ''' Utility for adding impulse noises - Supports 7 severity levels'''
    scale = [.01, .03, .05, .08, .12, .15, .18][extent-1]    
    return np.uint8((np.clip(random_noise(im/255, mode='s&p', amount=scale), 0, 1)) * 255)

def add_glass_blur(im, extent=2):
    ''' Utility for adding glass blur - Supports 7 severity levels'''
    c = [(0.4, 2, 2), (0.75, 2, 1), (.9, 2, 2), (1, 2, 3), (1.1, 4, 2), (1.2, 3, 2), (1.3, 4, 3)][extent - 1]
    im = np.uint8(gaussian(np.array(im) / 255., sigma=c[0], multichannel=True) * 255)
    for i in range(c[2]):
        for h in range(28 - c[1], c[1], -1):
            for w in range(28 - c[1], c[1], -1):
                if np.random.choice([True, False], 1)[0]:
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    im[h, w], im[h_prime, w_prime] = im[h_prime, w_prime], im[h, w]

    x = np.clip(gaussian(im / 255., sigma=c[0], multichannel=True), 0, 1) * 255
    return np.uint8(x)

def add_gaussian_blur(im, extent=2):
    ''' Utility for adding gaussian blur - Supports 7 severity levels'''
    scale = [.25, .4, .55, .7, .85, 1, 1.1][extent - 1]
    return np.uint8(np.clip(gaussian(im/255, sigma=scale, multichannel=True), 0, 1) * 255)

def random_image_eraser(im):
    ''' Utility for random image masking '''
    im_ = im.copy()
    H, W, C = im_.shape
    point = np.random.randint(int(H/5), int(H/2))
    w_ = np.random.randint(int(W/5), int(W/2))
    h_ = np.random.randint(int(H/5), int(H/2))
    im_[point:point + h_, point:point + w_, :] = np.random.uniform(0, 255)
    return np.uint8(im_)

class TransformDataset(object):
    def return_function(self, name, im, severity=None):
        if severity is not None:
            return getattr(self, 'if_' + name)(im, severity)
        else:
            return getattr(self, 'if_' + name)(im)
    def if_flip_vertical_np(self, im):
        return flip_vertical_np(im)
    def if_flip_horizontal_np(self, im):
        return flip_horizontal_np(im)
    def if_rotate_np(self, im, severity):
        return rotate_np(im, severity)
    def if_flip_rotate(self, im, severity):
        return flip_rotate(im, severity)
    def if_perform_swirl_transformation(self, im, severity):
        return perform_swirl_transformation(im, severity)
    def if_perform_random_affine_transform(self, im, severity):
        return perform_random_affine_transform(im, severity) 
    def if_add_multiplicative_noise(self, im, severity):
        return add_multiplicative_noise(im, severity)
    def if_add_shot_noise(self, im, severity):
        return add_shot_noise(im, severity)
    def if_add_gaussian_noise(self, im, severity):
        return add_gaussian_noise(im, severity)
    def if_add_impulse_noise(self, im, severity):
        return add_impulse_noise(im, severity)
    def if_add_glass_blur(self, im, severity):
        return add_glass_blur(im, severity)
    def if_add_gaussian_blur(self, im, severity):
        return add_gaussian_blur(im, severity)
    def if_random_image_eraser(self, im):
        return random_image_eraser(im)