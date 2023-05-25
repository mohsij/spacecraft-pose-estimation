# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch

def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img

def BlendTransform(img, src_image, src_weight, dst_weight):
    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = src_weight * src_image + dst_weight * img
        return np.clip(img, 0, 255).astype(np.uint8)
    else:
        return src_weight * src_image + dst_weight * img

class EventNoise(object):
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """

    def __call__(self, image):
        img = np.array(image)
        H, W, C = img.shape
        assert img.min()>=0

        if np.random.rand() < 0.3:
            return img
        # threshold = np.random.rand() * (0.05 - 0.001) + 0.001
        # # Add negative events.
        # indices = np.random.choice(H*W, replace=False, size=int(H*W*threshold))
        # img[np.unravel_index(indices, img.shape[:2])] = np.array([0,0,0])
        
        threshold = np.random.rand() * (0.05 - 0.001) + 0.001
        # Add negative events.
        indices = np.random.choice(H*W, replace=False, size=int(H*W*threshold))
        img[np.unravel_index(indices, img.shape[:2])] = np.array([255,255,255])

        return img

class EventLines(object):
    def __init__(self, x_jitter=5):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        self.x_jitter = x_jitter

    def y(self, x, m, x1, y1):
        return m*(x - x1) + y1

    def add_line(self, img):
        H, W, C = img.shape
        if np.random.rand() < 0.3:
            return img
        
        x_shift = np.random.uniform(-200,200)

        x1,y1 = [np.random.uniform(x_shift, W - x_shift), 0]
        x2,y2 = [x1 + x_shift, H]

        m = (y2 - float(y1)) / (x2 - float(x1))

        density = np.random.randint(100,10000)
        X = np.linspace(x1, x2, density, dtype=np.int32)
        Y = self.y(X, m, x1, y1)
        X = np.array([x + np.random.uniform(-self.x_jitter,self.x_jitter) for x in X])
        Y = np.array([y + np.random.uniform(-100 + x_shift / 2, 100 - x_shift / 2) for y in Y])
        X = np.clip(X, 0, W - 1)
        Y = np.clip(Y, 0, H - 1)
        X = np.floor(X)
        Y = np.floor(Y)
        X = X.astype(np.int32)
        Y = Y.astype(np.int32)
        indices = np.column_stack((Y, X))
        indices = indices.tolist()

        img[Y, X] = np.array([255,255,255])

        return img

    def __call__(self, image):
        img = np.array(image)
        H, W, C = img.shape
        assert img.min()>=0

        if np.random.rand() < 0.3:
            return img

        for x in range(3):
            img = self.add_line(img)

        return img

### Augmentations created for speedplus ###
class RandomHaze(object):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self, mean_min=0.05, mean_max=0.15, std_min=0.03, std_max=0.05):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """

        self.std_min = std_min
        self.std_range = std_max - std_min
        self.mean_min = mean_min
        self.mean_range = mean_max - mean_min

    def _translate_image(self, image, W=1920, H=1200, t_x=0, t_y=0):
        M = np.float32([[1, 0, t_x], 
                        [0, 1, t_y],
                        [0, 0, 1]])
        return cv2.warpPerspective(image, M, (W, H))
        
    def _scale_image(self, image, W=1920, H=1200, s_x=0, s_y=0):
        M = np.float32([[1.5, 0, 0], 
                        [0, 1.8, 0],
                        [0, 0, 1]])
        return cv2.warpPerspective(image, M, (W, H))

    def __call__(self, image):
        H, W, C = image.shape
        assert image.min()>=0

        noise = np.random.randn(H, W, 1).repeat(3, -1)

        white_pixel = 255
        std = np.random.rand(1)*self.std_range + self.std_min
        mean = np.random.rand(1)*self.mean_range + self.mean_min
        noise = (white_pixel*std)*(white_pixel*noise)+(white_pixel*mean)
        noise = noise.clip(min=0., max=255.)
        noise = np.array(Image.fromarray(np.uint8(noise)).filter(ImageFilter.GaussianBlur(radius=5)))

        noise[noise < np.random.randint(125,140)] = 0
        noise = self._scale_image(noise, s_x=np.random.randint(0.75, 1.25), s_y=np.random.randint(0.75, 1.25), W=W, H=H)
        noise = np.array(Image.fromarray(np.uint8(noise)).filter(ImageFilter.GaussianBlur(radius=np.random.uniform(25,40))))

        # either overlay or don't
        w = np.random.uniform(0.1, 0.8)
        return BlendTransform(image, src_image=noise, src_weight=w, dst_weight=1)

class RandomFlares(object):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """


    def _rotate_image(self, image, angle, W=1920, H=1200):
        image_center = (W/2, H/2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _polygon_vertices(self, x, y, r, sides=6):
        vertices = [[x, y + r]]
        for angle in np.linspace(0, 2 * np.pi, sides):
            vertices.append([x + r * np.sin(angle), y + r * np.cos(angle)])
        vertices = np.array(vertices, dtype=np.int32)
        return vertices

    def _shear_image(self, image, W=1920, H=1200, scale=0):
        M = np.float32([[1, scale, 0], 
                        [0, 1, 0],
                        [0, 0, 1]])
        return cv2.warpPerspective(image, M, (W, H))
        
    def __call__(self, image):
        H, W, C = image.shape
        assert image.min()>=0.
        #assert image.max()<=1.

        blank = np.full((H, W, 1), 0, dtype=np.uint8).repeat(3, -1)

        for i in range(np.random.randint(1, 10)):
            centre_x = 1920/2
            centre_y = 1200/2
            x_offset = np.random.randint(centre_x - 500, centre_x + 500)
            y_offset = np.random.randint(centre_y - 400, centre_y + 400)
            # make a pentagon
            pts = self._polygon_vertices(x_offset, y_offset, np.random.randint(5,100))
     
            color = (255, 255, 255)
            cv2.fillPoly(blank, [pts], color)
            blank  = self._rotate_image(blank, np.random.randint(0, 180), W=W, H=H)
            blank = self._shear_image(blank, scale=np.random.uniform(0, 0.75), W=W, H=H)
            blank = np.array(Image.fromarray(np.uint8(blank)).filter(ImageFilter.GaussianBlur(radius=np.random.uniform(1,5))))
            blank = blank * np.random.uniform(0.4, 1.2)

        # either overlay or don't
        w = np.random.uniform(0, 1)
        return BlendTransform(image, src_image=blank, src_weight=w, dst_weight=1)

class RandomStreaks(object):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self, mean_min=0.05, mean_max=0.15, std_min=0.03, std_max=0.05):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """

        self.std_min = std_min
        self.std_range = std_max - std_min
        self.mean_min = mean_min
        self.mean_range = mean_max - mean_min

    def _rotate_image(self, image, angle, W=1920, H=1200):
        rot_mat = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _translate_image(self, image, t_x=0, t_y=0, W=1920, H=1200):
        M = np.float32([[1, 0, t_x], 
                        [0, 1, t_y],
                        [0, 0, 1]])
        return cv2.warpPerspective(image, M, (W, H))
        
    def _scale_image(self, image, s_x=1, s_y=1, W=1920, H=1200):
        M = np.float32([[s_x, 0, 0], 
                        [0, s_y, 0],
                        [0, 0, 1]])
        return cv2.warpPerspective(image, M, (W, H))

    def _shear_image(self, image, scale=0, W=1920, H=1200):
        M = np.float32([[1, scale, 0], 
                        [0, 1, 0],
                        [0, 0, 1]])
        return cv2.warpPerspective(image, M, (W, H))

    def _motion_blur(self, image, kernel_size=15):
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size

        # applying the kernel to the input image
        return cv2.filter2D(image, -1, kernel_motion_blur)

    def _radial_blur(self, image, blur_amount=0.01, iterations=5, center_x=0, center_y=0):
        # From : https://stackoverflow.com/questions/7607464/implement-radial-blur-with-opencv
        blur = blur_amount

        w, h = image.shape[:2]

        growMapx = np.tile(np.arange(h) + ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
        shrinkMapx = np.tile(np.arange(h) - ((np.arange(h) - center_x)*blur), (w, 1)).astype(np.float32)
        growMapy = np.tile(np.arange(w) + ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
        shrinkMapy = np.tile(np.arange(w) - ((np.arange(w) - center_y)*blur), (h, 1)).transpose().astype(np.float32)
        growMapx, growMapy = np.abs(growMapx), np.abs(growMapy)
        for i in range(iterations):
            tmp1 = cv2.remap(image, growMapx, growMapy, cv2.INTER_LINEAR)
            tmp2 = cv2.remap(image, shrinkMapx, shrinkMapy, cv2.INTER_LINEAR)
            image = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
        return image

    def _radial_fade(self, image, W=1920, H=1200):
        # From: https://stackoverflow.com/questions/62045155/how-to-create-a-transparent-radial-gradient-with-python

        # Create radial alpha/transparency layer. 255 in centre, 0 at edge
        X = np.linspace(-1, 1, H)[:, None]*255
        Y = np.linspace(-1, 1, W)[None, :]*255
        alpha = np.sqrt(X**2 + Y**2)
        alpha = 255 - np.clip(0,255,alpha)
        alpha = np.expand_dims(alpha, -1)
        alpha = alpha.repeat(3, -1)
        # Push that radial gradient transparency onto red image and save
        #return Image.fromarray(image.astype(np.uint8)).putalpha(Image.fromarray(alpha.astype(np.uint8)))
        return image * (alpha / 255)

    def __call__(self, image):
        H, W, C = image.shape
        assert image.min()>=0.
        #assert image.max()<=1.

        noise = np.random.randn(H, W, 1).repeat(3, -1)

        white_pixel = 255
        std = np.random.rand(1)*self.std_range + self.std_min
        mean = np.random.rand(1)*self.mean_range + self.mean_min
        noise = (white_pixel*std)*(white_pixel*noise)+(white_pixel*mean)
        noise = noise.clip(min=0., max=255.)
        noise = np.array(Image.fromarray(np.uint8(noise)).filter(ImageFilter.GaussianBlur(radius=1)))
        noise[noise < np.random.randint(150,200)] = 0
        noise = self._radial_blur(noise, np.random.uniform(0.01, 0.04), 5, np.random.randint(0,1920), np.random.randint(0,1200))
        noise = self._radial_fade(noise, W, H)
        if np.random.randint(2) == 1:
            noise = self._scale_image(noise, s_x=np.random.uniform(2, 4), W=W, H=H)
        else:
            noise = self._scale_image(noise, s_y=np.random.uniform(2, 4), W=W, H=H)

        noise = self._rotate_image(noise, np.random.uniform(0, np.pi), W=W, H=H)
        # either overlay or don't
        w = np.random.uniform(0, 1)
        return BlendTransform(image, src_image=noise, src_weight=w, dst_weight=1)

class RandomBloom(object):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """

    def __call__(self, image):
        H, W, C = image.shape
        
        offset = np.random.randint(10, 100)

        if not offset % 2 == 0:
            offset += 1

        bloom_image = np.copy(image)
        bloom_image = cv2.GaussianBlur(bloom_image, ksize=(9 + offset, 9 + offset), sigmaX=10, sigmaY=10)
        bloom_image = cv2.blur(bloom_image, ksize=(5 + offset, 5 + offset))

        offset = np.random.randint(0, 200)

        w = 1
        return BlendTransform(image, src_image=bloom_image, src_weight=w, dst_weight=1)

class ToNumpy(object):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """

    def __call__(self, image):
        return np.array(image)

class RandomNoise(torch.nn.Module):
    def __init__(self, mean_min=0.05, mean_max=0.15, std_min=0.03, std_max=0.05):
        super().__init__()
        self.std_min = std_min
        self.std_range = std_max - std_min
        self.mean_min = mean_min
        self.mean_range = mean_max - mean_min
    def forward(self, img):
        C, H, W = img.size()
        assert img.min()>=0.
        assert img.max()<=1.

        noise = torch.randn(1,H,W).repeat(3,1,1)
        
        std = torch.rand(1)*self.std_range + self.std_min
        mean = torch.rand(1)*self.mean_range + self.mean_min
        img = std*noise+mean+img
        return img.clamp(min=0., max=1.)

