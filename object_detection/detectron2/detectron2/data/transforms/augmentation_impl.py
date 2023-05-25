# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Implement many useful :class:`Augmentation`.
"""
import numpy as np
import sys
from typing import Tuple
import torch
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    PadTransform,
    Transform,
    TransformList,
    VFlipTransform,
)
from PIL import Image

from .augmentation import Augmentation, _transform_to_aug
from .transform import ExtentTransform, ResizeTransform, RotationTransform, TranslationTransform

__all__ = [
    "FixedSizeCrop",
    "RandomApply",
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomExtent",
    "RandomFlip",
    "RandomSaturation",
    "RandomLighting",
    "RandomRotation",
    "Resize",
    "ResizeScale",
    "ResizeShortestEdge",
    "RandomCrop_CategoryAreaConstraint",
    "RandomEventNoise",
    "FillBlack",
    "RandomTranslation",
    "RandomEventLines",
    "RandomNoise",
    "RandomStars",
    "RandomHaze",
    "RandomFlares",
    "RandomStreaks",
    "RandomBloom",
    "RandomErasing"
]


class RandomApply(Augmentation):
    """
    Randomly apply an augmentation with a given probability.
    """

    def __init__(self, tfm_or_aug, prob=0.5):
        """
        Args:
            tfm_or_aug (Transform, Augmentation): the transform or augmentation
                to be applied. It can either be a `Transform` or `Augmentation`
                instance.
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
        """
        super().__init__()
        self.aug = _transform_to_aug(tfm_or_aug)
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob

    def get_transform(self, *args):
        do = self._rand_range() < self.prob
        if do:
            return self.aug.get_transform(*args)
        else:
            return NoOpTransform()

    def __call__(self, aug_input):
        do = self._rand_range() < self.prob
        if do:
            return self.aug(aug_input)
        else:
            return NoOpTransform()


class RandomFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class Resize(Augmentation):
    """Resize image to a fixed target size"""

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, image):
        return ResizeTransform(
            image.shape[0], image.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(Augmentation):
    """
    Resize the image while keeping the aspect ratio unchanged.
    It attempts to scale the shorter edge to the given `short_edge_length`,
    as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    @torch.jit.unused
    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style

        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._init(locals())

    @torch.jit.unused
    def get_transform(self, image):
        h, w = image.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()

        newh, neww = ResizeShortestEdge.get_output_shape(h, w, size, self.max_size)
        return ResizeTransform(h, w, newh, neww, self.interp)

    @staticmethod
    def get_output_shape(
        oldh: int, oldw: int, short_edge_length: int, max_size: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class ResizeScale(Augmentation):
    """
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant.
    This implements the resize part of the Google's 'resize_and_crop' data augmentation:
    https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/input_utils.py#L127
    """

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        target_height: int,
        target_width: int,
        interp: int = Image.BILINEAR,
    ):
        """
        Args:
            min_scale: minimum image scale range.
            max_scale: maximum image scale range.
            target_height: target image height.
            target_width: target image width.
            interp: image interpolation method.
        """
        super().__init__()
        self._init(locals())

    def _get_resize(self, image: np.ndarray, scale: float) -> Transform:
        input_size = image.shape[:2]

        # Compute new target size given a scale.
        target_size = (self.target_height, self.target_width)
        target_scale_size = np.multiply(target_size, scale)

        # Compute actual rescaling applied to input image and output size.
        output_scale = np.minimum(
            target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1]
        )
        output_size = np.round(np.multiply(input_size, output_scale)).astype(int)

        return ResizeTransform(
            input_size[0], input_size[1], output_size[0], output_size[1], self.interp
        )

    def get_transform(self, image: np.ndarray) -> Transform:
        random_scale = np.random.uniform(self.min_scale, self.max_scale)
        return self._get_resize(image, random_scale)


class RandomRotation(Augmentation):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            return NoOpTransform()

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)

class RandomTranslation(Augmentation):
    def __init__(self, x_range=[-20,20], y_range=[-20,20], interp=None):
        super().__init__()
        self._init(locals())
        self.x_range = x_range
        self.y_range = y_range

    def get_transform(self, image):
        h, w = image.shape[:2]
        x_shift = np.random.uniform(self.x_range[0], self.x_range[1])
        y_shift = np.random.uniform(self.y_range[0], self.y_range[1])
        return TranslationTransform(h, w, x_shift, y_shift)

class FixedSizeCrop(Augmentation):
    """
    If `crop_size` is smaller than the input image size, then it uses a random crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the right and the bottom of the image to the crop size if `pad` is True, otherwise
    it returns the smaller image.
    """

    def __init__(self, crop_size: Tuple[int], pad: bool = True, pad_value: float = 128.0):
        """
        Args:
            crop_size: target image (height, width).
            pad: if True, will pad images smaller than `crop_size` up to `crop_size`
            pad_value: the padding value.
        """
        super().__init__()
        self._init(locals())

    def _get_crop(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)
        return CropTransform(
            offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0]
        )

    def _get_pad(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        original_size = np.minimum(input_size, output_size)
        return PadTransform(
            0, 0, pad_size[1], pad_size[0], original_size[1], original_size[0], self.pad_value
        )

    def get_transform(self, image: np.ndarray) -> TransformList:
        transforms = [self._get_crop(image)]
        if self.pad:
            transforms.append(self._get_pad(image))
        return TransformList(transforms)


class RandomCrop(Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.

        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        # TODO style of relative_range and absolute_range are not consistent:
        # one takes (h, w) but another takes (min, max)
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError("Unknown crop type {}".format(self.crop_type))


class RandomCrop_CategoryAreaConstraint(Augmentation):
    """
    Similar to :class:`RandomCrop`, but find a cropping window such that no single category
    occupies a ratio of more than `single_category_max_area` in semantic segmentation ground
    truth, which can cause unstability in training. The function attempts to find such a valid
    cropping window for at most 10 times.
    """

    def __init__(
        self,
        crop_type: str,
        crop_size,
        single_category_max_area: float = 1.0,
        ignored_category: int = None,
    ):
        """
        Args:
            crop_type, crop_size: same as in :class:`RandomCrop`
            single_category_max_area: the maximum allowed area ratio of a
                category. Set to 1.0 to disable
            ignored_category: allow this category in the semantic segmentation
                ground truth to exceed the area ratio. Usually set to the category
                that's ignored in training.
        """
        self.crop_aug = RandomCrop(crop_type, crop_size)
        self._init(locals())

    def get_transform(self, image, sem_seg):
        if self.single_category_max_area >= 1.0:
            return self.crop_aug.get_transform(image)
        else:
            h, w = sem_seg.shape
            for _ in range(10):
                crop_size = self.crop_aug.get_crop_size((h, w))
                y0 = np.random.randint(h - crop_size[0] + 1)
                x0 = np.random.randint(w - crop_size[1] + 1)
                sem_seg_temp = sem_seg[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
                labels, cnt = np.unique(sem_seg_temp, return_counts=True)
                if self.ignored_category is not None:
                    cnt = cnt[labels != self.ignored_category]
                if len(cnt) > 1 and np.max(cnt) < np.sum(cnt) * self.single_category_max_area:
                    break
            crop_tfm = CropTransform(x0, y0, crop_size[1], crop_size[0])
            return crop_tfm


class RandomExtent(Augmentation):
    """
    Outputs an image by cropping a random "subrect" of the source image.

    The subrect can be parameterized to include pixels outside the source image,
    in which case they will be set to zeros (i.e. black). The size of the output
    image will vary with the size of the random subrect.
    """

    def __init__(self, scale_range, shift_range):
        """
        Args:
            output_size (h, w): Dimensions of output image
            scale_range (l, h): Range of input-to-output size scaling factor
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        img_h, img_w = image.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

        # Apply a random scaling to the src_rect.
        src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(int(src_rect[3] - src_rect[1]), int(src_rect[2] - src_rect[0])),
        )


class RandomContrast(Augmentation):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=image.mean(), src_weight=1 - w, dst_weight=w)


class RandomBrightness(Augmentation):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)


class RandomSaturation(Augmentation):
    """
    Randomly transforms saturation of an RGB image.
    Input images are assumed to have 'RGB' channel order.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        assert image.shape[-1] == 3, "RandomSaturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = image.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)


class RandomLighting(Augmentation):
    """
    The "lighting" augmentation described in AlexNet, using fixed PCA over ImageNet.
    Input images are assumed to have 'RGB' channel order.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self._init(locals())
        self.eigen_vecs = np.array(
            [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def get_transform(self, image):
        assert image.shape[-1] == 3, "RandomLighting only works on RGB images"
        weights = np.random.normal(scale=self.scale, size=3)
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals), src_weight=1.0, dst_weight=1.0
        )

class RandomEventNoise(Augmentation):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        H, W, C = image.shape
        assert image.min()>=0.
        #print(image.shape)

        blank = np.full((H, W, 1), 0, dtype=np.uint8).repeat(3, -1)
        threshold = np.random.rand() * (0.05 - 0.001) + 0.001
        # Add negative events.
        indices = np.random.choice(H*W, replace=False, size=int(H*W*threshold))
        blank[np.unravel_index(indices, blank.shape[:2])] = np.array([255,255,255])

        w = 1
        if np.random.rand() < 0.3:
            w = 0
        return BlendTransform(src_image=blank, src_weight=w, dst_weight=1)

class FillBlack(Augmentation):
    """
    Fill black pixels with gray.
    0 -> 127
    127 -> 127
    255 -> 255

    0 - 127 = -127
    127 - 127 = 0
    255 - 127 = 128

    clamp to [0,255]
    0
    0
    128

    * 2 then clamp to [0,255]
    + 127
    """
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())
        
    def get_transform(self, image):
        H, W, C = image.shape
        assert image.min()>=0.

        copy = np.copy(image)
        black_pixels_mask = np.all(copy <= [100,100,100], axis=-1)
        copy[black_pixels_mask] = np.array([127,127,127])

        # print(np.unique(image[:,:,0]))

        # new_image = np.copy(image)
        # new_image -= 127
        # np.clip(new_image, 0, 255, new_image)
        # new_image *= 2
        # np.clip(new_image, 0, 255, new_image)
        # new_image += 127

        return BlendTransform(src_image=copy, src_weight=1, dst_weight=0)

class RandomEventLines(Augmentation):
    """
    Add random noisy event lines.
    """
    def __init__(self, x_jitter=5):
        super().__init__()
        self.x_jitter = x_jitter
        self._init(locals())

    def y(self, x, m, x1, y1):
        return m*(x - x1) + y1
        
    def get_transform(self, image):
        H, W, C = image.shape
        assert image.min()>=0.

        blank = np.full((H, W, 1), 0, dtype=np.uint8).repeat(3, -1)
        
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

        blank[Y, X] = np.array([255,255,255])
        w = 1
        if np.random.rand() < 0.3:
            w = 0
        return BlendTransform(src_image=blank, src_weight=w, dst_weight=1)

class RandomNoise(Augmentation):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self, mean_min=0.05, mean_max=0.15, std_min=0.03, std_max=0.05):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

        self.std_min = std_min
        self.std_range = std_max - std_min
        self.mean_min = mean_min
        self.mean_range = mean_max - mean_min
        
    def get_transform(self, image):
        H, W, C = image.shape
        assert image.min()>=0.
        #assert image.max()<=1.

        # dice = torch.rand(1)
        # if dice<0.5:
        #     noise = 2*torch.rand(1,H,W).repeat(3,1,1)-1
        # else:
        #     noise = torch.randn(1,H,W).repeat(3,1,1)
        noise = np.random.randn(H, W, 1).repeat(3, -1)

        white_pixel = 255
        # shift = torch.randn(1).item()*self.shift_std
        std = np.random.rand(1)*self.std_range + self.std_min
        mean = np.random.rand(1)*self.mean_range + self.mean_min
        noisy_image = (white_pixel*std)*(white_pixel*noise)+(white_pixel*mean)+image
        noisy_image = noisy_image.clip(min=0., max=255.)

        w = np.random.uniform(0, 0.2)
        return BlendTransform(src_image=noisy_image, src_weight=w, dst_weight=1)

class RandomStars(Augmentation):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self, mean_min=0.05, mean_max=0.15, std_min=0.03, std_max=0.05):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

        self.std_min = std_min
        self.std_range = std_max - std_min
        self.mean_min = mean_min
        self.mean_range = mean_max - mean_min
        
    def get_transform(self, image):
        H, W, C = image.shape
        assert image.min()>=0.
        #assert image.max()<=1.

        # dice = torch.rand(1)
        # if dice<0.5:
        #     noise = 2*torch.rand(1,H,W).repeat(3,1,1)-1
        # else:
        #     noise = torch.randn(1,H,W).repeat(3,1,1)
        noise = np.random.randn(H, W, 1).repeat(3, -1)

        white_pixel = 255
        # shift = torch.randn(1).item()*self.shift_std
        std = np.random.rand(1)*self.std_range + self.std_min
        mean = np.random.rand(1)*self.mean_range + self.mean_min
        noise = (white_pixel*std)*(white_pixel*noise)+(white_pixel*mean)
        noise = noise.clip(min=0., max=255.)
        noise = np.array(Image.fromarray(np.uint8(noise)).filter(ImageFilter.GaussianBlur(radius=3.5)))

        noise[noise < 160] = 0
        noise = np.array(Image.fromarray(np.uint8(noise)).filter(ImageFilter.GaussianBlur(radius=np.random.uniform(1.5,2))))

        # either overlay or don't
        w = np.random.randint(2)
        return BlendTransform(src_image=noise, src_weight=w, dst_weight=1)

class RandomHaze(Augmentation):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self, mean_min=0.05, mean_max=0.15, std_min=0.03, std_max=0.05):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

        self.std_min = std_min
        self.std_range = std_max - std_min
        self.mean_min = mean_min
        self.mean_range = mean_max - mean_min

    def _translate_image(self, image, W=1920, H=1200, t_x=0, t_y=0):
        M = np.float32([[1, 0, t_x], 
                        [0, 1, t_y],
                        [0, 0, 1]])
        return cv.warpPerspective(image, M, (W, H))
        
    def _scale_image(self, image, W=1920, H=1200, s_x=0, s_y=0):
        M = np.float32([[1.5, 0, 0], 
                        [0, 1.8, 0],
                        [0, 0, 1]])
        return cv.warpPerspective(image, M, (W, H))

    def get_transform(self, image):
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
        return BlendTransform(src_image=noise, src_weight=w, dst_weight=1)

class RandomFlares(Augmentation):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())


    def _rotate_image(self, image, angle, W=1920, H=1200):
        image_center = (W/2, H/2)
        rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
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
        return cv.warpPerspective(image, M, (W, H))
        
    def get_transform(self, image):
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
            cv.fillPoly(blank, [pts], color)
            blank  = self._rotate_image(blank, np.random.randint(0, 180), W=W, H=H)
            blank = self._shear_image(blank, scale=np.random.uniform(0, 0.75), W=W, H=H)
            blank = np.array(Image.fromarray(np.uint8(blank)).filter(ImageFilter.GaussianBlur(radius=np.random.uniform(1,5))))
            blank = blank * np.random.uniform(0.4, 1.2)

        # either overlay or don't
        w = np.random.uniform(0, 1)
        return BlendTransform(src_image=blank, src_weight=w, dst_weight=1)

class RandomStreaks(Augmentation):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self, mean_min=0.05, mean_max=0.15, std_min=0.03, std_max=0.05):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

        self.std_min = std_min
        self.std_range = std_max - std_min
        self.mean_min = mean_min
        self.mean_range = mean_max - mean_min

    def _rotate_image(self, image, angle, W=1920, H=1200):
        rot_mat = cv.getRotationMatrix2D((W/2, H/2), angle, 1.0)
        result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
        return result

    def _translate_image(self, image, t_x=0, t_y=0, W=1920, H=1200):
        M = np.float32([[1, 0, t_x], 
                        [0, 1, t_y],
                        [0, 0, 1]])
        return cv.warpPerspective(image, M, (W, H))
        
    def _scale_image(self, image, s_x=1, s_y=1, W=1920, H=1200):
        M = np.float32([[s_x, 0, 0], 
                        [0, s_y, 0],
                        [0, 0, 1]])
        return cv.warpPerspective(image, M, (W, H))

    def _shear_image(self, image, scale=0, W=1920, H=1200):
        M = np.float32([[1, scale, 0], 
                        [0, 1, 0],
                        [0, 0, 1]])
        return cv.warpPerspective(image, M, (W, H))

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
            tmp1 = cv.remap(image, growMapx, growMapy, cv.INTER_LINEAR)
            tmp2 = cv.remap(image, shrinkMapx, shrinkMapy, cv.INTER_LINEAR)
            image = cv.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
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

    def get_transform(self, image):
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
        return BlendTransform(src_image=noise, src_weight=w, dst_weight=1)

class RandomBloom(Augmentation):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        H, W, C = image.shape
        
        offset = np.random.randint(10, 100)

        if not offset % 2 == 0:
            offset += 1

        image = cv.GaussianBlur(image, ksize=(9 + offset, 9 + offset), sigmaX=10, sigmaY=10)
        image = cv.blur(image, ksize=(5 + offset, 5 + offset))

        offset = np.random.randint(0, 200)

        w = 1
        return BlendTransform(src_image=image, src_weight=w, dst_weight=1)

class RandomErasing(Augmentation):
    """
    Add random gaussian noise to an image.
    """
    def __init__(self):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def _translate_image(self, image, t_x=0, t_y=0, W=1920, H=1200):
        M = np.float32([[1, 0, t_x], 
                        [0, 1, t_y],
                        [0, 0, 1]])
        return cv.warpPerspective(image, M, (W, H))
        
    def _scale_image(self, image, s_x=1, s_y=1, W=1920, H=1200):
        M = np.float32([[s_x, 0, 0], 
                        [0, s_y, 0],
                        [0, 0, 1]])
        return cv.warpPerspective(image, M, (W, H))

    def _shear_image(self, image, scale=0, W=1920, H=1200):
        M = np.float32([[1, scale, 0], 
                        [0, 1, 0],
                        [0, 0, 1]])
        return cv.warpPerspective(image, M, (W, H))

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
            tmp1 = cv.remap(image, growMapx, growMapy, cv.INTER_LINEAR)
            tmp2 = cv.remap(image, shrinkMapx, shrinkMapy, cv.INTER_LINEAR)
            image = cv.addWeighted(tmp1, 0.5, tmp2, 0.5, 0)
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

    def get_transform(self, image):
        H, W, C = image.shape
        
        offset = np.random.randint(10, 100)

        if not offset % 2 == 0:
            offset += 1

        blank = np.full((H, W, 1), 0, dtype=np.uint8).repeat(3, -1)

        half_width = int(W/2)
        half_height = int(H/2)
        cv.ellipse(blank, (half_width, half_height), (half_height, half_height), 0, 0, 360, (255,255,255), -1);

        blank = self._radial_blur(blank, np.random.uniform(0.01, 0.04), 5, np.random.randint(0,1920), np.random.randint(0,1200))
        blank = self._radial_fade(blank, W, H)
        blank = self._scale_image(blank, s_x=np.random.uniform(0.1, 0.3), s_y=np.random.uniform(0.1, 0.3), W=W, H=H)
        blank = self._translate_image(blank, t_x=np.random.randint(-300, 300), t_y=np.random.randint(-300,300), W=W, H=H)

        w = 1
        return BlendTransform(src_image=blank, src_weight=w, dst_weight=1)