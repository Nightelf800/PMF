import numbers
from typing import Tuple, List, Optional
import numpy as np
import cv2

class ImageColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self.check_input(brightness, "brightness")
        self.contrast = self.check_input(contrast, "contrast")
        self.saturation = self.check_input(saturation, "saturation")
        self.hue = self.check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def get_params(self,
                   brightness: Optional[List[float]],
                   contrast: Optional[List[float]],
                   saturation: Optional[List[float]],
                   hue: Optional[List[float]]
                   ) -> Tuple[np.array, Optional[float], Optional[float], Optional[float], Optional[float]]:
        fn_idx = np.arange(4)
        np.random.shuffle(fn_idx)

        b = None if brightness is None else float(np.random.uniform(brightness[0], brightness[1]))
        c = None if contrast is None else float(np.random.uniform(contrast[0], contrast[1]))
        s = None if saturation is None else float(np.random.uniform(saturation[0], saturation[1]))
        h = None if hue is None else float(np.random.uniform(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def data_arg(self, img):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = self.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = self.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = self.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = self.adjust_hue(img, hue_factor)

        return img

    def rgb_to_grayscale(self, img):
        if img.ndim < 3:
            raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")

        r = img[0, :, :]
        g = img[1, :, :]
        b = img[2, :, :]
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b)
        l_img = np.expand_dims(l_img, -3)

        return l_img

    def adjust_brightness(self, img1, ratio: float):
        return self.blend(img1, np.zeros_like(img1), ratio)

    def adjust_contrast(self, img, contrast_factor: float):
        c = 1 if img.ndim == 2 else img.shape[-3]
        if c == 3:
            mean = np.mean(self.rgb_to_grayscale(img), axis=(-3, -2, -1), dtype=np.float32, keepdims=True)
        else:
            mean = np.mean(img, axis=(-3, -2, -1), dtype=np.float32, keepdims=True)
        return self.blend(img, mean, contrast_factor)

    def adjust_saturation(self, img, saturation_factor: float):
        return self.blend(img, self.rgb_to_grayscale(img), saturation_factor)

    def adjust_hue(self, img, hue_factor: float):
        channels = 1 if img.ndim == 2 else img.shape[-3]
        if channels == 1:  # Match PIL behaviour
            return img

        orig_dtype = img.dtype
        if img.dtype == np.uint8:
            img = img.astype('float32') / 255.0

        img = self._rgb2hsv(img)
        h = img[0, :, :]
        s = img[1, :, :]
        v = img[2, :, :]
        h = (h + hue_factor) % 1.0
        img = np.stack((h, s, v), axis=-3)
        img_hue_adj = self._hsv2rgb(img)

        if orig_dtype == np.uint8:
            img_hue_adj = (img_hue_adj * 255.0).astype('float32')

        return img_hue_adj

    def _rgb2hsv(self, img):
        r = img[0, :, :]
        g = img[1, :, :]
        b = img[2, :, :]
        maxc = np.max(img, axis=-3)
        minc = np.min(img, axis=-3)
        eqc = maxc == minc

        cr = maxc - minc
        ones = np.ones_like(maxc)
        s = cr / np.where(eqc, ones, maxc)
        cr_divisor = np.where(eqc, ones, cr)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = np.fmod((h / 6.0 + 1.0), 1.0)
        return np.stack((h, s, maxc), axis=-3)

    def _hsv2rgb(self, img):
        h = img[0, :, :]
        s = img[1, :, :]
        v = img[2, :, :]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype('int32')

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6

        mask = np.expand_dims(i, -3) == np.arange(6).view(-1, 1, 1)

        a1 = np.stack((v, q, p, p, t, v), axis=-3)
        a2 = np.stack((t, v, v, q, p, p), axis=-3)
        a3 = np.stack((p, p, t, v, v, q), axis=-3)
        a4 = np.stack((a1, a2, a3), axis=-4)

        return np.einsum("...ijk, ...xijk -> ...xjk", mask.astype(img.dtype), a4)


    def blend(self, img1, img2, ratio: float):
        ratio = float(ratio)
        bound = 1.0 if img1.dtype == np.float32 else 255.0
        return np.clip(ratio * img1 + (1.0 - ratio) * img2, 0, bound)


class ImageRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def data_arg(self, img):
        if np.random.rand(1) < self.p:
            img = np.flip(img, -1)
        return img


class ImageRandomRotation:
    def __init__(self, angle, expand=False, center=None, fill=0):
        self.angle = angle
        self.center = center
        self.expand = expand
        self.fill = fill

    def data_arg(self, img):
        _, h, w = img.shape
        noise_angle = float(np.random.uniform(-self.angle, self.angle))
        a, b = h / 2, w / 2
        M = cv2.getRotationMatrix2D((a, b), noise_angle, 1)
        """img_trans = img.reshape(img.shape[1], img.shape[2], img.shape[0])
        rotated_img = cv2.warpAffine(img_trans, M, (w, h))
        rotated_img = rotated_img.reshape(img.shape[0], img.shape[1], img.shape[2])"""
        for i in range(img.shape[0]):
            img[i] = cv2.warpAffine(img[i], M, (w, h))
        return img


class ImageRandomCrop:
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def data_arg(self, img):
        _, height, width = img.shape
        i, j, h, w = self.get_params(img, self.size)

        return self.crop(img, i, j, h, w)

    def get_params(self, img, output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        _, h, w = img.shape
        th, tw = output_size
        tw = int(tw)
        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        return i, j, th, tw

    def crop(self, img, top: int, left: int, height: int, width: int):
        _, h, w = img.shape
        right = left + width
        bottom = top + height
        return img[..., top:bottom, left:right]


class ImageCenterCrop:
    def __init__(self, size):
        self.size = size

    def data_arg(self, img):
        _, image_height, image_width = img.shape
        crop_height, crop_width = self.size

        if crop_width > image_width or crop_height > image_height:
            w = (crop_width - image_width) // 2 if crop_width > image_width else 0
            h = (crop_height - image_height) // 2 if crop_height > image_height else 0
            img = np.pad(img, ((0, 0), (h, h), (w, w)), 'constant')  # PIL uses fill value 0
            _, image_height, image_width = img.shape
            if crop_width == image_width and crop_height == image_height:
                return img

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return self.crop(img, crop_top, crop_left, crop_height, crop_width)

    def crop(self, img, top: int, left: int, height: int, width: int):
        _, h, w = img.shape
        right = left + width
        bottom = top + height
        return img[..., top:bottom, left:right]


class ImagePad:
    def __init__(self, padding, fill=0, padding_mode="constant"):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def data_arg(self, img):
        w, h = self.padding
        img = np.pad(img, ((0, 0), (h, h), (w, w)), self.padding_mode)  # PIL uses fill value 0
        return img
