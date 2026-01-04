import typing as tp

import cv2
import grain.python as grain
import numpy as np

from ImageNet1K.ImageRecord import ImageRecord

Image: tp.TypeAlias = np.typing.NDArray[np.uint8]


class Record(tp.TypedDict):
    image: Image
    label: int


class ReadRecord(grain.MapTransform):
    @tp.override
    def map(self, element: bytes) -> Record:
        record = ImageRecord.GetRootAs(element)
        img = cv2.imdecode(record.DataAsNumpy(), cv2.IMREAD_COLOR)
        assert img is not None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = record.Label()
        return Record(image=img, label=label)


class RandomResize(grain.RandomMapTransform):
    def __init__(self, size: tuple[int, int]):
        self.sz = size

    def apply(self, img: Image, size: int) -> Image:
        h, w = img.shape[:2]
        scale = size / min(h, w)
        new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return img.astype(np.uint8)

    @tp.override
    def random_map(self, element: Record, rng: np.random.Generator) -> Record:
        sz = rng.integers(self.sz[0], self.sz[1] + 1)
        element["image"] = self.apply(element["image"], sz)
        return element


class RandomCrop(grain.RandomMapTransform):
    def __init__(self, size: int):
        self.sz = size

    def apply(self, img: Image, h0: int, w0: int) -> Image:
        return img[h0 : h0 + self.sz, w0 : w0 + self.sz]

    @tp.override
    def random_map(self, element: Record, rng: np.random.Generator) -> Record:
        img = element["image"]
        h, w = img.shape[:2]
        if h < self.sz or w < self.sz:
            img = cv2.resize(img, (max(self.sz, w), max(self.sz, h)))
            img = img.astype(np.uint8)
            h, w = img.shape[:2]
        bh, bw = rng.integers(
            np.array([0, 0]), np.array([h - self.sz + 1, w - self.sz + 1])
        )
        element["image"] = self.apply(img, bh, bw)
        return element


class RandomFlip(grain.RandomMapTransform):
    @tp.override
    def random_map(self, element: Record, rng: np.random.Generator) -> Record:
        if rng.random() < 0.5:
            element["image"] = np.ascontiguousarray(np.flip(element["image"], axis=1))
        return element


class RandomResizedCrop(grain.RandomMapTransform):
    def __init__(
        self, size: int, scale: tuple[float, float], ratio: tuple[float, float]
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    @tp.override
    def random_map(self, element: Record, rng: np.random.Generator) -> Record:
        img = element["image"]
        h, w = img.shape[:2]
        area = h * w

        for _ in range(10):
            target_area = rng.uniform(*self.scale) * area
            log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))
            aspect_ratio = np.exp(rng.uniform(*log_ratio))

            w_crop = int(round(np.sqrt(target_area * aspect_ratio)))
            h_crop = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < w_crop <= w and 0 < h_crop <= h:
                top = rng.integers(0, h - h_crop + 1)
                left = rng.integers(0, w - w_crop + 1)

                img_crop = img[top : top + h_crop, left : left + w_crop]
                img_resized = cv2.resize(
                    img_crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR
                )
                element["image"] = img_resized.astype(np.uint8)
                return element

        scale_fallback = self.size / min(h, w)
        nh, nw = int(h * scale_fallback), int(w * scale_fallback)
        img_fallback = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        t = (nh - self.size) // 2
        l = (nw - self.size) // 2
        img_fallback = img_fallback[t : t + self.size, l : l + self.size]
        element["image"] = img_fallback.astype(np.uint8)
        return element


class ColorJitter(grain.RandomMapTransform):
    def __init__(
        self,
        brightness: float | tuple[float, float] = 0,
        contrast: float | tuple[float, float] = 0,
        saturation: float | tuple[float, float] = 0,
        hue: float | tuple[float, float] = 0,
    ):
        # Brightness, Contrast, Saturation are centered at 1.0 (multiplicative)
        self.brightness = self._check_input(
            brightness, center=1.0, bound=(0, float("inf"))
        )
        self.contrast = self._check_input(contrast, center=1.0, bound=(0, float("inf")))
        self.saturation = self._check_input(
            saturation, center=1.0, bound=(0, float("inf"))
        )
        self.hue = self._check_input(hue, center=0.0, bound=(-0.5, 0.5))

    def _check_input(
        self,
        value: float | tuple[float, float],
        center: float,
        bound: tuple[float, float],
    ):
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(
                    "If value is a single number, it must be non-negative."
                )
            low, high = center - value, center + value
        else:
            low, high = center + value[0], center + value[1]

        return (max(bound[0], low), min(bound[1], high))

    def apply_brightness(self, img: Image, factor: float) -> Image:
        return cv2.convertScaleAbs(img, alpha=factor, beta=0)

    def apply_contrast(self, img: Image, factor: float) -> Image:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean = cv2.mean(gray)[0]
        beta = mean * (1.0 - factor)
        return cv2.addWeighted(img, factor, img, 0, beta)

    def apply_saturation(self, img: Image, factor: float) -> Image:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return cv2.addWeighted(img, factor, gray_3ch, (1 - factor), 0)

    def apply_hue(self, img: Image, factor: float) -> Image:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
        h, s, v = cv2.split(hsv)

        shift = int(factor * 180)
        h = (h + shift) % 180

        hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @tp.override
    def random_map(self, element: Record, rng: np.random.Generator) -> Record:
        img = element["image"]

        ops = [
            (self.apply_brightness, rng.uniform(*self.brightness)),
            (self.apply_contrast, rng.uniform(*self.contrast)),
            (self.apply_saturation, rng.uniform(*self.saturation)),
            (self.apply_hue, rng.uniform(*self.hue)),
        ]

        rng.shuffle(ops)

        for func, factor in ops:
            img = func(img, factor)

        element["image"] = img
        return element


class EvalCrop(grain.MapTransform):
    @tp.override
    def map(self, element: Record) -> Record:
        img = element["image"]
        h, w = img.shape[:2]
        scale = 256 / min(h, w)
        new_h, new_w = round(h * scale), round(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        bh, bw = (new_h - 224) // 2, (new_w - 224) // 2
        img = img[bh : bh + 224, bw : bw + 224]
        element["image"] = img.astype(np.uint8)
        return element
