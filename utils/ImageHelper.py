#  Copyright (c) 2021 by Yingdong Chen <chen.yingdong.cs9@is.naist.jp>,
#  Imaging-based Computational Biomedicine Laboratory, Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yingdong Chen.

import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
from tqdm import tqdm

class ImageHelper:
    @staticmethod
    def label_images(
            images: np.ndarray,
            labels: np.ndarray,
            colors: list,
            thickness: int,
            condit_labels: np.ndarray = None,
            condict_label_color=None,
    ):
        # images (N, H, W, 3) or (N, H, W, 1) or (N, H, W)
        # labels (N, H, W)

        assert images.dtype == np.uint8
        images = ImageHelper.vol_to_video_shape(images.copy())

        image_count = min(images.shape[0], labels.shape[0])
        for i in tqdm(range(image_count), desc="labeling images"):
            if condit_labels is not None and condict_label_color is not None:
                images[i] = ImageHelper.label_image(images[i], condit_labels[i], condict_label_color, 2)
            images[i] = ImageHelper.label_image(images[i], labels[i], colors, thickness)
        return images

    @staticmethod
    def label_image(image, label, colors: list, thickness: int) -> np.ndarray:
        # image (H, W, 3) (np.uint8)
        # label (H, W)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        assert image.ndim == 3
        if image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))
        for i in range(1, len(colors) + 1):
            label_mask = label == i
            label_mask = label_mask.astype(np.uint8)
            cont, _ = cv2.findContours(label_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            image = cv2.drawContours(image, cont, -1, colors[i-1], thickness)

        return image
        pass

    @staticmethod
    def convert_images(images, cv_code) -> np.ndarray or [np.ndarray, ...]:
        re = []
        if isinstance(images, list):
            num_images = len(images)
        elif isinstance(images, np.ndarray):
            num_images = images.shape[0]
        else:
            raise NotImplementedError("Unknown type for images: {}.".format(type(images)))

        for i in range(num_images):
            image = images[i]
            converted_image = cv2.cvtColor(image, cv_code)
            re.append(converted_image)
        if isinstance(images, list):
            return re
        else:
            return np.asarray(re)

    @staticmethod
    def plt_figure_to_opencv_array(fig, format="png", dpi=180):
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        return cv2.imdecode(img_arr, 1)

    @staticmethod
    def wrap_imgs_with_plt(imgs: np.ndarray, title=None, subtitles=None, if_color_bar=False, clim=None, ticks=None, dpi=180):
        figure = plt.figure()
        if title is not None:
            figure.suptitle(title, fontsize=16)
        figure.tight_layout()
        assert imgs.ndim == 4
        N, H, W, C = imgs.shape
        if C != 1:
            if_color_bar = False
            clim = None
            ticks = None

        for i in range(N):
            plot = figure.add_subplot(1, N, i+1)
            if subtitles is not None:
                plot.title.set_text(subtitles[i])
                plot.axis("off")
                plot_img = plot.imshow(imgs[i])
                if clim is not None and (isinstance(clim, tuple) or isinstance(clim, list)):
                    plot_img.set_clim(*clim)
                if if_color_bar:
                    plot.colorbar(plot_img, ticks=ticks)
        img = ImageHelper.plt_figure_to_opencv_array(figure, dpi=dpi)
        plt.close(figure)
        return img

    @staticmethod
    def wrap_img_with_plt(img: np.ndarray, title=None, if_color_bar=False, ticks=None, clim=None, dpi=180):
        figure = plt.figure()
        figure.tight_layout()

        plot = figure.add_subplot(1, 1, 1)
        plot.axis("off")
        if title is not None:
            figure.suptitle(title, fontsize=16)

        plt_img = plot.imshow(img)
        if clim is not None:
            plt_img.set_clim(*clim)
        if if_color_bar:
            figure.colorbar(plt_img, ticks=ticks)

        ret = ImageHelper.plt_figure_to_opencv_array(figure, dpi=dpi)
        plt.close(figure)
        return ret

    @staticmethod
    def resize(image: np.ndarray, dsize: tuple):
        _, _, = image.shape[: 2]
        assert isinstance(dsize, tuple)
        assert len(dsize) == 2
        image = cv2.resize(image, dsize)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        return image

    @staticmethod
    def standardize(image: np.ndarray, mean=None, std=None):
        if mean is None:
            mean = np.mean(image)
        if std is None:
            std = np.std(image)
        return (image - mean) / std

    @staticmethod
    def resize_deep_channel(img: np.ndarray, dsize: tuple, splitting_size=3):
        W, H = dsize
        C = img.shape[-1]
        ret = np.zeros((H, W, C), dtype=img.dtype)
        num_of_splitting = C // splitting_size
        if C % splitting_size != 0:
            num_of_splitting += 1
        for i in range(num_of_splitting):
            start_i = i * splitting_size
            end_i = start_i + splitting_size
            resized = cv2.resize(img[:, :, start_i: end_i], dsize=dsize)
            if resized.ndim < 3:
                resized = np.expand_dims(resized, -1)
            ret[:, :, start_i: end_i] = resized
        return ret

    @staticmethod
    def normalize_hu(image, dtype=np.float32):
        image_clim = (-150, 350.)

        image = np.clip(image, image_clim[0], image_clim[1])

        image = (image - image_clim[0]) / (image_clim[1] - image_clim[0])  # [0, 1]
        image *= 255.
        image = image.astype(dtype)

        return image  # [0, 255]

    @staticmethod
    def vol_to_video_shape(vol: np.ndarray) -> np.ndarray:
        re: np.ndarray = vol
        if vol.ndim == 3:  # (N, H, W)
            re = np.expand_dims(vol, axis=-1)  # (N, H, W, 1)
        if re.ndim == 4: # (N, H, W, C)
            C = re.shape[-1]
            if C < 3:
                assert C == 1
                re = np.tile(re, (1, 1, 1, 3))
            assert re.shape[-1] == 3
        return re

    @staticmethod
    def gray2color(gray_array, color_map):

        rows, cols = gray_array.shape
        color_array = np.zeros((rows, cols, 3), np.uint8)

        for i in range(0, rows):
            for j in range(0, cols):
                color_array[i, j] = color_map[gray_array[i, j]]
        return color_array








