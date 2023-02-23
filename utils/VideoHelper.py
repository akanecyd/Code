import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from .ImageHelper import ImageHelper
from tqdm import tqdm


class VideoHelper:
    @staticmethod
    def wrap_video_with_plt(video, title=None, if_color_bar=False, ticks=None, clim=None, dpi=300):
        """

        :param ticks:
        :param if_color_bar:
        :param video: (N, H, W, C)
        :param title:
        :param dpi:
        :return:
        """
        ret = []
        video_length = video.shape[0]
        for i in range(video_length):
            print("wrapping ... {}/{}".format(i + 1, video_length))
            frame = video[i]
            frame_title = None
            if title is not None:
                frame_title = "{0}_{1:03d}".format(title, i)
            ret.append(ImageHelper.wrap_img_with_plt(
                frame, frame_title, if_color_bar=if_color_bar, ticks=ticks, clim=clim, dpi=dpi))

        return np.asarray(ret)


    @staticmethod
    def write_vol_to_video(
            vol: np.ndarray,
            case_name: str,
            output_path: str,
            if_reverse: bool = False,
    ):
        """

        :param vol:  dtype = np.uint8
        :param case_name:
        :param output_path:
        :param if_reverse:
        :return:
        """
        assert vol.dtype == np.uint8
        vol = ImageHelper.vol_to_video_shape(vol.copy())

        N, H, W, _ = vol.shape  # (N, H, W 3)


        video_label = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
        for j in tqdm(reversed(range(N)) if if_reverse else range(N), desc="write {}".format(output_path)):
            slice = vol[j]  # (H, W, 3)
            if case_name is None or case_name != "":
                cv2.putText(
                    slice,
                    '{0:03d}_{1}'.format(j, case_name),
                    (20, 20),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.3,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA
                )
            video_label.write(slice)
        video_label.release()
        print(output_path, "wrote.")

    @staticmethod
    def combine_video(
            input_file1_path: str,
            input_file2_path: str,
            output_file_path: str):


        reader1 = cv2.VideoCapture(input_file1_path)
        reader2 = cv2.VideoCapture(input_file2_path)
        width = int(reader1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_file_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 30,  # fps
                                 (width, height // 2))  # resolution

        print(reader1.isOpened())
        print(reader2.isOpened())
        have_more_frame = True
        c = 0
        while have_more_frame:
            have_more_frame, frame1 = reader1.read()
            _, frame2 = reader2.read()
            frame1 = cv2.resize(frame1, (width // 2, height // 2))
            frame2 = cv2.resize(frame2, (width // 2, height // 2))
            img = np.hstack((frame1, frame2))
            cv2.waitKey(1)
            writer.write(img)
            c += 1
            print(str(c) + ' is ok')

        writer.release()
        reader1.release()
        cv2.destroyAllWindows()

    @staticmethod
    def layout_videos(
            videos: [np.ndarray, ...],
            titles: [str, ...],
            case_name,
            color_bar_mask=None,
            tickses=None,
            clims=None,
    ):

        video_length = videos[0].shape[0]
        video_num = len(videos)

        if clims is None:
            clims = [None] * video_num
        if tickses is None:
            tickses = [None] * video_num

        if color_bar_mask is None:
            color_bar_mask = [False] * video_num
        new_video = []
        for i in range(video_length):
            print("combine {}/{}".format(i+1, video_length))
            frames = [video[i] for video in videos]
            for j, frame in enumerate(frames):
                if frame.shape[-1] == 3:
                    frames[j] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    assert frame.shape[-1] == 1

            figure = plt.figure()
            figure.suptitle('{0}_{1:03d}'.format(case_name, i), fontsize=16)

            figure.tight_layout()
            plt.tight_layout()
            for j in range(video_num):    # TODO: replace with ImageHelper.wrap_imgs_with_plt
                if_color_bar = color_bar_mask[j]
                title = titles[j]
                clim = clims[j]
                ticks = tickses[j]
                plot = figure.add_subplot(1, video_num, j+1)
                plot.title.set_text(title)
                plot.axis("off")
                plot_img = plot.imshow(frames[j])
                if clim is not None and (isinstance(clim, tuple) or isinstance(clim, list)):
                    plot_img.set_clim(*clim)
                if if_color_bar:
                    plot.colorbar(plot_img, ticks=ticks)

            img = ImageHelper.plt_figure_to_opencv_array(figure)
            new_video.append(img)
            plt.close(figure)

        new_video = np.asarray(new_video)  # (N, H, W, C)
        return new_video





