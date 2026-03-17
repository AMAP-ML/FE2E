import os
import json
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class DrivingStereoDataset(MonoDataset):
    RAW_HEIGHT = 800
    RAW_WIDTH = 1762

    def __init__(self, *args, **kwargs):
        super(DrivingStereoDataset, self).__init__(*args, **kwargs)
        self.forename = {"rainy": "2018-08-17-09-45-58_2018-08-17-10-", "foggy": "2018-10-25-07-37-26_2018-10-25-", "sunny": "2018-10-19-09-30-39_2018-10-19-", "cloudy": "2018-10-31-06-55-01_2018-10-31-"}

    def get_color(self, weather, name, do_flip):
        path, name = self.get_image_path(weather, name)
        color = self.loader(path)

        return color, name

    def get_image_path(self, weather, frame_name):
        folder = "left-image-full-size"
        image_path = os.path.join(self.opts.data_path, weather, folder, frame_name)
        image_name = os.path.join(weather, folder, frame_name)
        if self.opts.debug >= 3:
            print(image_name)
        return image_path, image_name

    def index_to_name(self, weather, index):
        return self.forename[weather] + self.filenames[index] + ".png"
