import os
import logging
import subprocess
import dtlpy as dl
from pathlib import Path

from models.model_adapter import NvidiaBase

logger = logging.getLogger('[DashCamNet]')


class DashCamNet(NvidiaBase):

    def get_cmd(self):
        return [
            f'detectnet_v2 inference '
            f'-e {os.path.join(os.path.dirname(str(__file__)), "inference_spec")} '
            f'-i {self.images_path} '
            f'-r {self.res_dir} '
            f'-k {self.model_key}']
