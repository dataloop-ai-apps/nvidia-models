import os
import logging
from models.model_adapter import NvidiaBase
import dtlpy as dl
from pathlib import Path

logger = logging.getLogger('[DashCamNet]')


class DashCamNet(NvidiaBase):

    def get_cmd(self):
        return [
            'detectnet_v2', 'inference',
            '-e', str(os.path.join(Path(__file__).parent.absolute(), "inference_spec.txt")),
            '-i', self.images_path,
            '-r', self.res_dir,
            '-k', self.model_key
        ]
