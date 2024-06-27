import os
import logging
from models.model_adapter import NvidiaBase
from pathlib import Path

logger = logging.getLogger('[TrafficCamNet]')


class TrafficCamNet(NvidiaBase):
    def get_cmd(self):
        return [
            f'detectnet_v2 inference '
            f'-e {os.path.join(Path(__file__).parent.absolute(), "inference_spec.txt")} '
            f'-i {self.images_path} '
            f'-r {self.res_dir} '
            f'-k {self.model_key}']
