import os
import shutil
import logging
import dtlpy as dl
from pathlib import Path
from models.model_adapter import NvidiaBase

logger = logging.getLogger('[LPRNet]')


class LPRNet(NvidiaBase):

    def get_cmd(self):
        tlt_filepath = os.path.join('/tmp', 'tao_models', 'lprnet_vtrainable_v1.0', 'us_lprnet_baseline18_trainable.tlt')
        return [
            f'lprnet inference '
            f'-e {os.path.join(Path(__file__).parent.absolute(), "lprnet_spec.txt")}  '
            f'-i {self.images_path} '
            f'-r {self.res_dir} '
            f'-k nvidia_tlt '
            f'-m {tlt_filepath}'
            ]
