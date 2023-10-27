import os
import logging
import dtlpy as dl
from pathlib import Path
import shutil
try:
    from ..tao_model import TaoModel
except Exception:
    from tao_model import TaoModel

logger = logging.getLogger('[LPRNet]')


class LPRNet(TaoModel):
    def __init__(self):
        super().__init__()
        self.key = 'nvidia_tlt'
        self.res_dir = 'lpr_res'
        os.mkdir(self.res_dir)

        # download model - the txt config file points to this location for the model
        os.system(
            'ngc registry model download-version "nvidia/tao/lprnet:trainable_v1.0" --dest /tmp/tao_models/')

        if not os.path.isfile("/tmp/tao_models/lprnet_vtrainable_v1.0/us_lprnet_baseline18_trainable.tlt"):
            raise Exception("Failed loading the model")

        shutil.copyfile(f'{os.getcwd()}/LicensePlateRecognition/us_lp_characters.txt',
                        '/tmp/tao_models/lprnet_vtrainable_v1.0/us_lp_characters.txt')

    def detect(self, image_path):
        # TODO: Change "image_path" to "images_dir". this model crashes with a single image.
        ret = []
        try:
            with os.popen(
                    f'lprnet inference -e {os.getcwd()}/LicensePlateRecognition/lprnet_spec.txt '
                    f'-i {image_path} -r {os.getcwd()}/{self.res_dir} -k {self.key} '
                    f'-m /tmp/tao_models/lprnet_vtrainable_v1.0/us_lprnet_baseline18_trainable.tlt') as f:
                output_lines = f.readlines()
                logger.info(f"Full Model Output:\n{''.join(output_lines)}")
                res_lines = [res_line for res_line in output_lines if res_line.startswith(image_path)]

                # results = [res.split(':')[1] for res in res_lines]
                # dl.Text(...)

            return ret
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    @staticmethod
    def get_name():
        return "lpr-net"

    @staticmethod
    def get_labels():
        return []

    @staticmethod
    def get_output_type():
        return dl.AnnotationType.TEXT
