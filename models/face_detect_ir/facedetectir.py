import os
import logging
from models.model_adapter import NvidiaBase
import dtlpy as dl

logger = logging.getLogger('[FaceDetectIR]')


class FaceDetectIR(NvidiaBase):
    def __init__(self, ngc_api_key_secret_name, ngc_org_secret_name, model_entity: dl.Model = None):
        super(FaceDetectIR, self).__init__(ngc_api_key_secret_name, ngc_org_secret_name, model_entity)

    def get_cmd(self):
        return [
            f'detectnet_v2 inference '
            f'-e {os.path.join(os.getcwd(), "inference_spec.txt")} '
            f'-i {self.images_path} '
            f'-r {self.res_dir} '
            f'-k {self.model_key}']
