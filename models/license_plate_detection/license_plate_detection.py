import os
import logging
from models.model_adapter import NvidiaBase
import dtlpy as dl

logger = logging.getLogger('[FaceDetectIR]')


class LPDNet(NvidiaBase):

    def get_cmd(self):
        tlt_filepath = os.path.join('/tmp', 'tao_models', 'lpdnet_vunpruned_v2.1', 'yolov4_tiny_usa_trainable.tlt')
        return [
            f'detectnet_v2 inference '
            f'-e {os.path.join(os.getcwd(), "yolo_v4_tiny_retrain_kitti.txt")} '
            f'-i {self.images_path} '
            f'-r {self.res_dir} '
            f'-k {self.model_key}'
            f'-m {tlt_filepath}'
        ]
