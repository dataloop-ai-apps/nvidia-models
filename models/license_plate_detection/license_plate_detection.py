import os
import logging
from models.model_adapter import NvidiaBase
from pathlib import Path

logger = logging.getLogger('[FaceDetectIR]')


class LPDNet(NvidiaBase):

    def get_cmd(self):
        tlt_filepath = os.path.join('/tmp', 'tao_models', 'lpdnet_vunpruned_v2.1', 'yolov4_tiny_usa_trainable.tlt')
        return [
            f'yolo_v4_tiny inference '
            f'-e {os.path.join(Path(__file__).parent.absolute(), "yolo_v4_tiny_retrain_kitti.txt")} '
            f'-i {self.images_path} '
            f'-r {self.res_dir} '
            f'-k {self.model_key} '
            f'-m {tlt_filepath}'
        ]
