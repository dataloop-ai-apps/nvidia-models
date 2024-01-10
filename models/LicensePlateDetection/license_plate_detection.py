import os
import logging
import subprocess
import dtlpy as dl
from pathlib import Path
try:
    from ..tao_model import TaoModel
except Exception:
    from tao_model import TaoModel

logger = logging.getLogger('[LPDNet]')


class LPDNet(TaoModel):
    def __init__(self, **model_config):
        super().__init__(**model_config)
        self.key = 'nvidia_tlt'
        self.res_dir = 'lpd_res'
        os.makedirs(self.res_dir, exist_ok=True)

        # download model - the txt config file points to this location for the model
        subprocess.Popen(['/tmp/ngccli/ngc-cli/ngc registry model download-version "nvidia/tao/lpdnet:unpruned_v2.1" --dest /tmp/tao_models/'],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True).wait()

        if not os.path.isfile("/tmp/tao_models/lpdnet_vunpruned_v2.1/yolov4_tiny_usa_trainable.tlt"):
            raise Exception("Failed loading the model")

    def detect(self, images_dir):
        ret = []
        try:
            with os.popen(
                    f'yolo_v4_tiny inference '
                    f'-e {os.getcwd()}/models/LicensePlateDetection/yolo_v4_tiny_retrain_kitti.txt '
                    f'-i {images_dir} '
                    f'-r {os.getcwd()}/{self.res_dir} '
                    f'-k {self.key} '
                    f'-m /tmp/tao_models/lpdnet_vunpruned_v2.1/yolov4_tiny_usa_trainable.tlt') as f:
                output = f.read().strip()
            logger.info(f"Full Model Output:\n{output}")

            for image_path in os.listdir(images_dir):
                image_annotations = dl.AnnotationCollection()
                with open(f'{self.res_dir}/labels/{Path(image_path).stem}.txt', 'r') as f:
                    for line in f.readlines():
                        vals = line.split(' ')
                        if vals[0] == 'lpd':
                            image_annotations.add(
                                annotation_definition=dl.Box(label='lpd', top=vals[5], left=vals[4], bottom=vals[7],
                                                             right=vals[6]),
                                model_info={
                                    'name': self.get_name(),
                                    'confidence': float(vals[16])
                                })
                            logger.info(f'detected [left, top, bottom, right]: {vals[4:8]}')
                ret.append(image_annotations)
            return ret
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    @staticmethod
    def get_name():
        return "lpd-net"

    @staticmethod
    def get_labels():
        return ['lpd']

    @staticmethod
    def get_output_type():
        return dl.AnnotationType.BOX
