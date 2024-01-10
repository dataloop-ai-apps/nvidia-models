import os
import logging
import subprocess
import dtlpy as dl
from pathlib import Path
import urllib.request

try:
    from ..tao_model import TaoModel
except Exception:
    from tao_model import TaoModel

logger = logging.getLogger('[FaceDetect]')


class FaceDetect(TaoModel):
    def __init__(self, **model_config):
        super().__init__(**model_config)
        self.key = 'tlt_encode'
        self.res_dir = 'facedetect_res'
        os.makedirs(self.res_dir, exist_ok=True)
        # download model - the txt config file points to this location for the model
        subprocess.Popen([
            '/tmp/ngccli/ngc-cli/ngc registry model download-version "nvidia/tao/facedetect:unpruned_v2.0" --dest /tmp/tao_models/'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, shell=True).wait()
        if not os.path.isfile("/tmp/tao_models/facedetect_vunpruned_v2.0/model.tlt"):
            raise Exception("Failed loading the model")

    def detect(self, images_dir):
        ret = []
        try:
            logger.info(f"Running detectnet_v2 inference on {images_dir}, Content {os.listdir(images_dir)}")
            os.makedirs(f'{os.getcwd()}/{self.res_dir}', exist_ok=True)
            with os.popen(
                    f'detectnet_v2 inference '
                    f'-e {os.getcwd()}/models/FaceDetect/inference_spec.txt '
                    f'-i {images_dir} '
                    f'-r {os.getcwd()}/{self.res_dir} '
                    f'-k {self.key}') as f:
                output = f.read().strip()
            logger.info(f"Full Model Output:\n{output}")
            for image_path in os.listdir(images_dir):
                image_annotations = dl.AnnotationCollection()
                logger.info(f"**** res dir {os.getcwd()}/{self.res_dir}")
                logger.info(f"**** res dir content {os.listdir(f'{os.getcwd()}/{self.res_dir}')}")
                with open(f'{os.getcwd()}/{self.res_dir}/labels/{Path(image_path).stem}.txt', 'r') as f:
                    for line in f.readlines():
                        vals = line.split(' ')
                        if vals[0] == 'face':
                            image_annotations.add(
                                annotation_definition=dl.Box(label='face', top=vals[5], left=vals[4], bottom=vals[7],
                                                             right=vals[6]),
                                model_info={
                                    'name': self.get_name(),
                                    'confidence': 0.5
                                })
                            logger.info(f'detected [left, top, bottom, right]: {vals[4:8]}')
                ret.append(image_annotations)
            return ret
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    @staticmethod
    def get_name():
        return "face-detect"

    @staticmethod
    def get_labels():
        return ['face']

    @staticmethod
    def get_output_type():
        return dl.AnnotationType.BOX
