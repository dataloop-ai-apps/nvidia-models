import os
import logging
import subprocess
import dtlpy as dl
from pathlib import Path

logger = logging.getLogger('[PeopleNet]')


class PeopleNet:
    def __init__(self):
        self.key = 'tlt_encode'
        self.res_dir = 'peoplenet_res'
        os.makedirs(self.res_dir, exist_ok=True)
        # download model - the txt config file points to this location for the model
        logger.info("Downloading model artifacts")

        subprocess.Popen([
            '/tmp/ngccli/ngc-cli/ngc registry model download-version "nvidia/tao/peoplenet:trainable_v2.5" --dest /tmp/tao_models/'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, shell=True).wait()
        if not os.path.isfile("/tmp/tao_models/peoplenet_vtrainable_v2.5/resnet34_peoplenet.tlt"):
            raise Exception("Failed loading the model")

    def detect(self, images_dir):
        ret = []
        try:
            logger.info(f"Running detectnet_v2 inference on {images_dir}, Content {os.listdir(images_dir)}")
            os.makedirs(f'{os.getcwd()}/{self.res_dir}', exist_ok=True)
            with os.popen(
                    f'detectnet_v2 inference '
                    f'-e {os.getcwd()}/models/PeopleNet/inference_spec.txt '
                    f'-i {images_dir} '
                    f'-r {os.getcwd()}/{self.res_dir} '
                    f'-k {self.key}') as f:
                output = f.read().strip()
            logger.info(f"Full Model Output:\n{output}")

            for image_path in os.listdir(images_dir):
                image_annotations = dl.AnnotationCollection()
                with open(f'{os.getcwd()}/{self.res_dir}/labels/{Path(image_path).stem}.txt', 'r') as f:
                    for line in f.readlines():
                        vals = line.split(' ')
                        if vals[0] in self.get_labels():
                            image_annotations.add(
                                annotation_definition=dl.Box(
                                    label=vals[0],
                                    top=vals[5],
                                    left=vals[4],
                                    bottom=vals[7],
                                    right=vals[6]
                                ),
                                model_info={
                                    'name': self.get_name(),
                                    'confidence': 0.5
                                }
                            )
                            logger.info(f'detected [left, top, bottom, right]: {vals[4:8]}')
                            logger.info(f'Full Annotation Result: {vals}')
                ret.append(image_annotations)
            return ret
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    @staticmethod
    def get_name():
        return "people-net"

    @staticmethod
    def get_labels():
        return ["person", "bag", "face"]
