import os
import logging
import dtlpy as dl
from pathlib import Path
from ..tao_model import TaoModel

logger = logging.getLogger('[TrafficCamNet]')


class TrafficCamNet(TaoModel):
    def __init__(self):
        super().__init__()
        self.key = 'tlt_encode'
        self.res_dir = 'res'
        os.mkdir(self.res_dir)

    def detect(self, image_path):
        ret = []
        try:
            with os.popen(
                    f'detectnet_v2 inference -e /model/inference_spec.txt -i {image_path} -o {os.getcwd()}/{self.res_dir} -k {self.key}') as f:
                output = f.read().strip()
            logger.info(f"Full Model Output:\n{output}")
            with open(f'res/labels/{Path(image_path).stem}.txt', 'r') as f:
                for line in f.readlines():
                    vals = line.split(' ')
                    if vals[0] == 'car':
                        ret.append(dl.Box(label='car', top=vals[5], left=vals[4], bottom=vals[7], right=vals[6]))
                        logger.info(f'detected [left, top, bottom, right]: {vals[4:8]}')
            return ret
        except Exception as e:
            logger.error(f"Error: {e}")

    @staticmethod
    def get_name():
        return "traffic-cam-net"

    @staticmethod
    def get_labels():
        return ['car']

    @staticmethod
    def get_output_type():
        return dl.AnnotationType.BOX
