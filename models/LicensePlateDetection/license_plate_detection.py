import os
import logging
import subprocess
import dtlpy as dl
from pathlib import Path

logger = logging.getLogger('[LPDNet]')


class LPDNet:
    def __init__(self):
        self.name = "lpd-net"
        self.key = 'nvidia_tlt'
        self.res_dir = os.path.join(os.getcwd(), 'lpd_res')
        self.model_download_version = "nvidia/tao/lpdnet:unpruned_v2.1"
        self.current_dir = os.path.dirname(str(__file__))

        # download model - the txt config file points to this location for the model
        # logger.info("Downloading model artifacts")

        cli_filepath = os.path.join('/tmp', 'ngccli', 'ngc-cli', 'ngc')
        dest_path = os.path.join('/tmp', 'tao_models')
        download_status = subprocess.Popen(
            [f'{cli_filepath} registry model download-version "{self.model_download_version}" --dest {dest_path}'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        download_status.wait()
        if download_status.returncode != 0:
            (out, err) = download_status.communicate()
            raise Exception(f'Failed loading the model: {err}')

        # if not os.path.isfile("/tmp/tao_models/lpdnet_vunpruned_v2.1/yolov4_tiny_usa_trainable.tlt"):
        #     raise Exception("Failed loading the model")

    def detect(self, images_dir):
        ret = list()

        specs_filepath = os.path.join(self.current_dir, "yolo_v4_tiny_retrain_kitti.txt")
        tlt_filepath = os.path.join('/tmp', 'tao_models', 'lpdnet_vunpruned_v2.1', 'yolov4_tiny_usa_trainable.tlt')
        os.makedirs(self.res_dir, exist_ok=True)
        with os.popen(
            f'yolo_v4_tiny inference '
            f'-e {specs_filepath} '
            f'-i {images_dir} '
            f'-r {self.res_dir} '
            f'-k {self.key} '
            f'-m {tlt_filepath}'
        ) as f:
            output = f.read().strip()
            # logger.info(f"Full Model Output:\n{output}")

        for image_path in os.listdir(images_dir):
            image_annotations = dl.AnnotationCollection()
            output_filepath = os.path.join(self.res_dir, "labels", f"{Path(image_path).stem}.txt")
            with open(output_filepath, 'r') as f:
                for line in f.readlines():
                    vals = line.split(' ')
                    image_annotations.add(
                        annotation_definition=dl.Box(
                            label=vals[0],
                            top=vals[5],
                            left=vals[4],
                            bottom=vals[7],
                            right=vals[6]
                        ),
                        model_info={
                            'name': self.name,
                            'confidence': float(vals[-1])
                        }
                    )
                    # logger.info(f'detected [left, top, bottom, right]: {vals[4:8]}')
                    # logger.info(f'Full Annotation Result: {vals}')
            ret.append(image_annotations)
        return ret
