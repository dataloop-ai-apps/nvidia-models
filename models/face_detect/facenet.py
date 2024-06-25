import os
import logging
import subprocess
import dtlpy as dl
from pathlib import Path

logger = logging.getLogger('[FaceNet]')


class FaceNet:
    def __init__(self):
        self.model_name = "face-net"
        self.model_key = 'tlt_encode'
        self.res_dir = os.path.join(os.getcwd(), 'facenet_res')
        self.model_version = "nvidia/tao/facenet:unpruned_v2.0"
        self.current_dir = os.path.dirname(str(__file__))

        # download model - the txt config file points to this location for the model
        # logger.info("Downloading model artifacts")

        cli_filepath = os.path.join('/tmp', 'ngccli', 'ngc-cli', 'ngc')
        dest_path = os.path.join('/tmp', 'tao_models')
        download_status = subprocess.Popen(
            [f'{cli_filepath} registry model download-version "{self.model_version}" --dest {dest_path}'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        download_status.wait()
        if download_status.returncode != 0:
            (stdout, stderr) = download_status.communicate()
            logger.info(f'STDOUT:\n{stdout}')
            logger.info(f'STDERR:\n{stderr}')
            raise Exception(f'Failed running nvidia cli command: "{' '.join(cmd)}". more logs above')

        # if not os.path.isfile("/tmp/tao_models/facenet_vunpruned_v2.0/model.tlt"):
        #     raise Exception("Failed loading the model")

    def detect(self, images_dir):
        # TODO: Find the correct model to load model.tlt
        ret = list()
        logger.info(f"Running detectnet_v2 inference on {images_dir}, Content {os.listdir(images_dir)}")

        specs_filepath = os.path.join(self.current_dir, "inference_spec.txt")
        os.makedirs(self.res_dir, exist_ok=True)
        predict_status = subprocess.Popen([
            f'detectnet_v2 inference '
            f'-e {specs_filepath} '
            f'-i {images_dir} '
            f'-r {self.res_dir} '
            f'-k {self.model_key}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        predict_status.wait()
        if predict_status.returncode != 0:
            (out, err) = predict_status.communicate()
            raise Exception(f'Failed loading the model: {err}')

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
                            'name': self.model_name,
                            'confidence': float(vals[-1]) / 100
                        }
                    )
                    # logger.info(f'detected [left, top, bottom, right]: {vals[4:8]}')
                    # logger.info(f'Full Annotation Result: {vals}')
            ret.append(image_annotations)
        return ret