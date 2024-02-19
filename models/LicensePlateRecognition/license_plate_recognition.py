import os
import logging
import subprocess
import dtlpy as dl
import shutil

logger = logging.getLogger('[LPRNet]')


class LPRNet:
    def __init__(self):
        self.name = "lpr-net"
        self.key = 'nvidia_tlt'
        self.res_dir = os.path.join(os.getcwd(), 'lpr_res')
        self.model_download_version = "nvidia/tao/lprnet:trainable_v1.0"
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

        # if not os.path.isfile("/tmp/tao_models/lprnet_vtrainable_v1.0/us_lprnet_baseline18_trainable.tlt"):
        #     raise Exception("Failed loading the model")

        src_filepath = os.path.join(self.current_dir, 'us_lp_characters.txt')
        dst_filepath = os.path.join(dest_path, 'lprnet_vtrainable_v1.0, us_lp_characters.txt')
        shutil.copyfile(src_filepath, dst_filepath)

        # shutil.copyfile(f'{os.getcwd()}/models/LicensePlateRecognition/us_lp_characters.txt',
        #                 f'/tmp/tao_models/lprnet_vtrainable_v1.0/us_lp_characters.txt')

    def detect(self, images_dir):
        ret = list()
        with os.popen(
            f'lprnet inference '
            f'-e {os.getcwd()}/models/LicensePlateRecognition/lprnet_spec.txt '
            f'-i {images_dir} '
            f'-r {os.getcwd()}/{self.res_dir} '
            f'-k {self.key} '
            f'-m /tmp/tao_models/lprnet_vtrainable_v1.0/us_lprnet_baseline18_trainable.tlt'
        ) as f:
            output_lines = f.readlines()
            # logger.info(f"Full Model Output:\n{''.join(output_lines)}")

        res_lines = [res_line for res_line in output_lines if
                     res_line.startswith(tuple(os.listdir(images_dir)))]
        results = {res.split(':')[0]: res.split(':')[1] for res in res_lines}
        for image_path in os.listdir(images_dir):
            image_annotations = dl.AnnotationCollection()
            image_annotations.add(
                annotation_definition=dl.Note(
                    left=0,
                    top=0,
                    right=100,
                    bottom=100,
                    label=results[image_path]
                ),
                model_info={
                    'name': self.name,
                    'confidence': 0.5
                }
            )
            ret.append(image_annotations)
            # logger.info(f'Full Annotation Result: {results[image_path]}')
        return ret
