import os
import logging
import subprocess
import dtlpy as dl
import shutil

logger = logging.getLogger('[LPRNet]')


class LPRNet:
    def __init__(self, **model_config):
        super().__init__(**model_config)
        self.key = 'nvidia_tlt'
        self.res_dir = 'lpr_res'
        os.makedirs(self.res_dir, exist_ok=True)

        # download model - the txt config file points to this location for the model
        subprocess.Popen(['/tmp/ngccli/ngc-cli/ngc registry model download-version "nvidia/tao/lprnet:trainable_v1.0" --dest /tmp/tao_models/'],
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True).wait()

        if not os.path.isfile("/tmp/tao_models/lprnet_vtrainable_v1.0/us_lprnet_baseline18_trainable.tlt"):
            raise Exception("Failed loading the model")

        shutil.copyfile(f'{os.getcwd()}/models/LicensePlateRecognition/us_lp_characters.txt',
                        '/tmp/tao_models/lprnet_vtrainable_v1.0/us_lp_characters.txt')

    def detect(self, images_dir):
        ret = []
        try:
            with os.popen(
                    f'lprnet inference '
                    f'-e {os.getcwd()}/models/LicensePlateRecognition/lprnet_spec.txt '
                    f'-i {images_dir} '
                    f'-r {os.getcwd()}/{self.res_dir} '
                    f'-k {self.key} '
                    f'-m /tmp/tao_models/lprnet_vtrainable_v1.0/us_lprnet_baseline18_trainable.tlt') as f:
                output_lines = f.readlines()
                logger.info(f"Full Model Output:\n{''.join(output_lines)}")

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
                            'name': self.get_name(),
                            'confidence': 0.5
                        })
                    ret.append(image_annotations)

            return ret
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    @staticmethod
    def get_name():
        return "lpr-net"

    @staticmethod
    def get_labels():
        return []
