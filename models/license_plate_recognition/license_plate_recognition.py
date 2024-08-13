import os
import shutil
import logging
import dtlpy as dl
from pathlib import Path
from models.model_adapter import NvidiaBase

logger = logging.getLogger('[LPRNet]')


class LPRNet(NvidiaBase):

    def get_cmd(self):
        tlt_filepath = os.path.join('/tmp', 'tao_models', 'lprnet_vtrainable_v1.0', 'us_lprnet_baseline18_trainable.tlt')
        return [
            f'lprnet inference '
            f'-e {os.path.join(Path(__file__).parent.absolute(), "lprnet_spec.txt")}  '
            f'-i {self.images_path} '
            f'-r {self.res_dir} '
            f'-k nvidia_tlt '
            f'-m {tlt_filepath}'
            ]

    # def predict(self, batch, **kwargs):
    #     try:
    #         logger.info('predicting batch of size: {}'.format(len(batch)))
    #         logger.info(f'batch = {batch}')
    #
    #         os.mkdir(self.images_path)
    #         for i, item in enumerate(batch):
    #             logger.info(f'item = {item}')
    #             item.download(local_path=self.images_path)
    #
    #         os.makedirs(self.res_dir, exist_ok=True)
    #         cmd = self.get_cmd()
    #         with os.popen(cmd[0]) as f:
    #             output_lines = f.readlines()
    #         output_lines = [line[len(self.images_path) + 1:-2] for line in output_lines if
    #                         line.startswith(self.images_path)]
    #         outputs = {line.split(":")[0]: line.split(":")[1] for line in output_lines}
    #
    #         annotations_batch = list()
    #         for image_path in os.listdir(self.images_path):
    #             image_annotations = dl.AnnotationCollection()
    #             result = outputs[image_path]
    #             image_annotations.add(
    #                 annotation_definition=dl.Classification(
    #                     label=result
    #                     ),
    #                 model_info={
    #                     'name': self.model_name,
    #                     'confidence': 1.0
    #                     }
    #                 )
    #             annotations_batch.append(image_annotations)
    #
    #     finally:
    #         shutil.rmtree(self.images_path)
    #     return annotations_batch
