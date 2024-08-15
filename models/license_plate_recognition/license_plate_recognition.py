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

    def parse_results(self, predict_status):
        # Read outputs
        output_lines = predict_status.stdout.readlines()
        parsed_outputs = dict()
        for line in output_lines:
            line = line.decode('utf-8')
            if line.startswith(self.images_path):
                image_filepath, result = line.strip().split(":")
                output_filename = f"{Path(image_filepath).stem}.txt"
                output_results = parsed_outputs.get(output_filename, list())
                output_results.append(result)
                parsed_outputs.update({output_filename: output_results})

        # Write outputs
        os.makedirs(os.path.join(self.res_dir, "labels"), exist_ok=True)
        for output_filename, output_results in parsed_outputs.items():
            output_filepath = os.path.join(self.res_dir, "labels", output_filename)
            with open(output_filepath, 'w') as f:
                output_results = "\n".join(output_results)
                f.write(output_results)
