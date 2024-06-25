import os
import logging
import shutil
import subprocess
from pathlib import Path
import dtlpy as dl

logger = logging.getLogger('[Nvidia Models]')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for NVIDIA TAO models',
                              init_inputs={
                                  'ngc_api_key_secret_name': 'String',
                                  'ngc_org_secret_name': 'String',
                                  'model_entity': dl.Model
                              })
class NvidiaBase(dl.BaseModelAdapter):
    def __init__(self, ngc_api_key_secret_name, ngc_org_secret_name, model_entity: dl.Model = None):
        self.images_path = None
        self.tao_model = None
        self.cmd = None
        self.ngc_config = {
            "ngc_api_key": os.environ.get(ngc_api_key_secret_name),
            "ngc_org": os.environ.get(ngc_org_secret_name),
        }
        super(NvidiaBase, self).__init__(model_entity)

    def get_cmd(self):
        raise NotImplementedError("Please implement 'get_cmd' method in {}".format(self.__class__.__name__))

    def load(self, local_path, **kwargs):
        model_name = self.model_entity.configuration.get("dash-cam-net")
        model_key = self.model_entity.configuration.get("key")
        model_version = self.model_entity.configuration("model_version")

        # Remove "downloading ngc" section when there is a working image with ngc-cli
        logger.info('downloading ngc')
        os.makedirs('/tmp/ngccli', exist_ok=True)
        with os.popen(f'wget "https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip" -P /tmp/ngccli') as f:
            output = f.read().strip()
        print("WGET:\n" + output)
        with os.popen(f'unzip -u /tmp/ngccli/ngccli_cat_linux.zip -d /tmp/ngccli/') as f:
            output = f.read().strip()
        print("UNZIP:\n" + output)
        os.environ["PATH"] = "/tmp/ngccli/ngc-cli:{}".format(os.getenv("PATH", ""))

        logger.info('login to ngc')
        process = subprocess.Popen(['/tmp/ngccli/ngc-cli/ngc config set'],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=True)
        input_data = (
                self.ngc_config["ngc_api_key"].encode() + b'\n\n' +
                self.ngc_config["ngc_org"].encode() + b'\n\n\n'
        )
        process.communicate(input=input_data)
        os.makedirs('/tmp/tao_models', exist_ok=True)

        logger.info('loading model')
        self.images_path = os.path.join(os.getcwd(), 'images')

        self.model_name = model_name
        self.model_key = model_key
        self.model_version = model_version
        self.res_dir = os.path.join(os.getcwd(), 'results')

        # download model - the txt config file points to this location for the model
        # logger.info("Downloading model artifacts")

        cli_filepath = os.path.join('/tmp', 'ngccli', 'ngc-cli', 'ngc')
        dest_path = os.path.join('/tmp', 'tao_models')
        cmd = [f'{cli_filepath} registry model download-version "{self.model_version}" --dest {dest_path}']
        download_status = subprocess.Popen(cmd,
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
            raise Exception(f'Failed downloading cli command: {" ".join(cmd)}. more logs above')
        if os.path.isdir(self.images_path):
            shutil.rmtree(self.images_path)

    def predict(self, batch, **kwargs):
        try:

            logger.info('predicting batch of size: {}'.format(len(batch)))
            logger.info(f'batch = {batch}')

            os.mkdir(self.images_path)
            for i, item in enumerate(batch):
                logger.info(f'item = {item}')
                item.download(local_path=self.images_path)

            os.makedirs(self.res_dir, exist_ok=True)
            cmd = self.get_cmd()
            predict_status = subprocess.Popen(cmd,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE,
                                              shell=True
                                              )
            predict_status.wait()
            if predict_status.returncode != 0:
                (stdout, stderr) = predict_status.communicate()
                logger.info(f'STDOUT:\n{stdout}')
                logger.info(f'STDERR:\n{stderr}')
                raise Exception(f'Failed running nvidia cli command: {" ".join(cmd)}. more logs above')
            annotations_batch = list()
            for image_path in os.listdir(self.images_path):
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
                annotations_batch.append(image_annotations)

        finally:
            shutil.rmtree(self.images_path)
        return annotations_batch

    def prepare_item_func(self, item: dl.entities.Item):
        return item
