import os
import logging
import shutil
import subprocess
import dtlpy as dl
import json

from facedetect import FaceDetect

logger = logging.getLogger('[Nvidia Models]')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for NVIDIA TAO models',
                              init_inputs={'model_entity': dl.Model})
class TaoModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
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
        logger.info(f'Loading api configs from {os.getcwd()}: content: {os.listdir(os.getcwd())}')
        with open(f'models/api_configs.json', 'r') as f:
            _configuration = json.load(f)

        inputdata = _configuration["ngc_api_key"].encode() + b'\n\n' + \
                    _configuration["ngc_org"].encode() + b'\n\n\n'
        # inputdata = self.configuration["ngc_api_key"].encode() + b'\n\n' + \
        #             self.configuration["ngc_org"].encode() + b'\n\n\n'
        process.communicate(input=inputdata)
        os.makedirs('/tmp/tao_models', exist_ok=True)

        logger.info('loading model')
        self.tao_model = None
        self.images_path = os.path.join(os.getcwd(), 'images')
        self.tao_model = FaceDetect(**self.configuration["model_config"])

        # for model in models:
        #     if self.configuration["model_name"] == model.get_name():
        #         self.tao_model = model(**self.configuration["model_config"])
        #         break
        # else:
        #     logger.warning("invalid model_name in configuration")

        if os.path.isdir(self.images_path):
            shutil.rmtree(self.images_path)

    def predict(self, batch, **kwargs):
        logger.info('predicting batch of size: {}'.format(len(batch)))
        logger.info(f'batch = {batch}')

        os.mkdir(self.images_path)
        for i, item in enumerate(batch):
            logger.info(f'item = {item}')
            item.download(local_path=self.images_path)
        try:
            logger.info("*** images path: {}".format(self.images_path))
            return self.tao_model.detect(self.images_path)
        finally:
            shutil.rmtree(self.images_path)

    def prepare_item_func(self, item: dl.entities.Item):
        return item
