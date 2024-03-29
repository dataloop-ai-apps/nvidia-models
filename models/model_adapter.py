import os
import logging
import shutil
import subprocess
import dtlpy as dl

try:
    from models.all_models import models
except Exception:
    from all_models import models

logger = logging.getLogger('[Nvidia Models]')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for NVIDIA TAO models',
                              init_inputs={
                                  'ngc_api_key_secret_name': 'String',
                                  'ngc_org_secret_name': 'String',
                                  'model_entity': dl.Model
                              })
class TaoModelAdapter(dl.BaseModelAdapter):
    def __init__(self, ngc_api_key_secret_name, ngc_org_secret_name, model_entity: dl.Model = None):
        self.images_path = None
        self.tao_model = None
        self.ngc_config = {
            "ngc_api_key": os.environ.get(ngc_api_key_secret_name),
            "ngc_org": os.environ.get(ngc_org_secret_name),
        }
        super(TaoModelAdapter, self).__init__(model_entity)

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
        input_data = (
                self.ngc_config["ngc_api_key"].encode() + b'\n\n' +
                self.ngc_config["ngc_org"].encode() + b'\n\n\n'
        )
        process.communicate(input=input_data)
        os.makedirs('/tmp/tao_models', exist_ok=True)

        logger.info('loading model')
        self.images_path = os.path.join(os.getcwd(), 'images')

        tao_model_class = models.get(self.configuration["model_name"], None)
        if tao_model_class is not None:
            self.tao_model = tao_model_class()
        else:
            raise Exception("invalid model_name in configuration!")

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
