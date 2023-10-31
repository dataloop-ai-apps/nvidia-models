import os
import logging
import shutil

import dtlpy as dl
try:
    from models.all_models import models
except Exception:
    from all_models import models

logger = logging.getLogger('[Nvidia Models]')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for NVIDIA TAO models',
                              init_inputs={'model_entity': dl.Model})
class TaoModelAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        print('loading model')
        self.tao_model = None
        self.images_path = os.path.join(os.getcwd(), 'images')

        for model in models:
            if self.configuration["model_name"] == model.get_name():
                self.tao_model = model(**self.configuration["model_config"])
                break
        else:
            print("WARNING: invalid model_name in configuration")

        if os.path.isdir(self.images_path):
            shutil.rmtree(self.images_path)

    def predict(self, batch, **kwargs):
        logger.info('predicting batch of size: {}'.format(len(batch)))
        logger.info(f'batch = {batch}')

        os.mkdir(self.images_path)
        for i, item in enumerate(batch):
            logger.info(f'item = {item}')
            item.download(local_path=os.path.join(self.images_path, f'image_{i:020}.jpg'), to_items_folder=False)

        try:
            return self.tao_model.detect(self.images_path)
        finally:
            shutil.rmtree(self.images_path)

    def prepare_item_func(self, item: dl.entities.Item):
        return item
