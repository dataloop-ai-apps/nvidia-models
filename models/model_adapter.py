import os
import logging
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
        for model in models:
            if self.configuration["model_name"] == model.get_name():
                self.tao_model = model(**self.configuration["model_config"])
                break
        else:
            print("WARNING: invalid model_name in configuration")

    def predict(self, batch, **kwargs):
        logger.info('predicting batch of size: {}'.format(len(batch)))
        batch_annotations = []
        logger.info(f'batch = {batch}')
        for item in batch:
            logger.info(f'item = {item}')
            filename = item.download()
            logger.info(filename)
            os.replace(filename, 'tmp.jpg')
            filename = 'tmp.jpg'
            image_annotations = dl.AnnotationCollection()
            try:
                for annotation in self.tao_model.detect(filename):
                    image_annotations.add(annotation_definition=annotation,
                                          model_info={
                                              'name': 'NVIDIA-TAO',
                                              'confidence': 0.5
                                          })
                batch_annotations.append(image_annotations)
            finally:
                os.remove(filename)
        return batch_annotations

    def prepare_item_func(self, item: dl.entities.Item):
        return item
