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
        logger.info("[INFO] downloading image...")
        filename = batch.download()
        builder = batch.annotations.builder()

        try:
            for annotation in self.tao_model.detect(filename):
                builder.add(annotation)
        finally:
            os.remove(filename)

        return builder

    def predict_items(self, items, **kwargs):
        for item in items:
            builder = self.predict(batch=item)
            item.annotations.upload(builder)
