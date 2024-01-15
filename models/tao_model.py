from abc import ABC, abstractmethod


class TaoModel(ABC):
    def __init__(self, **model_config):
        print("Init TaoModel class")
        self.model_config = model_config

    @abstractmethod
    def detect(self, images_dir):
        pass

    @staticmethod
    @abstractmethod
    def get_name():
        pass

    @staticmethod
    def get_default_model_configuration():
        return {}

    @staticmethod
    @abstractmethod
    def get_labels():
        pass

    @staticmethod
    @abstractmethod
    def get_output_type():
        pass
