import unittest
import dtlpy as dl
import os
import json
import sys
sys.path[0] = ""
from models.model_adapter import TaoModelAdapter


BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
DATASET_NAME = os.environ['DATASET_NAME']


class MyTestCase(unittest.TestCase):
    project: dl.Project = None
    dataset: dl.Dataset = None
    root_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_path: str = os.path.join(root_path, 'models')

    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        os.chdir(cls.root_path)
        if dl.token_expired():
            dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        cls.project = dl.projects.get(project_id=PROJECT_ID)
        try:
            cls.dataset = cls.project.datasets.get(dataset_name=DATASET_NAME)
        except dl.exceptions.NotFound:
            cls.dataset = cls.project.datasets.create(dataset_name=DATASET_NAME)

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete all models
        for model in cls.project.models.list().all():
            model.delete()

        # Delete all apps
        for app in cls.project.apps.list().all():
            if app.project.id == cls.project.id:
                app.uninstall()

        # Delete all dpks
        filters = dl.Filters(resource=dl.FiltersResource.DPK)
        filters.add(field="scope", values="project")
        for dpk in cls.project.dpks.list(filters=filters).all():
            if dpk.project.id == cls.project.id and dpk.creator == BOT_EMAIL:
                dpk.delete()
        dl.logout()

    def _perform_model_evaluation(self, model_folder_name: str):
        model_folder_path = os.path.join(self.models_path, model_folder_name)

        # Open dataloop json
        dataloop_json_filepath = os.path.join(model_folder_path, 'dataloop.json')
        with open(dataloop_json_filepath, 'r') as f:
            dataloop_json = json.load(f)
        dataloop_json.pop('codebase')
        dataloop_json["scope"] = "project"
        dataloop_json["name"] = f'{dataloop_json["name"]}-{self.project.id}'
        model_name = dataloop_json.get('components', dict()).get('models', list())[0].get("name", None)

        # Publish dpk and install app
        dpk = dl.Dpk.from_json(_json=dataloop_json, client_api=dl.client_api, project=self.project)
        dpk = self.project.dpks.publish(dpk=dpk)
        app = self.project.apps.install(dpk=dpk)

        # Get model adapter
        model = app.project.models.get(model_name=model_name)
        model_adapter = TaoModelAdapter(
            ngc_api_key_secret_name='API_KEY',
            ngc_org_secret_name='API_ORG',
            model_entity=model
        )

        filters = dl.Filters()
        filters.add(field="metadata.system.tags.test", values=True)
        model_adapter.evaluate_model(model=model, dataset=self.dataset, filters=filters)
        return True

    def test_dash_cam_net(self):
        model_folder_name = "dash_cam_net"
        evaluation_result = self._perform_model_evaluation(model_folder_name=model_folder_name)
        self.assertTrue(evaluation_result)


if __name__ == '__main__':
    unittest.main()
