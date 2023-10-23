# Run from "models" directory.
# You can change the code inside "models" directory, and rerun.
# This will upload the new codebase as a new version, and you will be able to update to the new version in the platform.
import dtlpy as dl
from typing import List, Type
from models.tao_model import TaoModel
from models.model_adapter import TaoModelAdapter
from models.all_models import models
dl.setenv('rc')


def upload_models(project_name, dataset_name, tao_models: List[Type[TaoModel]]):
    project = dl.projects.get(project_name=project_name)
    dataset = project.datasets.get(dataset_name=dataset_name)

    codebase = project.codebases.pack(directory='./')
    metadata = dl.Package.get_ml_metadata(cls=TaoModelAdapter, default_configuration={})
    module = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')
    req = dl.PackageRequirement('dtlpy')

    package = project.packages.push(package_name=f'nv-tao-package',
                                    src_path='./',
                                    package_type='ml',
                                    codebase=codebase,
                                    modules=[module],
                                    is_global=False,
                                    requirements=[req],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type='gpu-t4',
                                                                        preemptible=False,
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=1,
                                                                            max_replicas=1),
                                                                        runner_image='gcr.io/viewo-g/piper/agent/runner/gpu/nvidia-tao:0.1.1',
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)

    for tao_model in tao_models:
        try:
            model = package.models.create(model_name=f'nv-{tao_model.get_name()}-model',
                                          description=f'nvidia-tao {tao_model.get_name()} model',
                                          tags=['pretrained'],
                                          dataset_id=dataset.id,
                                          project_id=package.project.id,
                                          configuration={
                                              "model_name": tao_model.get_name(),
                                              "model_config": tao_model.get_default_model_configuration()
                                          },
                                          model_artifacts=[],
                                          labels=tao_model.get_labels(),
                                          output_type=tao_model.get_output_type()
                                          )

            model.status = 'trained'
            model.update()
            model.deploy()
        except Exception:
            pass


def main():
    if dl.token_expired():
        dl.login()
    upload_models('DevProject', 'dev', tao_models=models)


if __name__ == '__main__':
    main()
