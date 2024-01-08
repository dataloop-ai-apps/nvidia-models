# Run from "models" directory.
# You can change the code inside "models" directory, and rerun.
# This will upload the new codebase as a new version, and you will be able to update to the new version in the platform.
import dtlpy as dl
from models.model_adapter import TaoModelAdapter
from models.all_models import models


def upload_models(project_name, dataset_name, tao_model):
    project = dl.projects.get(project_name=project_name)
    dataset = project.datasets.get(dataset_name=dataset_name)

    codebase = project.codebases.pack(directory='./')
    metadata = dl.Package.get_ml_metadata(cls=TaoModelAdapter, default_configuration={})
    module = dl.PackageModule.from_entry_point(entry_point='models/model_adapter.py')
    req = dl.PackageRequirement('dtlpy')
    package_name = f'nv-tao-package-{tao_model.get_name()}'
    package = project.packages.push(package_name=package_name,
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
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        runner_image='gcr.io/viewo-g/piper/agent/runner/gpu/nvidia-tao:0.1.2',
                                                                        # this image contains ngc but not working...
                                                                        # based on current Dockerfile:
                                                                        # runner_image='docker.io/yakirinven/nvidia_tao:latest',
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)

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
        model = package.models.get(model_name=f'nv-{tao_model.get_name()}-model')
        model.status = 'trained'
        model.update()
        model.deploy()
        pass


def main():
    project_name = "Nvidia Demo"
    dataset_name = "Nvidia"

    for tao_model in models:
        upload_models(project_name=project_name, dataset_name=dataset_name, tao_model=tao_model)


if __name__ == '__main__':
    main()
