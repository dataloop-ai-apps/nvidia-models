import dtlpy as dl

dl.setenv('rc')


project = dl.projects.get(project_name='Test Project')
dataset = project.datasets.get(dataset_name='test')

module = dl.PackageModule(
    functions=[
        dl.PackageFunction(
            name='detect',
            description='detection using TAO model',
            inputs=[
                dl.FunctionIO(name='item', type=dl.PackageInputType.ITEM)
            ]
        )
    ]
)
package = project.packages.push(package_name=f'tao-package',
                                src_path='./service',
                                modules=[module],
                                service_config={
                                    'runtime': dl.KubernetesRuntime(pod_type='gpu-t4',
                                                                    autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                        min_replicas=1,
                                                                        max_replicas=1),
                                                                    runner_image='gcr.io/viewo-g/piper/agent/runner/gpu/nvidia-tao:0.1.2',
                                                                    concurrency=1).to_json()},)

service = package.services.deploy(service_name=package.name,
                                  runtime=dl.KubernetesRuntime(
                                      autoscaler=dl.KubernetesAutoscaler(min_replicas=0),
                                      pod_type='gpu-t4'),
                                  execution_timeout=0)
