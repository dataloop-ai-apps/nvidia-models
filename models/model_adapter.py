import os
import logging
import shutil
import subprocess
from pathlib import Path
import dtlpy as dl

logger = logging.getLogger('[Nvidia Models]')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for NVIDIA TAO models',
                              init_inputs={
                                  'ngc_api_key_secret_name': 'String',
                                  'ngc_org_secret_name': 'String',
                                  'model_entity': dl.Model
                              })
class NvidiaBase(dl.BaseModelAdapter):
    def __init__(self, ngc_api_key_secret_name, ngc_org_secret_name, model_entity: dl.Model = None):
        # Define Variables
        self.model_name = None
        self.model_key = None
        self.model_version = None
        self.res_dir = None

        self.images_path = None
        self.tao_model = None
        self.cmd = None

        # Read NGC Config
        self.ngc_config = {
            "ngc_api_key": os.environ.get(ngc_api_key_secret_name),
            "ngc_org": os.environ.get(ngc_org_secret_name),
        }
        
        super(NvidiaBase, self).__init__(model_entity)

    def get_cmd(self):
        raise NotImplementedError("Please implement 'get_cmd' method in {}".format(self.__class__.__name__))

    @staticmethod
    def _build_bash_cmd(cmd):
        import shlex
        for entrypoint in ['/nvidia_entrypoint.sh', '/opt/nvidia/entrypoint.sh', '/opt/entrypoint.sh']:
            if os.path.isfile(entrypoint):
                logger.info(f"Using NVIDIA entrypoint: {entrypoint}")
                return ['bash', entrypoint] + cmd

        bash_cmd = ' '.join(shlex.quote(c) for c in cmd)
        logger.info(f"Running via bash (non-login shell): {bash_cmd}")
        return ['bash', '-c', bash_cmd]

    def parse_results(self, predict_status):
        # Currently used by lpr-net
        pass

    @staticmethod
    def _prepare_ngc_cli():
        os.makedirs(name='/tmp/ngccli', exist_ok=True)
        logger.info('downloading "https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip"')
        try:
            result = subprocess.run(
                ['wget', 'https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip',
                '-O', '/tmp/ngccli/ngccli_cat_linux.zip'],
                capture_output=True
            )
            if result.returncode != 0:
                logger.error(f'Failed downloading ngccli_cat_linux.zip: {result.stderr.decode()}')
                raise ValueError(f'Failed downloading ngccli_cat_linux.zip: {result.stderr.decode()}')
        except FileNotFoundError:
            logger.error('wget not found on system PATH')
            raise RuntimeError('wget not found on system PATH')
        except subprocess.SubprocessError as e:
            logger.error(f'Failed to run wget: {e}')
            raise RuntimeError(f'Failed to run wget: {e}')
        logger.info('unzipping ngccli_cat_linux.zip')
        unzip_command = subprocess.Popen(
            ['unzip', '-u', '/tmp/ngccli/ngccli_cat_linux.zip', '-d', '/tmp/ngccli/'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        _, stderr_output = unzip_command.communicate()
        if unzip_command.returncode != 0:
            logger.error(f'Failed unzipping ngccli_cat_linux.zip: {stderr_output.decode()}')
            raise ValueError(f'Failed unzipping ngccli_cat_linux.zip: {stderr_output.decode()}')
        logger.info('adding ngccli to system PATH environment variable')
        if "/tmp/ngccli/ngc-cli" not in os.environ["PATH"]:
            os.environ["PATH"] = "/tmp/ngccli/ngc-cli:{}".format(os.getenv("PATH", ""))
        for tao_bin_dir in ['/opt/conda/bin', '/opt/conda/envs/tao/bin', '/usr/local/bin', os.path.expanduser('~/.local/bin')]:
            if os.path.isdir(tao_bin_dir) and tao_bin_dir not in os.environ.get('PATH', ''):
                os.environ['PATH'] = f"{tao_bin_dir}:{os.environ.get('PATH', '')}"
        logger.info(f"PATH after setup: {os.environ.get('PATH')}")

    def load(self, local_path, **kwargs):
        model_name = self.model_entity.configuration.get("model_name")
        model_key = self.model_entity.configuration.get("model_key")
        model_version = self.model_entity.configuration.get("model_version")

        self._prepare_ngc_cli()
        logger.info('login to ngc')
        process = subprocess.Popen(
            ['/tmp/ngccli/ngc-cli/ngc', 'config', 'set'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info("sending ngc config set command")
        input_data = (
            self.ngc_config["ngc_api_key"].encode() + b'\n\n' +
            self.ngc_config["ngc_org"].encode() + b'\n\n\n'
        )
        process.communicate(input=input_data)
        logger.info("ngc config set command executed")
        os.makedirs('/tmp/tao_models', exist_ok=True)
        logger.info("model directory created")

        tao_pkg = subprocess.run(
            ['find', '/', '-name', 'nvidia_tao_tf1', '-type', 'd', '-maxdepth', '10'],
            capture_output=True, timeout=30
        )
        logger.info(f"nvidia_tao_tf1 package dirs: {tao_pkg.stdout.decode().splitlines()}")
        which_result = subprocess.run(['which', 'yolo_v4_tiny'], capture_output=True, text=True)
        logger.info(f"yolo_v4_tiny binary: {which_result.stdout.strip() or 'NOT FOUND in PATH'}")
        which_smi = subprocess.run(['which', 'nvidia-smi'], capture_output=True, text=True)
        logger.info(f"nvidia-smi binary: {which_smi.stdout.strip() or 'NOT FOUND in PATH'}")
        nvidia_bin = subprocess.run(['ls', '/usr/local/nvidia/bin/'], capture_output=True, text=True)
        logger.info(f"/usr/local/nvidia/bin: {nvidia_bin.stdout.strip() or nvidia_bin.stderr.strip()}")
        logger.info(f"IS_GPU_AVAILABLE={os.environ.get('IS_GPU_AVAILABLE', 'NOT SET')}")
        find_tao = subprocess.run(
            ['find', '/opt', '/', '-name', 'yolo_v4_tiny', '-maxdepth', '8'],
            capture_output=True, text=True, timeout=30
        )
        logger.info(f"yolo_v4_tiny locations: {find_tao.stdout.strip().splitlines()}")
        conda_envs = subprocess.run(['ls', '/opt/conda/envs/'], capture_output=True, text=True)
        logger.info(f"conda envs: {conda_envs.stdout.strip() or conda_envs.stderr.strip()}")
        ls_local_bin = subprocess.run(['ls', '/usr/local/bin/'], capture_output=True, text=True)
        logger.info(f"/usr/local/bin: {ls_local_bin.stdout.strip()}")
        find_tao_cmd = subprocess.run(['find', '/', '-name', 'tao', '-type', 'f'], capture_output=True, text=True, timeout=30)
        logger.info(f"tao binary locations: {find_tao_cmd.stdout.strip().splitlines()}")
        find_yolo_broad = subprocess.run(['find', '/', '-name', '*yolo*', '-maxdepth', '15'], capture_output=True, text=True, timeout=60)
        logger.info(f"yolo* files: {find_yolo_broad.stdout.strip().splitlines()[:20]}")

        logger.info('loading model')
        self.images_path = os.path.join(os.getcwd(), 'images')

        self.model_name = model_name
        self.model_key = model_key
        self.model_version = model_version
        self.res_dir = os.path.join(os.getcwd(), 'results')

        # download model - the txt config file points to this location for the model
        # logger.info("Downloading model artifacts")

        cli_filepath = os.path.join('/tmp', 'ngccli', 'ngc-cli', 'ngc')
        dest_path = os.path.join('/tmp', 'tao_models')
        cmd = [cli_filepath, 'registry', 'model', 'download-version', self.model_version, '--dest', dest_path]
        download_status = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        download_status.wait()
        if download_status.returncode != 0:
            (stdout, stderr) = download_status.communicate()
            logger.info(f'STDOUT:\n{stdout}')
            logger.info(f'STDERR:\n{stderr}')
            raise Exception(f'Failed downloading cli command: {" ".join(cmd)}. more logs above')
        if os.path.isdir(self.images_path):
            shutil.rmtree(self.images_path)

    def predict(self, batch, **kwargs):
        try:
            logger.info('predicting batch of size: {}'.format(len(batch)))
            logger.info(f'batch = {batch}')

            os.mkdir(self.images_path)
            for i, item in enumerate(batch):
                logger.info(f'item = {item}')
                item.download(local_path=self.images_path)

            os.makedirs(self.res_dir, exist_ok=True)
            cmd = self.get_cmd()
            logger.info(f"cmd: {cmd}")
            predict_status = subprocess.Popen(
                self._build_bash_cmd(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = predict_status.communicate()
            if predict_status.returncode != 0:
                logger.info(f'STDOUT:\n{stdout}')
                logger.info(f'STDERR:\n{stderr}')
                raise Exception(f'Failed running nvidia cli command: {" ".join(cmd)}. more logs above')
            self.parse_results(predict_status=predict_status)

            annotations_batch = list()
            for image_path in os.listdir(self.images_path):
                image_annotations = dl.AnnotationCollection()
                output_filepath = os.path.join(self.res_dir, "labels", f"{Path(image_path).stem}.txt")
                with open(output_filepath, 'r') as f:
                    for line in f.readlines():
                        if self.model_entity.output_type == 'class':
                            result = line.strip()
                            image_annotations.add(
                                annotation_definition=dl.Classification(
                                    label=result
                                ),
                                model_info={
                                    'name': self.model_name,
                                    'confidence': 1.0
                                }
                            )
                        else:
                            vals = line.split(' ')
                            image_annotations.add(
                                annotation_definition=dl.Box(
                                    label=vals[0],
                                    top=vals[5],
                                    left=vals[4],
                                    bottom=vals[7],
                                    right=vals[6]
                                ),
                                model_info={
                                    'name': self.model_name,
                                    'confidence': float(vals[-1]) / 100
                                }
                            )
                            # logger.info(f'detected [left, top, bottom, right]: {vals[4:8]}')
                            # logger.info(f'Full Annotation Result: {vals}')
                annotations_batch.append(image_annotations)

        finally:
            shutil.rmtree(self.images_path)
        return annotations_batch

    def prepare_item_func(self, item: dl.entities.Item):
        return item
