import os
import logging
import subprocess
import dtlpy as dl
from pathlib import Path
from glob import glob
import open3d as o3d
import numpy as np

logger = logging.getLogger('[PointPillarNet]')


class PointPillarNet:
    def __init__(self):
        self.name = "people-net"
        self.key = 'tlt_encode'
        self.res_dir = os.path.join(os.getcwd(), 'pointpillarnet_res')
        self.model_download_version = "nvidia/tao/pointpillarnet:trainable_v1.0"
        self.current_dir = os.path.dirname(str(__file__))

        # download model - the txt config file points to this location for the model
        # logger.info("Downloading model artifacts")

        cli_filepath = os.path.join('/tmp', 'ngccli', 'ngc-cli', 'ngc')
        dest_path = os.path.join('/tmp', 'tao_models')
        download_status = subprocess.Popen(
            [f'{cli_filepath} registry model download-version "{self.model_download_version}" --dest {dest_path}'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        download_status.wait()
        if download_status.returncode != 0:
            (out, err) = download_status.communicate()
            raise Exception(f'Failed loading the model: {err}')

        # if not os.path.isfile("/tmp/tao_models/pointpillarnet_vtrainable_v1.0/pointpillars_trainable.tlt"):
        #     raise Exception("Failed loading the model")

    @staticmethod
    def _extract_pcd_files_points(images_dir):
        pcd_filepaths = glob(pathname=f"{images_dir}/*.pcd")
        points = list()
        num_points = list()
        for pcd_filepath in pcd_filepaths:
            pcd_points = np.array(o3d.io.read_point_cloud(filename=pcd_filepath).points).tolist()
            points.append(pcd_points)
            num_points.append(len(pcd_points))
        return points, num_points

    def detect(self, images_dir):
        ret = list()
        logger.info(f"Running pointpillars inference on {images_dir}, Content {os.listdir(images_dir)}")

        # points, num_points = self._extract_pcd_files_points(images_dir=images_dir)
        # for point, num_point in zip(points, num_points):
        #     point = [point]
        #     num_point = [num_point]

        specs_filepath = os.path.join(self.current_dir, "inference_spec.yaml")
        os.makedirs(self.res_dir, exist_ok=True)
        with os.popen(
            f'pointpillars inference '
            f'-e {specs_filepath} '
            f'-r {self.res_dir} '
            f'-k {self.key}'
        ) as f:
            output = f.read().strip()
            # logger.info(f"Full Model Output:\n{output}")

        for image_path in os.listdir(images_dir):
            image_annotations = dl.AnnotationCollection()
            output_filepath = os.path.join(self.res_dir, "labels", f"{Path(image_path).stem}.txt")
            with open(output_filepath, 'r') as f:
                for line in f.readlines():
                    vals = line.split(' ')
                    print(vals)
                    image_annotations.add(
                        annotation_definition=dl.Cube3d(
                            label=vals[0],
                            position=[],
                            scale=[],
                            rotation=[]
                        ),
                        model_info={
                            'name': self.name,
                            'confidence': 0.5
                        }
                    )
                    # logger.info(f'detected [left, top, bottom, right]: {vals[4:8]}')
                    # logger.info(f'Full Annotation Result: {vals}')
            ret.append(image_annotations)
        return ret


def test_extract_pcd_files_points():
    images_dir = "./"
    points, num_points = PointPillarNet._extract_pcd_files_points(images_dir=images_dir)
    print(points)
    print(num_points)


if __name__ == '__main__':
    test_extract_pcd_files_points()
