import os
import logging
import subprocess
import dtlpy as dl
from pathlib import Path
from glob import glob
import open3d as o3d
import numpy as np
import urllib.request

try:
    from ..tao_model import TaoModel
except Exception:
    from tao_model import TaoModel

logger = logging.getLogger('[PointPillarNet]')


class PointPillarNet(TaoModel):
    def __init__(self, **model_config):
        super().__init__(**model_config)
        self.key = 'tlt_encode'
        self.res_dir = 'pointpillarnet_res'
        os.makedirs(self.res_dir, exist_ok=True)
        # download model - the txt config file points to this location for the model
        subprocess.Popen([
            '/tmp/ngccli/ngc-cli/ngc registry model download-version "nvidia/tao/pointpillarnet:trainable_v1.0" --dest /tmp/tao_models/'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, shell=True).wait()
        if not os.path.isfile("/tmp/tao_models/pointpillarnet_vtrainable_v1.0/pointpillars_trainable.tlt"):
            raise Exception("Failed loading the model")

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
        ret = []
        try:
            points, num_points = self._extract_pcd_files_points(images_dir=images_dir)
            logger.info(f"Running pointpillars inference on {images_dir}, Content {os.listdir(images_dir)}")
            os.makedirs(f'{os.getcwd()}/{self.res_dir}', exist_ok=True)
            with os.popen(
                    f'pointpillars inference '
                    f'-e {os.getcwd()}/models/PointPillarNet/inference_spec.txt '
                    f'-i {points} {num_points} '
                    f'-r {os.getcwd()}/{self.res_dir} '
                    f'-k {self.key}') as f:
                output = f.read().strip()
            logger.info(f"Full Model Output:\n{output}")
            for image_path in os.listdir(images_dir):
                image_annotations = dl.AnnotationCollection()
                logger.info(f"**** res dir {os.getcwd()}/{self.res_dir}")
                logger.info(f"**** res dir content {os.listdir(f'{os.getcwd()}/{self.res_dir}')}")
                with open(f'{os.getcwd()}/{self.res_dir}/labels/{Path(image_path).stem}.txt', 'r') as f:
                    for line in f.readlines():
                        vals = line.split(' ')
                        print(vals)
                        if vals[0] == 'Vehicle':
                            image_annotations.add(
                                annotation_definition=dl.Cube3d(
                                    label='Vehicle',
                                    position=[],
                                    scale=[],
                                    rotation=[]
                                ),
                                model_info={
                                    'name': self.get_name(),
                                    'confidence': 0.5
                                }
                            )
                            logger.info(f'detected [left, top, bottom, right]: {vals[4:8]}')
                        if vals[0] == 'Pedestrian':
                            image_annotations.add(
                                annotation_definition=dl.Cube3d(
                                    label='Pedestrian',
                                    position=[],
                                    scale=[],
                                    rotation=[]
                                ),
                                model_info={
                                    'name': self.get_name(),
                                    'confidence': 0.5
                                }
                            )
                            logger.info(f'detected [left, top, bottom, right]: {vals[4:8]}')
                        if vals[0] == 'Cyclist':
                            image_annotations.add(
                                annotation_definition=dl.Cube3d(
                                    label='Cyclist',
                                    position=[],
                                    scale=[],
                                    rotation=[]
                                ),
                                model_info={
                                    'name': self.get_name(),
                                    'confidence': 0.5
                                }
                            )
                            logger.info(f'detected [left, top, bottom, right]: {vals[4:8]}')
                ret.append(image_annotations)
            return ret
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    @staticmethod
    def get_name():
        return "point-pillar-net"

    @staticmethod
    def get_labels():
        return ['Vehicle', 'Pedestrian', 'Cyclist']

    @staticmethod
    def get_output_type():
        return dl.AnnotationType.CUBE3D


def test_extract_pcd_files_points():
    images_dir = "./"
    points, num_points = PointPillarNet._extract_pcd_files_points(images_dir=images_dir)
    print(points)
    print(num_points)


if __name__ == '__main__':
    test_extract_pcd_files_points()
