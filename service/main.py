import os
import subprocess
import dtlpy as dl
from pathlib import Path

dl.setenv('rc')


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        self.key = os.environ['TLT_KEY']
        self.res_dir = 'res'
        os.mkdir(self.res_dir)

    def detect(self, item: dl.Item):
        print("[INFO] downloading image...")
        filename = item.download()
        try:
            with os.popen(
                    f'detectnet_v2 inference -e /model/inference_spec.txt -i {filename} -o {os.getcwd()}/res -k {self.key}') as f:
                output = f.read().strip()
            print(output)
            with open(f'res/labels/{Path(filename).stem}.txt', 'r') as f:
                for line in f.readlines():
                    vals = line.split(' ')
                    if vals[0] == 'car':
                        print(f'detected: {vals[4:8]}')
        except Exception as e:
            print(f"Error: {e}")
        finally:
            os.remove(filename)
