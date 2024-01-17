# NVIDIA TAO Models in Dataloop

This repo provides a way to run NVIDIA TAO Models as a Dataloop service.

## How to deploy a TAO service?

The easiest way is to simply run the `upload.py` script with your project and dataset 
(change at the top of the script).

The `upload.py` script uploads an example service of the 
[TrafficCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet)
model.

If you wish to deploy a service for a different TAO model, you need a docker image based on
an image from [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/containers/tao-toolkit)
that supports your model.
You can use the `Dockerfile` as a template for making your own image.
Make sure to `COPY` your model and any other files you need.

Once you have an image, change it in the `upload.py` script 
and change the command to run the model in `service/main.py`