# NVIDIA TAO Models in Dataloop

This repository provides an adapter to NVIDIA TAO Models for the Dataloop platform.

# Available TAO Model:

The following models from [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/containers/tao-toolkit) are available for installation on Dataloop platform:
1. [DashCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/dashcamnet):
   1. `Description:` The model detects one or more physical objects from four categories within an image and returns a box around each object, as well as a category label for each object. The four categories of objects detected by this model are – car, persons, road signs and bicycles.
   2. `Labels:` car, bicycle, person, road_sign
   3. `Annotations Type:` Bounding Box
2. [FaceDetectIR](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facedetectir):
   1. `Description:` The model detects one or more faces in the given image / video. Compared to the PeopleNet model, this model gives better results detecting large faces, such as faces in webcam images.
   2. `Labels:` face
   3. `Annotations Type:` Bounding Box
3. [LicensePlateDetection](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/lpdnet):
   1. `Description:` The model detects one or more license plate objects from a car image and return a box around each object, as well as an lpd label for each object.
   2. `Labels:` lpd
   3. `Annotations Type:` Bounding Box
4. [PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet): 
   1. `Description:` The Peoplenet model detects persons, bags, and faces in an image. This model is ready for commercial use.
   2. `Labels:` person, bag, face
   3. `Annotations Type:` Bounding Box
5. [TrafficCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet):
   1. `Description:` The model described in this card detects one or more physical objects from four categories within an image and returns a box around each object, as well as a category label for each object. The four categories of objects detected by this model are – car, persons, road signs and two-wheelers.
   2. `Labels:` car, bicycle, person, road_sign
   3. `Annotations Type:` Bounding Box

# Using the model as a Predict Model node in a pipeline:

To use the model as a Predict Model node in a pipeline, after it was deployed, do as follows:
1. Create a pipeline or edit an existing pipeline.
2. Add a Predict Model pipeline node to the canvas.
3. Select the requested model from the Model list.

![predict_node.png](assets%2Fpredict_node.png)
