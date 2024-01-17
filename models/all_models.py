try:
    from models.DashCamNet.dashcamnet import DashCamNet
    from models.FaceDetectIR.facedetectir import FaceDetectIR
    from models.LicensePlateDetection.license_plate_detection import LPDNet
    from models.LicensePlateRecognition.license_plate_recognition import LPRNet
    from models.PeopleNet.peoplenet import PeopleNet
    from models.TrafficCamNet.trafficcamnet import TrafficCamNet

except Exception:
    from DashCamNet.dashcamnet import DashCamNet
    from FaceDetectIR.facedetectir import FaceDetectIR
    from LicensePlateDetection.license_plate_detection import LPDNet
    from LicensePlateRecognition.license_plate_recognition import LPRNet
    from PeopleNet.peoplenet import PeopleNet
    from TrafficCamNet.trafficcamnet import TrafficCamNet


models = [
    DashCamNet,
    FaceDetectIR,
    LPDNet,
    LPRNet,
    PeopleNet,
    TrafficCamNet,
]
