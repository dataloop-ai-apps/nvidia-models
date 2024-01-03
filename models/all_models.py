try:
    from models.TrafficCamNet.trafficcamnet import TrafficCamNet
    from models.LicensePlateDetection.license_plate_detection import LPDNet
    from models.LicensePlateRecognition.license_plate_recognition import LPRNet
except Exception:
    from TrafficCamNet.trafficcamnet import TrafficCamNet
    from LicensePlateDetection.license_plate_detection import LPDNet
    from LicensePlateRecognition.license_plate_recognition import LPRNet

models = [
    TrafficCamNet,
    LPDNet,
    LPRNet
]
