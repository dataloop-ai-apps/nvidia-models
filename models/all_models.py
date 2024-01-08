try:
    from models.PeopleNet.peoplenet import PeopleNet
    from models.TrafficCamNet.trafficcamnet import TrafficCamNet
    from models.LicensePlateDetection.license_plate_detection import LPDNet
    from models.LicensePlateRecognition.license_plate_recognition import LPRNet
except Exception:
    from models.PeopleNet.peoplenet import PeopleNet
    from TrafficCamNet.trafficcamnet import TrafficCamNet
    from LicensePlateDetection.license_plate_detection import LPDNet
    from LicensePlateRecognition.license_plate_recognition import LPRNet

models = [
    PeopleNet,
    TrafficCamNet,
    LPDNet,
    LPRNet
]
