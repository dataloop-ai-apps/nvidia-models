try:
    from models.dash_cam_net.dashcamnet import DashCamNet
    # from models.face_detect.facenet import FaceNet
    from models.face_detect_ir.facedetectir import FaceDetectIR
    from models.license_plate_detection.license_plate_detection import LPDNet
    from models.license_plate_recognition.license_plate_recognition import LPRNet
    from models.people_net.peoplenet import PeopleNet
    from models.traffic_cam_net.trafficcamnet import TrafficCamNet

except Exception:
    from dash_cam_net.dashcamnet import DashCamNet
    # from models.face_detect.facenet import FaceNet
    from face_detect_ir.facedetectir import FaceDetectIR
    from license_plate_detection.license_plate_detection import LPDNet
    from license_plate_recognition.license_plate_recognition import LPRNet
    from people_net.peoplenet import PeopleNet
    from traffic_cam_net.trafficcamnet import TrafficCamNet


models = [
    DashCamNet,
    # FaceNet,
    FaceDetectIR,
    LPDNet,
    LPRNet,
    PeopleNet,
    TrafficCamNet,
]
