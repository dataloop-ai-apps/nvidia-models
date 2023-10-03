try:
    from models.TrafficCamNet.trafficcamnet import TrafficCamNet
except Exception:
    from TrafficCamNet.trafficcamnet import TrafficCamNet

models = [
    TrafficCamNet
]