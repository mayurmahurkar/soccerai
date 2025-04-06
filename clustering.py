import supervision as sv
from tqdm import tqdm
from scripts.team import TeamClassifier
from ultralytics import YOLO
from scripts.utils import load_config

# Load configuration
config = load_config()

# Use configuration values
SOURCE_VIDEO_PATH = config["SOURCE_VIDEO_PATH"]
PLAYER_DETECTION_MODEL = YOLO(config["PLAYER_DETECTION_MODEL"])
PLAYER_ID = config["PLAYER_ID"]
STRIDE = 30

frame_generator = sv.get_video_frames_generator(
    source_path=SOURCE_VIDEO_PATH, stride=STRIDE)

crops = []
for frame in tqdm(frame_generator, desc='collecting crops'):
    result = PLAYER_DETECTION_MODEL.predict(frame, verbose = False)[0]
    detections = sv.Detections.from_ultralytics(result)
    players_detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    crops += players_crops

team_classifier = TeamClassifier(device=config["DEVICE"])
team_classifier.fit(crops)
team_classifier.save(config["TEAM_CLASSIFIER_MODEL"])