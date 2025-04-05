import supervision as sv
from tqdm import tqdm
from scripts.team import TeamClassifier
from ultralytics import YOLO

SOURCE_VIDEO_PATH = "input/121364_0.mp4"
PLAYER_DETECTION_MODEL = YOLO("models/player_detect/weights/best.pt")
PLAYER_ID = 2
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

team_classifier = TeamClassifier(device="cuda")
team_classifier.fit(crops)
team_classifier.save("models/team_classifier.pkl")