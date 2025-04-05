from ultralytics import YOLO
import supervision as sv
import cv2

from scripts.annotators import *
from scripts.utils import *
from scripts.view import ViewTransformer

FIELD_DETECTION_MODEL = YOLO("models/last_400.pt")
SOURCE_VIDEO_PATH = "input/121364_0.mp4"

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

# Run detection
result = FIELD_DETECTION_MODEL.predict(frame, verbose = False)[0]
# Convert to Supervision KeyPoints format
key_points = sv.KeyPoints.from_ultralytics(result)
# print(detections)

filter = key_points.confidence[0] > 0.5
frame_reference_points = key_points.xy[0][filter]
frame_reference_key_points = sv.KeyPoints(
    xy=frame_reference_points[np.newaxis, ...])

pitch_reference_points = np.array(CONFIG.vertices)[filter]

transformer = ViewTransformer(
    source=pitch_reference_points,
    target=frame_reference_points
)

pitch_all_points = np.array(CONFIG.vertices)
frame_all_points = transformer.transform_points(points=pitch_all_points)

frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

annotated_frame = frame.copy()
annotated_frame = edge_annotator.annotate(
    scene=annotated_frame,
    key_points=frame_all_key_points)
annotated_frame = vertex_annotator_2.annotate(
    scene=annotated_frame,
    key_points=frame_all_key_points)
annotated_frame = vertex_annotator.annotate(
    scene=annotated_frame,
    key_points=frame_reference_key_points)

cv2.imwrite("output/annotated_pitch_1.jpeg", annotated_frame)
