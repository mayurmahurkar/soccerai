from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import os

from scripts.annotators import *
from scripts.utils import *
from scripts.football_pitch import *
from scripts.team import TeamClassifier
from scripts.view import ViewTransformer
from tqdm import tqdm

PLAYER_DETECTION_MODEL = YOLO("models/player_detect/weights/best.pt")
FIELD_DETECTION_MODEL = YOLO("models/last_400.pt")
team_classifier = TeamClassifier.load("models/team_classifier.pkl")

SOURCE_VIDEO_PATH = "input/121364_0.mp4"
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

tracker = sv.ByteTrack()
tracker.reset()

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)


# ball, goalkeeper, player, referee detection
result = PLAYER_DETECTION_MODEL.predict(frame, verbose = False)[0]
detections = sv.Detections.from_ultralytics(result)
# print(detections)

ball_detections = detections[detections.class_id == BALL_ID]
ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

all_detections = detections[detections.class_id != BALL_ID]
all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
all_detections = tracker.update_with_detections(detections=all_detections)

goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
players_detections = all_detections[all_detections.class_id == PLAYER_ID]
referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

# team assignment
players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
players_detections.class_id = team_classifier.predict(players_crops)

goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
    players_detections, goalkeepers_detections)

referees_detections.class_id -= 1

all_detections = sv.Detections.merge([
    players_detections, goalkeepers_detections, referees_detections])

# frame visualization
labels = [
    f"#{tracker_id}"
    for tracker_id
    in all_detections.tracker_id
]

all_detections.class_id = all_detections.class_id.astype(int)

annotated_frame = frame.copy()
annotated_frame = ellipse_annotator.annotate(
    scene=annotated_frame,
    detections=all_detections)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame,
    detections=all_detections,
    labels=labels)
annotated_frame = triangle_annotator.annotate(
    scene=annotated_frame,
    detections=ball_detections)

# Initialize video writers for each annotation type
output_video_path_annotated = "output/annotated_video.mp4"
output_video_path_pitch = "output/annotated_pitch_video.mp4"
output_video_path_voronoi = "output/voronoi_diagram_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
fps = 15  # Set the desired frames per second
frame_width = int(cv2.VideoCapture(SOURCE_VIDEO_PATH).get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cv2.VideoCapture(SOURCE_VIDEO_PATH).get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writers for each output
video_writer_annotated = cv2.VideoWriter(output_video_path_annotated, fourcc, fps, (frame_width, frame_height))
video_writer_pitch = cv2.VideoWriter(output_video_path_pitch, fourcc, fps, (frame_width, frame_height))
video_writer_voronoi = cv2.VideoWriter(output_video_path_voronoi, fourcc, fps, (frame_width, frame_height))

# Frame processing loop
for frame in frame_generator:

    # ball, goalkeeper, player, referee detection
    result = PLAYER_DETECTION_MODEL.predict(frame, verbose = False)[0]
    detections = sv.Detections.from_ultralytics(result)
    # print(detections)

    ball_detections = detections[detections.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    all_detections = detections[detections.class_id != BALL_ID]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections = tracker.update_with_detections(detections=all_detections)

    goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
    players_detections = all_detections[all_detections.class_id == PLAYER_ID]
    referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

    # team assignment
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
        players_detections, goalkeepers_detections)

    referees_detections.class_id -= 1

    all_detections = sv.Detections.merge([
        players_detections, goalkeepers_detections, referees_detections])

    # frame visualization
    labels = [
        f"#{tracker_id}"
        for tracker_id
        in all_detections.tracker_id
    ]

    all_detections.class_id = all_detections.class_id.astype(int)

    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(
        scene=annotated_frame,
        detections=all_detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=all_detections,
        labels=labels)
    annotated_frame = triangle_annotator.annotate(
        scene=annotated_frame,
        detections=ball_detections)

    # cv2.imwrite("output/annotated_frame_1.jpeg", annotated_frame)
    video_writer_annotated.write(annotated_frame)

    players_detections = sv.Detections.merge([
        players_detections, goalkeepers_detections
    ])

    ### --------------------------------------------------

    # detect pitch key points
    pitch_result = FIELD_DETECTION_MODEL.predict(frame, verbose = False)[0]
    key_points = sv.KeyPoints.from_ultralytics(pitch_result)

    # project ball, players and referies on pitch
    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

    referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_referees_xy = transformer.transform_points(points=referees_xy)

    # visualize video game-style radar view
    annotated_frame = draw_pitch(CONFIG)
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=annotated_frame)
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_players_xy[players_detections.class_id == 0],
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame)
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_players_xy[players_detections.class_id == 1],
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame)
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_referees_xy,
        face_color=sv.Color.from_hex('FFD700'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=annotated_frame)

    # cv2.imwrite("output/annotated_pitch_1.jpeg", annotated_frame)
    video_writer_pitch.write(annotated_frame)

    # visualize voronoi diagram
    annotated_frame = draw_pitch(CONFIG)
    annotated_frame = draw_pitch_voronoi_diagram(
        config=CONFIG,
        team_1_xy=pitch_players_xy[players_detections.class_id == 0],
        team_2_xy=pitch_players_xy[players_detections.class_id == 1],
        team_1_color=sv.Color.from_hex('00BFFF'),
        team_2_color=sv.Color.from_hex('FF1493'),
        pitch=annotated_frame)

    cv2.imwrite("output/voronoi_diagram.jpeg", annotated_frame)

    # visualize voronoi diagram with blend
    annotated_frame = draw_pitch(
        config=CONFIG,
        background_color=sv.Color.WHITE,
        line_color=sv.Color.BLACK
    )
    annotated_frame = draw_pitch_voronoi_diagram_2(
        config=CONFIG,
        team_1_xy=pitch_players_xy[players_detections.class_id == 0],
        team_2_xy=pitch_players_xy[players_detections.class_id == 1],
        team_1_color=sv.Color.from_hex('00BFFF'),
        team_2_color=sv.Color.from_hex('FF1493'),
        pitch=annotated_frame)
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.WHITE,
        radius=8,
        thickness=1,
        pitch=annotated_frame)
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_players_xy[players_detections.class_id == 0],
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.WHITE,
        radius=16,
        thickness=1,
        pitch=annotated_frame)
    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_players_xy[players_detections.class_id == 1],
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.WHITE,
        radius=16,
        thickness=1,
        pitch=annotated_frame)

    # cv2.imwrite("output/voronoi_diagram_blend.jpeg", annotated_frame)
    video_writer_voronoi.write(annotated_frame)

# Release the video writers
video_writer_annotated.release()
video_writer_pitch.release()
video_writer_voronoi.release()

