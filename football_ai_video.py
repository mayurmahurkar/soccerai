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

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

PLAYER_DETECTION_MODEL = YOLO("models/player_detect/weights/best.pt")
FIELD_DETECTION_MODEL = YOLO("models/last_400.pt")
team_classifier = TeamClassifier.load("models/team_classifier.pkl")

SOURCE_VIDEO_PATH = "input/test.mp4"
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

tracker = sv.ByteTrack()
tracker.reset()

# Get video info to ensure consistent dimensions
video_capture = cv2.VideoCapture(SOURCE_VIDEO_PATH)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)
video_capture.release()

# Setup video writers with a codec that should be widely available
output_video_path_annotated = "output/annotated_video.mp4"
output_video_path_pitch = "output/annotated_pitch_video.mp4"
output_video_path_voronoi = "output/voronoi_diagram_video.mp4"

# Try different codec options
try:
    # Try XVID codec first (widely supported)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Set file extensions to .avi for XVID
    output_video_path_annotated = "output/annotated_video.avi"
    output_video_path_pitch = "output/annotated_pitch_video.avi"
    output_video_path_voronoi = "output/voronoi_diagram_video.avi"
    
    # Create test writer to verify codec works
    test_writer = cv2.VideoWriter("output/test.avi", fourcc, fps, (frame_width, frame_height))
    if not test_writer.isOpened():
        raise Exception("XVID codec not supported")
    test_writer.release()
    os.remove("output/test.avi")
    
except Exception:
    # Fall back to MJPG if XVID fails
    try:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # Set file extensions to .avi for MJPG
        output_video_path_annotated = "output/annotated_video.avi"
        output_video_path_pitch = "output/annotated_pitch_video.avi"
        output_video_path_voronoi = "output/voronoi_diagram_video.avi"
        
        # Create test writer to verify codec works
        test_writer = cv2.VideoWriter("output/test.avi", fourcc, fps, (frame_width, frame_height))
        if not test_writer.isOpened():
            raise Exception("MJPG codec not supported")
        test_writer.release()
        os.remove("output/test.avi")
        
    except Exception:
        # Last resort: try with uncompressed codec
        fourcc = 0
        # Set file extensions to .avi
        output_video_path_annotated = "output/annotated_video.avi"
        output_video_path_pitch = "output/annotated_pitch_video.avi"
        output_video_path_voronoi = "output/voronoi_diagram_video.avi"

print(f"Using codec: {fourcc}")

# Create video writers for each output with same dimensions
video_writer_annotated = cv2.VideoWriter(output_video_path_annotated, fourcc, fps, (frame_width, frame_height))
video_writer_pitch = cv2.VideoWriter(output_video_path_pitch, fourcc, fps, (frame_width, frame_height))
video_writer_voronoi = cv2.VideoWriter(output_video_path_voronoi, fourcc, fps, (frame_width, frame_height))

# Ensure the video writers were opened successfully
if not video_writer_annotated.isOpened() or not video_writer_pitch.isOpened() or not video_writer_voronoi.isOpened():
    print("Failed to open one or more video writers with selected codec.")
    print("Trying with default codec (0)...")
    
    # Try with default codec as a last resort
    fourcc = 0
    video_writer_annotated = cv2.VideoWriter(output_video_path_annotated, fourcc, fps, (frame_width, frame_height))
    video_writer_pitch = cv2.VideoWriter(output_video_path_pitch, fourcc, fps, (frame_width, frame_height))
    video_writer_voronoi = cv2.VideoWriter(output_video_path_voronoi, fourcc, fps, (frame_width, frame_height))
    
    if not video_writer_annotated.isOpened() or not video_writer_pitch.isOpened() or not video_writer_voronoi.isOpened():
        raise Exception("Failed to open video writers with any codec. Check OpenCV installation.")

# Get frames from original video
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Process frames with a progress bar
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) if 'video_capture' in locals() else 0
for frame in tqdm(frame_generator, total=total_frames, desc="Processing frames"):
    # Ball, goalkeeper, player, referee detection
    result = PLAYER_DETECTION_MODEL.predict(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    ball_detections = detections[detections.class_id == BALL_ID]
    if len(ball_detections) > 0:
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    all_detections = detections[detections.class_id != BALL_ID]
    if len(all_detections) > 0:
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

        # Team assignment
        if len(players_detections) > 0:
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            players_detections.class_id = team_classifier.predict(players_crops)

            if len(goalkeepers_detections) > 0:
                goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
                    players_detections, goalkeepers_detections)

        if len(referees_detections) > 0:
            referees_detections.class_id -= 1

        all_detections = sv.Detections.merge([
            players_detections, goalkeepers_detections, referees_detections
        ])

        # Frame visualization for annotated video
        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]
        all_detections.class_id = all_detections.class_id.astype(int)
    else:
        players_detections = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty((0,)), confidence=np.empty((0,)))
        goalkeepers_detections = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty((0,)), confidence=np.empty((0,)))
        referees_detections = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty((0,)), confidence=np.empty((0,)))
        all_detections = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty((0,)), confidence=np.empty((0,)))
        labels = []

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

    video_writer_annotated.write(annotated_frame)

    # Create merged player detections for both teams
    players_detections = sv.Detections.merge([
        players_detections, goalkeepers_detections
    ])

    # Detect pitch key points
    pitch_result = FIELD_DETECTION_MODEL.predict(frame, verbose=False)[0]
    key_points = sv.KeyPoints.from_ultralytics(pitch_result)

    # Make sure we have enough key points for transformation
    if len(key_points.xy) > 0 and len(key_points.confidence) > 0:
        filter_indices = key_points.confidence[0] > 0.5
        if np.any(filter_indices):
            frame_reference_points = key_points.xy[0][filter_indices]
            pitch_reference_points = np.array(CONFIG.vertices)[filter_indices]

            # Only proceed if we have enough reference points
            if len(frame_reference_points) >= 4 and len(pitch_reference_points) >= 4:
                transformer = ViewTransformer(
                    source=frame_reference_points,
                    target=pitch_reference_points
                )

                # Create pitch visualization
                pitch_frame = draw_pitch(CONFIG)

                # Transform and draw ball positions if any detected
                if len(ball_detections) > 0:
                    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)
                    pitch_frame = draw_points_on_pitch(
                        config=CONFIG,
                        xy=pitch_ball_xy,
                        face_color=sv.Color.WHITE,
                        edge_color=sv.Color.BLACK,
                        radius=10,
                        pitch=pitch_frame)

                # Transform and draw team positions
                if len(players_detections) > 0:
                    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    pitch_players_xy = transformer.transform_points(points=players_xy)

                    # Draw team 1 players (class_id == 0)
                    team1_indices = players_detections.class_id == 0
                    if np.any(team1_indices):
                        pitch_frame = draw_points_on_pitch(
                            config=CONFIG,
                            xy=pitch_players_xy[team1_indices],
                            face_color=sv.Color.from_hex('00BFFF'),
                            edge_color=sv.Color.BLACK,
                            radius=16,
                            pitch=pitch_frame)

                    # Draw team 2 players (class_id == 1)
                    team2_indices = players_detections.class_id == 1
                    if np.any(team2_indices):
                        pitch_frame = draw_points_on_pitch(
                            config=CONFIG,
                            xy=pitch_players_xy[team2_indices],
                            face_color=sv.Color.from_hex('FF1493'),
                            edge_color=sv.Color.BLACK,
                            radius=16,
                            pitch=pitch_frame)

                # Draw referees
                if len(referees_detections) > 0:
                    referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    pitch_referees_xy = transformer.transform_points(points=referees_xy)
                    pitch_frame = draw_points_on_pitch(
                        config=CONFIG,
                        xy=pitch_referees_xy,
                        face_color=sv.Color.from_hex('FFD700'),
                        edge_color=sv.Color.BLACK,
                        radius=16,
                        pitch=pitch_frame)

                # Resize pitch_frame to match original frame dimensions
                pitch_frame = cv2.resize(pitch_frame, (frame_width, frame_height))
                video_writer_pitch.write(pitch_frame)

                # Create Voronoi diagram visualization
                if len(players_detections) > 0 and np.any(players_detections.class_id == 0) and np.any(players_detections.class_id == 1):
                    voronoi_frame = draw_pitch(
                        config=CONFIG,
                        background_color=sv.Color.WHITE,
                        line_color=sv.Color.BLACK
                    )
                    
                    voronoi_frame = draw_pitch_voronoi_diagram_2(
                        config=CONFIG,
                        team_1_xy=pitch_players_xy[players_detections.class_id == 0],
                        team_2_xy=pitch_players_xy[players_detections.class_id == 1],
                        team_1_color=sv.Color.from_hex('00BFFF'),
                        team_2_color=sv.Color.from_hex('FF1493'),
                        pitch=voronoi_frame
                    )

                    # Add players and ball to the Voronoi diagram
                    if len(ball_detections) > 0:
                        voronoi_frame = draw_points_on_pitch(
                            config=CONFIG,
                            xy=pitch_ball_xy,
                            face_color=sv.Color.WHITE,
                            edge_color=sv.Color.WHITE,
                            radius=8,
                            thickness=1,
                            pitch=voronoi_frame
                        )

                    if np.any(players_detections.class_id == 0):
                        voronoi_frame = draw_points_on_pitch(
                            config=CONFIG,
                            xy=pitch_players_xy[players_detections.class_id == 0],
                            face_color=sv.Color.from_hex('00BFFF'),
                            edge_color=sv.Color.WHITE,
                            radius=16,
                            thickness=1,
                            pitch=voronoi_frame
                        )
                    
                    if np.any(players_detections.class_id == 1):
                        voronoi_frame = draw_points_on_pitch(
                            config=CONFIG,
                            xy=pitch_players_xy[players_detections.class_id == 1],
                            face_color=sv.Color.from_hex('FF1493'),
                            edge_color=sv.Color.WHITE,
                            radius=16,
                            thickness=1,
                            pitch=voronoi_frame
                        )

                    # Resize voronoi_frame to match original frame dimensions
                    voronoi_frame = cv2.resize(voronoi_frame, (frame_width, frame_height))
                    video_writer_voronoi.write(voronoi_frame)
                else:
                    # If we don't have both teams, just write a copy of the pitch view
                    video_writer_voronoi.write(pitch_frame)
            else:
                # Not enough reference points, write blank frames
                blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                video_writer_pitch.write(blank_frame)
                video_writer_voronoi.write(blank_frame)
        else:
            # No valid key points detected, write blank frames
            blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            video_writer_pitch.write(blank_frame)
            video_writer_voronoi.write(blank_frame)
    else:
        # No key points detected, write blank frames
        blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        video_writer_pitch.write(blank_frame)
        video_writer_voronoi.write(blank_frame)

# Explicitly release the video writers
video_writer_annotated.release()
video_writer_pitch.release()
video_writer_voronoi.release()

print(f"Processing complete. Videos saved to output directory:")
print(f" - {output_video_path_annotated}")
print(f" - {output_video_path_pitch}")
print(f" - {output_video_path_voronoi}")