from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import os
import time
import torch

from scripts.annotators import *
from scripts.utils import *
from scripts.football_pitch import *
from scripts.team import TeamClassifier
from scripts.view import ViewTransformer
from tqdm import tqdm

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

PLAYER_DETECTION_MODEL = YOLO("models/player_detect/weights/best.pt").cuda()
FIELD_DETECTION_MODEL = YOLO("models/last_400.pt").cuda()
team_classifier = TeamClassifier.load("models/team_classifier.pkl")
# Move team classifier to GPU if available
team_classifier.device = "cuda" if torch.cuda.is_available() else "cpu"
team_classifier.features_model = team_classifier.features_model.to(team_classifier.device)

SOURCE_VIDEO_PATH = "input/test.mp4"
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
BATCH_SIZE = 8  # Process 8 frames at a time

tracker = sv.ByteTrack()
tracker.reset()

# Get video info to ensure consistent dimensions
video_capture = cv2.VideoCapture(SOURCE_VIDEO_PATH)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
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

# Process frames in batches for better performance
start_time = time.time()
batch_frames = []
frame_count = 0

with tqdm(total=total_frames, desc="Processing frames") as pbar:
    for frame in frame_generator:
        batch_frames.append(frame)
        frame_count += 1
        
        # Process when we have a full batch or at the end of the video
        if len(batch_frames) == BATCH_SIZE or frame_count == total_frames:
            # Run batch prediction for player detection
            player_results = PLAYER_DETECTION_MODEL.predict(batch_frames, verbose=False)
            
            # Run batch prediction for field detection
            pitch_results = FIELD_DETECTION_MODEL.predict(batch_frames, verbose=False)
            
            # Process frame detections first to collect all player crops for batch prediction
            frame_data = []
            for i, (frame, player_result, pitch_result) in enumerate(zip(batch_frames, player_results, pitch_results)):
                # Ball, goalkeeper, player, referee detection
                detections = sv.Detections.from_ultralytics(player_result)
                
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

                    # Collect crops for team classification
                    players_crops = None
                    if len(players_detections) > 0:
                        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
                    
                    frame_data.append({
                        'frame': frame,
                        'ball_detections': ball_detections,
                        'all_detections': all_detections,
                        'players_detections': players_detections,
                        'goalkeepers_detections': goalkeepers_detections,
                        'referees_detections': referees_detections,
                        'players_crops': players_crops,
                        'pitch_result': pitch_result
                    })
                else:
                    frame_data.append({
                        'frame': frame,
                        'ball_detections': ball_detections,
                        'all_detections': sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty((0,)), confidence=np.empty((0,))),
                        'players_detections': sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty((0,)), confidence=np.empty((0,))),
                        'goalkeepers_detections': sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty((0,)), confidence=np.empty((0,))),
                        'referees_detections': sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty((0,)), confidence=np.empty((0,))),
                        'players_crops': None,
                        'pitch_result': pitch_result
                    })
            
            # Batch predict team classifications
            all_players_crops = [data['players_crops'] for data in frame_data if data['players_crops'] is not None]
            if all_players_crops:
                team_predictions = team_classifier.predict_batch(all_players_crops)
                
                # Assign predictions back to the frames
                pred_idx = 0
                for i, data in enumerate(frame_data):
                    if data['players_crops'] is not None:
                        data['players_detections'].class_id = team_predictions[pred_idx]
                        pred_idx += 1
            
            # Finalize processing for each frame
            for i, data in enumerate(frame_data):
                # Team assignment for goalkeepers
                if len(data['goalkeepers_detections']) > 0 and len(data['players_detections']) > 0:
                    data['goalkeepers_detections'].class_id = resolve_goalkeepers_team_id(
                        data['players_detections'], data['goalkeepers_detections'])

                # Referee ID adjustment
                if len(data['referees_detections']) > 0:
                    data['referees_detections'].class_id -= 1

                # Merge all detections
                all_detections = sv.Detections.merge([
                    data['players_detections'], data['goalkeepers_detections'], data['referees_detections']
                ])

                # Frame visualization for annotated video
                labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]
                all_detections.class_id = all_detections.class_id.astype(int)

                annotated_frame = data['frame'].copy()
                annotated_frame = ellipse_annotator.annotate(
                    scene=annotated_frame,
                    detections=all_detections)
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=all_detections,
                    labels=labels)
                annotated_frame = triangle_annotator.annotate(
                    scene=annotated_frame,
                    detections=data['ball_detections'])

                video_writer_annotated.write(annotated_frame)

                # Create merged player detections for both teams
                players_detections = sv.Detections.merge([
                    data['players_detections'], data['goalkeepers_detections']
                ])

                # Process pitch key points
                key_points = sv.KeyPoints.from_ultralytics(data['pitch_result'])

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
                            if len(data['ball_detections']) > 0:
                                frame_ball_xy = data['ball_detections'].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
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
                            if len(data['referees_detections']) > 0:
                                referees_xy = data['referees_detections'].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
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
                                if len(data['ball_detections']) > 0:
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
                    
                pbar.update(1)
                
            # Clear the batch for the next round
            batch_frames = []

end_time = time.time()
processing_time = end_time - start_time
print(f"Total processing time: {processing_time:.2f} seconds")
print(f"Average FPS: {total_frames / processing_time:.2f}")

# Explicitly release the video writers
video_writer_annotated.release()
video_writer_pitch.release()
video_writer_voronoi.release()

print(f"Processing complete. Videos saved to output directory:")
print(f" - {output_video_path_annotated}")
print(f" - {output_video_path_pitch}")
print(f" - {output_video_path_voronoi}")