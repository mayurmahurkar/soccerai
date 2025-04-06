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

SOURCE_VIDEO_PATH = "input/test_10sec.mp4"
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
BATCH_SIZE = 8  # Process 8 frames at a time
FIELD_DETECTION_INTERVAL = 5  # Detect field every 5 frames
CHANGE_THRESHOLD = 0.15  # If frame difference exceeds this, force field detection
MAP_PITCH = False # Flag to enable pitch mapping visualization

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

# Function to check if two frames are significantly different using Mean Squared Error
def frames_differ_significantly(frame1, frame2, threshold=0.15):
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Resize to smaller dimensions for faster processing
    size = (int(frame_width/8), int(frame_height/8))
    gray1 = cv2.resize(gray1, size)
    gray2 = cv2.resize(gray2, size)
    
    # Calculate MSE (Mean Squared Error)
    diff = cv2.absdiff(gray1, gray2)
    mse = np.mean(diff ** 2)
    max_possible_mse = 255 ** 2
    normalized_mse = mse / max_possible_mse
    
    # Higher value means more difference
    return normalized_mse > threshold

# Get frames from original video
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# Field detection state variables
last_keypoints = None
last_frame_for_comparison = None

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
            # Determine which frames need field detection
            frames_needing_field_detection = []
            
            for i, frame in enumerate(batch_frames):
                absolute_frame_num = frame_count - len(batch_frames) + i + 1
                
                # Always detect field on first frame
                if last_keypoints is None:
                    frames_needing_field_detection.append(i)
                    last_frame_for_comparison = frame.copy()
                # Detect on regular intervals
                elif absolute_frame_num % FIELD_DETECTION_INTERVAL == 1:
                    frames_needing_field_detection.append(i)
                    last_frame_for_comparison = frame.copy()
                # Or if frame differs significantly from the last keyframe
                elif last_frame_for_comparison is not None and frames_differ_significantly(
                    frame, last_frame_for_comparison, CHANGE_THRESHOLD):
                    frames_needing_field_detection.append(i)
                    last_frame_for_comparison = frame.copy()
            
            # Run batch prediction for player detection for all frames
            player_results = PLAYER_DETECTION_MODEL.predict(batch_frames, verbose=False)
            
            # Run batch prediction for field detection only on frames that need it
            field_detection_frames = [batch_frames[i] for i in frames_needing_field_detection]
            pitch_results = [None] * len(batch_frames)  # Initialize with None for all frames
            
            if field_detection_frames:
                field_detection_results = FIELD_DETECTION_MODEL.predict(field_detection_frames, verbose=False)
                
                # Assign results to proper indices
                for i, result_idx in enumerate(frames_needing_field_detection):
                    pitch_results[result_idx] = field_detection_results[i]
                    
                    # Process and cache keypoints for this frame
                    key_points = sv.KeyPoints.from_ultralytics(field_detection_results[i])
                    if len(key_points.xy) > 0 and len(key_points.confidence) > 0:
                        filter_indices = key_points.confidence[0] > 0.5
                        if np.any(filter_indices) and len(key_points.xy[0][filter_indices]) >= 4:
                            frame_reference_points = key_points.xy[0][filter_indices]
                            pitch_reference_points = np.array(CONFIG.vertices)[filter_indices]
                            
                            # Store for later use
                            last_keypoints = {
                                'frame_reference_points': frame_reference_points.copy(),
                                'pitch_reference_points': pitch_reference_points.copy(),
                                'confidence': key_points.confidence[0][filter_indices].copy()
                            }
            
            # Process frame detections to collect all player crops for batch prediction
            frame_data = []
            for i, (frame, player_result) in enumerate(zip(batch_frames, player_results)):
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
                        'pitch_result': pitch_results[i]
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
                        'pitch_result': pitch_results[i]
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

                # Create merged player detections for both teams
                players_detections = sv.Detections.merge([
                    data['players_detections'], data['goalkeepers_detections']
                ])

                # Determine pitch transformation - either from current frame or cached
                transformer = None
                
                if data['pitch_result'] is not None:
                    # If we have field detection for this frame, use it directly
                    key_points = sv.KeyPoints.from_ultralytics(data['pitch_result'])
                    if len(key_points.xy) > 0 and len(key_points.confidence) > 0:
                        filter_indices = key_points.confidence[0] > 0.5
                        if np.any(filter_indices) and len(key_points.xy[0][filter_indices]) >= 4:
                            frame_reference_points = key_points.xy[0][filter_indices]
                            pitch_reference_points = np.array(CONFIG.vertices)[filter_indices]
                            
                            if len(frame_reference_points) >= 4 and len(pitch_reference_points) >= 4:
                                # For pitch visualization, we need to swap source and target
                                viz_transformer = ViewTransformer(
                                    source=pitch_reference_points,
                                    target=frame_reference_points
                                )
                                # For player position mapping, keep original transform direction
                                transformer = ViewTransformer(
                                    source=frame_reference_points,
                                    target=pitch_reference_points
                                )
                                
                                # Add pitch mapping visualization if enabled
                                if MAP_PITCH:
                                    # Get all pitch points and transform them to frame coordinates
                                    pitch_all_points = np.array(CONFIG.vertices)
                                    frame_all_points = viz_transformer.transform_points(points=pitch_all_points)
                                    frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])
                                    
                                    # Draw detected keypoints
                                    frame_reference_key_points = sv.KeyPoints(xy=frame_reference_points[np.newaxis, ...])
                                    annotated_frame = vertex_annotator.annotate(
                                        scene=annotated_frame,
                                        key_points=frame_reference_key_points)
                                    
                                    # Draw all pitch lines and points
                                    annotated_frame = edge_annotator.annotate(
                                        scene=annotated_frame,
                                        key_points=frame_all_key_points)
                                    annotated_frame = vertex_annotator_2.annotate(
                                        scene=annotated_frame,
                                        key_points=frame_all_key_points)
                
                elif last_keypoints is not None:
                    # Use cached keypoints from the last detected frame
                    frame_points = last_keypoints['frame_reference_points']
                    pitch_points = last_keypoints['pitch_reference_points']
                    
                    if len(frame_points) >= 4 and len(pitch_points) >= 4:
                        # For pitch visualization, we need to swap source and target
                        viz_transformer = ViewTransformer(
                            source=pitch_points,
                            target=frame_points
                        )
                        # For player position mapping, keep original transform direction
                        transformer = ViewTransformer(
                            source=frame_points,
                            target=pitch_points
                        )
                        
                        # Add pitch mapping visualization if enabled
                        if MAP_PITCH:
                            # Get all pitch points and transform them to frame coordinates
                            pitch_all_points = np.array(CONFIG.vertices)
                            frame_all_points = viz_transformer.transform_points(points=pitch_all_points)
                            frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])
                            
                            # Draw cached keypoints
                            frame_reference_key_points = sv.KeyPoints(xy=frame_points[np.newaxis, ...])
                            annotated_frame = vertex_annotator.annotate(
                                scene=annotated_frame,
                                key_points=frame_reference_key_points)
                            
                            # Draw all pitch lines and points
                            annotated_frame = edge_annotator.annotate(
                                scene=annotated_frame,
                                key_points=frame_all_key_points)
                            annotated_frame = vertex_annotator_2.annotate(
                                scene=annotated_frame,
                                key_points=frame_all_key_points)

                video_writer_annotated.write(annotated_frame)

                # Draw pitch visualization if transformer is available
                if transformer is not None:
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
                    # No field transform available, write blank frames
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