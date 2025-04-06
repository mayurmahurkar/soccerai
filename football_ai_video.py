from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import os
import time
import torch
import yaml
from tqdm import tqdm

from scripts.annotators import *
from scripts.utils import resolve_goalkeepers_team_id
from scripts.football_pitch import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram_2
from scripts.team import TeamClassifier
from scripts.view import ViewTransformer


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)


def load_models(config, device):
    """Load all required models."""
    player_model = YOLO(config["PLAYER_DETECTION_MODEL"]).to(device)
    field_model = YOLO(config["FIELD_DETECTION_MODEL"]).to(device)
    
    team_classifier = TeamClassifier.load(config["TEAM_CLASSIFIER_MODEL"])
    team_classifier.device = device
    team_classifier.features_model = team_classifier.features_model.to(device)
    
    return player_model, field_model, team_classifier


def get_video_info(video_path):
    """Extract video metadata."""
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.release()
    return frame_width, frame_height, fps, total_frames


def setup_video_writers(output_dir, codec, ext, frame_width, frame_height, fps):
    """Setup video writers with fallback options if codec fails."""
    output_paths = {
        "annotated": f"{output_dir}/annotated_video.{ext}",
        "pitch": f"{output_dir}/annotated_pitch_video.{ext}",
        "voronoi": f"{output_dir}/voronoi_diagram_video.{ext}"
    }
    
    # Try with the specified codec
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Test if the codec works
        test_path = f"{output_dir}/test.{ext}"
        test_writer = cv2.VideoWriter(test_path, fourcc, fps, (frame_width, frame_height))
        if not test_writer.isOpened():
            raise Exception(f"{codec} codec not supported")
        test_writer.release()
        os.remove(test_path)
        
    except Exception as e:
        print(f"Failed with {codec} codec: {e}")
        # Fallback to MJPG
        try:
            codec = "MJPG"
            ext = "avi"
            fourcc = cv2.VideoWriter_fourcc(*codec)
            
            # Update output paths
            for key in output_paths:
                output_paths[key] = f"{output_dir}/{key}_video.{ext}"
                
            # Test if MJPG works
            test_path = f"{output_dir}/test.{ext}"
            test_writer = cv2.VideoWriter(test_path, fourcc, fps, (frame_width, frame_height))
            if not test_writer.isOpened():
                raise Exception("MJPG codec not supported")
            test_writer.release()
            os.remove(test_path)
            
        except Exception:
            # Last resort: uncompressed
            print("Falling back to uncompressed video")
            codec = "0"
            fourcc = 0
            ext = "avi"
            
            # Update output paths again
            for key in output_paths:
                output_paths[key] = f"{output_dir}/{key}_video.{ext}"
    
    print(f"Using codec: {codec}, extension: {ext}")
    
    # Create the actual video writers
    writers = {
        "annotated": cv2.VideoWriter(output_paths["annotated"], fourcc, fps, (frame_width, frame_height)),
        "pitch": cv2.VideoWriter(output_paths["pitch"], fourcc, fps, (frame_width, frame_height)),
        "voronoi": cv2.VideoWriter(output_paths["voronoi"], fourcc, fps, (frame_width, frame_height))
    }
    
    # Final check
    for name, writer in writers.items():
        if not writer.isOpened():
            raise Exception(f"Failed to open {name} video writer with selected codec")
    
    return writers, output_paths


def frames_differ_significantly(frame1, frame2, threshold=0.15, scale_factor=8):
    """Check if two frames are significantly different using Mean Squared Error."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Resize to smaller dimensions for faster processing
    size = (frame1.shape[1] // scale_factor, frame1.shape[0] // scale_factor)
    gray1 = cv2.resize(gray1, size)
    gray2 = cv2.resize(gray2, size)
    
    # Calculate MSE (Mean Squared Error)
    diff = cv2.absdiff(gray1, gray2)
    mse = np.mean(diff ** 2)
    max_possible_mse = 255 ** 2
    normalized_mse = mse / max_possible_mse
    
    # Higher value means more difference
    return normalized_mse > threshold


def process_field_detection(frame, field_detection_model, last_keypoints):
    """Process field detection and extract keypoints."""
    field_result = field_detection_model.predict(frame, verbose=False)[0]
    key_points = sv.KeyPoints.from_ultralytics(field_result)
    
    if len(key_points.xy) > 0 and len(key_points.confidence) > 0:
        filter_indices = key_points.confidence[0] > 0.5
        if np.any(filter_indices) and len(key_points.xy[0][filter_indices]) >= 4:
            frame_reference_points = key_points.xy[0][filter_indices]
            pitch_reference_points = np.array(CONFIG.vertices)[filter_indices]
            
            # Store for later use
            return {
                'frame_reference_points': frame_reference_points.copy(),
                'pitch_reference_points': pitch_reference_points.copy(),
                'confidence': key_points.confidence[0][filter_indices].copy()
            }, field_result
    
    return last_keypoints, field_result


def create_transformers(keypoints):
    """Create view transformers from keypoints."""
    if keypoints is None or 'frame_reference_points' not in keypoints:
        return None, None
    
    frame_points = keypoints['frame_reference_points']
    pitch_points = keypoints['pitch_reference_points']
    
    if len(frame_points) < 4 or len(pitch_points) < 4:
        return None, None
    
    # For pitch visualization (pitch to frame)
    viz_transformer = ViewTransformer(source=pitch_points, target=frame_points)
    
    # For player position mapping (frame to pitch)
    transformer = ViewTransformer(source=frame_points, target=pitch_points)
    
    return transformer, viz_transformer


def add_pitch_mapping_visualization(frame, keypoints, viz_transformer):
    """Add pitch mapping visualization to the frame."""
    if not keypoints or viz_transformer is None:
        return frame
    
    # Get all pitch points and transform them to frame coordinates
    pitch_all_points = np.array(CONFIG.vertices)
    frame_all_points = viz_transformer.transform_points(points=pitch_all_points)
    frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])
    
    # Draw detected/reference keypoints
    frame_reference_key_points = sv.KeyPoints(xy=keypoints['frame_reference_points'][np.newaxis, ...])
    annotated_frame = vertex_annotator.annotate(scene=frame.copy(), key_points=frame_reference_key_points)
    
    # Draw all pitch lines and points
    annotated_frame = edge_annotator.annotate(scene=annotated_frame, key_points=frame_all_key_points)
    annotated_frame = vertex_annotator_2.annotate(scene=annotated_frame, key_points=frame_all_key_points)
    
    return annotated_frame


def process_batch(
    batch_frames, 
    frame_count, 
    player_model, 
    field_model, 
    team_classifier, 
    tracker, 
    last_keypoints, 
    last_frame_for_comparison,
    field_detection_interval,
    change_threshold,
    map_pitch,
    class_ids
):
    """Process a batch of frames."""
    # Determine which frames need field detection
    frames_needing_field_detection = []
    
    for i, frame in enumerate(batch_frames):
        absolute_frame_num = frame_count - len(batch_frames) + i + 1
        
        # Always detect field on first frame
        if last_keypoints is None:
            frames_needing_field_detection.append(i)
            last_frame_for_comparison = frame.copy()
        # Detect on regular intervals
        elif absolute_frame_num % field_detection_interval == 1:
            frames_needing_field_detection.append(i)
            last_frame_for_comparison = frame.copy()
        # Or if frame differs significantly from the last keyframe
        elif last_frame_for_comparison is not None and frames_differ_significantly(
            frame, last_frame_for_comparison, change_threshold):
            frames_needing_field_detection.append(i)
            last_frame_for_comparison = frame.copy()
    
    # Run batch prediction for player detection for all frames
    player_results = player_model.predict(batch_frames, verbose=False)
    
    # Run batch prediction for field detection only on frames that need it
    field_detection_frames = [batch_frames[i] for i in frames_needing_field_detection]
    pitch_results = [None] * len(batch_frames)  # Initialize with None for all frames
    
    if field_detection_frames:
        field_detection_results = field_model.predict(field_detection_frames, verbose=False)
        
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
        
        ball_detections = detections[detections.class_id == class_ids['ball']]
        if len(ball_detections) > 0:
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != class_ids['ball']]
        if len(all_detections) > 0:
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = tracker.update_with_detections(detections=all_detections)

            goalkeepers_detections = all_detections[all_detections.class_id == class_ids['goalkeeper']]
            players_detections = all_detections[all_detections.class_id == class_ids['player']]
            referees_detections = all_detections[all_detections.class_id == class_ids['referee']]

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
    
    return frame_data, last_keypoints, last_frame_for_comparison


def process_frame_visualizations(frame_data, video_writers, last_keypoints, map_pitch, class_ids):
    """Process and visualize each frame in the batch."""
    for data in frame_data:
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
        viz_transformer = None
        
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
                        if map_pitch:
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
            transformer, viz_transformer = create_transformers(last_keypoints)
            
            # Add pitch mapping visualization if enabled
            if map_pitch and viz_transformer is not None:
                # Get all pitch points and transform them to frame coordinates
                pitch_all_points = np.array(CONFIG.vertices)
                frame_all_points = viz_transformer.transform_points(points=pitch_all_points)
                frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])
                
                # Draw cached keypoints
                frame_reference_key_points = sv.KeyPoints(xy=last_keypoints['frame_reference_points'][np.newaxis, ...])
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

        video_writers["annotated"].write(annotated_frame)

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
            pitch_frame = cv2.resize(pitch_frame, (data['frame'].shape[1], data['frame'].shape[0]))
            video_writers["pitch"].write(pitch_frame)

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
                voronoi_frame = cv2.resize(voronoi_frame, (data['frame'].shape[1], data['frame'].shape[0]))
                video_writers["voronoi"].write(voronoi_frame)
            else:
                # If we don't have both teams, just write a copy of the pitch view
                video_writers["voronoi"].write(pitch_frame)
        else:
            # No field transform available, write blank frames
            blank_frame = np.zeros((data['frame'].shape[0], data['frame'].shape[1], 3), dtype=np.uint8)
            video_writers["pitch"].write(blank_frame)
            video_writers["voronoi"].write(blank_frame)


def main():
    """Main function to process the soccer video."""
    # Load config
    config = load_config()
    
    # Setup
    setup_directories(config["OUTPUT_DIR"])
    
    # Determine device
    device = config["DEVICE"] if torch.cuda.is_available() else "cpu"
    
    # Load models
    player_model, field_model, team_classifier = load_models(config, device)
    
    # Setup tracker
    tracker = sv.ByteTrack()
    tracker.reset()
    
    # Get video info
    frame_width, frame_height, fps, total_frames = get_video_info(config["SOURCE_VIDEO_PATH"])
    
    # Setup video writers
    video_writers, output_paths = setup_video_writers(
        config["OUTPUT_DIR"], 
        config["VIDEO_CODEC"], 
        config["VIDEO_EXTENSION"],
        frame_width, 
        frame_height, 
        fps
    )
    
    # Class IDs mapping
    class_ids = {
        "ball": config["BALL_ID"],
        "goalkeeper": config["GOALKEEPER_ID"],
        "player": config["PLAYER_ID"],
        "referee": config["REFEREE_ID"]
    }
    
    # Get frames from original video
    frame_generator = sv.get_video_frames_generator(config["SOURCE_VIDEO_PATH"])
    
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
            if len(batch_frames) == config["BATCH_SIZE"] or frame_count == total_frames:
                # Process batch
                frame_data, last_keypoints, last_frame_for_comparison = process_batch(
                    batch_frames, 
                    frame_count, 
                    player_model, 
                    field_model, 
                    team_classifier, 
                    tracker, 
                    last_keypoints, 
                    last_frame_for_comparison,
                    config["FIELD_DETECTION_INTERVAL"],
                    config["CHANGE_THRESHOLD"],
                    config["MAP_PITCH"],
                    class_ids
                )
                
                # Process visualizations for each frame
                process_frame_visualizations(
                    frame_data, 
                    video_writers, 
                    last_keypoints, 
                    config["MAP_PITCH"],
                    class_ids
                )
                
                # Update progress bar
                pbar.update(len(batch_frames))
                
                # Clear the batch for the next round
                batch_frames = []
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Average FPS: {total_frames / processing_time:.2f}")
    
    # Explicitly release the video writers
    for writer in video_writers.values():
        writer.release()
    
    print(f"Processing complete. Videos saved to output directory:")
    for name, path in output_paths.items():
        print(f" - {path}")


if __name__ == "__main__":
    main()