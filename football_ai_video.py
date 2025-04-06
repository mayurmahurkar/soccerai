from ultralytics import YOLO
import supervision as sv
import cv2
import os
import time
import torch
from tqdm import tqdm

from scripts.annotators import CONFIG
from scripts.utils import load_config, setup_directories
from scripts.video_utils import get_video_info, setup_video_writers, get_video_frames_batch
from scripts.team import TeamClassifier
from scripts.field_detection import determine_frames_needing_detection, process_batch_field_detection
from scripts.detection import process_batch_detections, assign_team_classifications
from scripts.visualization import process_frame_visualizations


def load_models(config, device):
    """Load all required models."""
    player_model = YOLO(config["PLAYER_DETECTION_MODEL"]).to(device)
    field_model = YOLO(config["FIELD_DETECTION_MODEL"]).to(device)
    
    team_classifier = TeamClassifier.load(config["TEAM_CLASSIFIER_MODEL"])
    team_classifier.device = device
    team_classifier.features_model = team_classifier.features_model.to(device)
    
    return player_model, field_model, team_classifier


def process_batch(batch_frames, frame_count, player_model, field_model, team_classifier, 
                  tracker, last_keypoints, last_frame_for_comparison, config, class_ids):
    """Process a batch of frames."""
    # Determine which frames need field detection
    frames_needing_field_detection, last_frame_for_comparison = determine_frames_needing_detection(
        batch_frames, 
        frame_count, 
        config["BATCH_SIZE"],
        last_keypoints, 
        last_frame_for_comparison,
        config["FIELD_DETECTION_INTERVAL"],
        config["CHANGE_THRESHOLD"]
    )
    
    # Run batch prediction for player detection for all frames
    player_results = player_model.predict(batch_frames, verbose=False)
    
    # Run batch prediction for field detection only on frames that need it
    pitch_results, last_keypoints = process_batch_field_detection(
        batch_frames, 
        frames_needing_field_detection, 
        field_model, 
        last_keypoints, 
        CONFIG
    )
    
    # Process frame detections to collect all player crops for batch prediction
    frame_data = process_batch_detections(
        batch_frames, 
        player_results, 
        pitch_results, 
        tracker, 
        class_ids
    )
    
    # Batch predict team classifications
    all_players_crops = [data['players_crops'] for data in frame_data if data['players_crops'] is not None]
    team_predictions = []
    
    if all_players_crops:
        team_predictions = team_classifier.predict_batch(all_players_crops)
        
        # Assign predictions back to the frames
        frame_data = assign_team_classifications(frame_data, team_predictions)
    
    return frame_data, last_keypoints, last_frame_for_comparison


def main():
    """Main function to process the soccer video."""
    # Load config
    config = load_config()
    
    # Setup output directory
    setup_directories(config["OUTPUT_DIR"])
    
    # Determine device
    device = config["DEVICE"] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    player_model, field_model, team_classifier = load_models(config, device)
    
    # Setup tracker
    tracker = sv.ByteTrack()
    tracker.reset()
    
    # Get video info
    frame_width, frame_height, fps, total_frames = get_video_info(config["SOURCE_VIDEO_PATH"])
    print(f"Video info: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
    
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
    
    # Field detection state variables
    last_keypoints = None
    last_frame_for_comparison = None
    
    # Process frames in batches for better performance
    start_time = time.time()
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        # Use our batch processing generator
        for batch_frames, frame_count in get_video_frames_batch(
            config["SOURCE_VIDEO_PATH"], 
            config["BATCH_SIZE"],
            callback=pbar.update
        ):
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
                config,
                class_ids
            )
            
            # Process visualizations for each frame
            process_frame_visualizations(
                frame_data, 
                video_writers, 
                last_keypoints, 
                config["MAP_PITCH"],
                class_ids,
                CONFIG
            )
    
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