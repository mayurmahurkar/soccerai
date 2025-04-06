import supervision as sv
import numpy as np


def process_player_detections(frame, player_result, tracker, class_ids):
    """
    Process player detection results into structured detections.
    
    Args:
        frame: Video frame
        player_result: Detection results from YOLO model
        tracker: ByteTrack tracker
        class_ids: Dictionary mapping class names to IDs
        
    Returns:
        Dictionary containing processed detections
    """
    # Extract all detections from YOLO results
    detections = sv.Detections.from_ultralytics(player_result)
    
    # Process ball detections
    ball_detections = detections[detections.class_id == class_ids['ball']]
    if len(ball_detections) > 0:
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    # Process player and other detections
    all_detections = detections[detections.class_id != class_ids['ball']]
    
    # Initialize empty detections in case nothing is found
    players_detections = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty((0,)), confidence=np.empty((0,)))
    goalkeepers_detections = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty((0,)), confidence=np.empty((0,)))
    referees_detections = sv.Detections(xyxy=np.empty((0, 4)), class_id=np.empty((0,)), confidence=np.empty((0,)))
    players_crops = None
    
    if len(all_detections) > 0:
        # Apply non-maximum suppression and tracking
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        # Split detections by class
        goalkeepers_detections = all_detections[all_detections.class_id == class_ids['goalkeeper']]
        players_detections = all_detections[all_detections.class_id == class_ids['player']]
        referees_detections = all_detections[all_detections.class_id == class_ids['referee']]

        # Extract crops for team classification
        if len(players_detections) > 0:
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    
    # Return all detections in a structured way
    return {
        'ball_detections': ball_detections,
        'all_detections': all_detections,
        'players_detections': players_detections,
        'goalkeepers_detections': goalkeepers_detections,
        'referees_detections': referees_detections,
        'players_crops': players_crops
    }


def process_batch_detections(batch_frames, player_results, pitch_results, tracker, class_ids):
    """
    Process detection results for a batch of frames.
    
    Args:
        batch_frames: List of video frames
        player_results: List of player detection results
        pitch_results: List of pitch detection results
        tracker: ByteTrack tracker
        class_ids: Dictionary mapping class names to IDs
        
    Returns:
        List of dictionaries containing processed detection data for each frame
    """
    frame_data = []
    
    for i, (frame, player_result) in enumerate(zip(batch_frames, player_results)):
        # Process player detections
        detections = process_player_detections(
            frame=frame,
            player_result=player_result,
            tracker=tracker,
            class_ids=class_ids
        )
        
        # Add the frame and pitch result to the detection data
        detections['frame'] = frame
        detections['pitch_result'] = pitch_results[i]
        
        frame_data.append(detections)
    
    return frame_data


def assign_team_classifications(frame_data, team_predictions):
    """
    Assign team classification results to player detections.
    
    Args:
        frame_data: List of detection data for each frame
        team_predictions: List of lists of team predictions
        
    Returns:
        Updated frame_data with team classifications
    """
    if not team_predictions:
        return frame_data
    
    pred_idx = 0
    for i, data in enumerate(frame_data):
        if data['players_crops'] is not None:
            data['players_detections'].class_id = team_predictions[pred_idx]
            pred_idx += 1
    
    return frame_data 