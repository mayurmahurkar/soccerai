import supervision as sv
import numpy as np
from scripts.view import ViewTransformer
from scripts.video_utils import frames_differ_significantly


def process_field_detection(frame, field_detection_model, last_keypoints, config):
    """
    Process field detection and extract keypoints.
    
    Args:
        frame: Video frame to process
        field_detection_model: YOLO model for field detection
        last_keypoints: Previously detected keypoints (if any)
        config: Pitch configuration with vertex data
        
    Returns:
        Tuple of (updated keypoints, field detection result)
    """
    field_result = field_detection_model.predict(frame, verbose=False)[0]
    key_points = sv.KeyPoints.from_ultralytics(field_result)
    
    if len(key_points.xy) > 0 and len(key_points.confidence) > 0:
        filter_indices = key_points.confidence[0] > 0.5
        if np.any(filter_indices) and len(key_points.xy[0][filter_indices]) >= 4:
            frame_reference_points = key_points.xy[0][filter_indices]
            pitch_reference_points = np.array(config.vertices)[filter_indices]
            
            # Store for later use
            return {
                'frame_reference_points': frame_reference_points.copy(),
                'pitch_reference_points': pitch_reference_points.copy(),
                'confidence': key_points.confidence[0][filter_indices].copy()
            }, field_result
    
    return last_keypoints, field_result


def create_transformers(keypoints):
    """
    Create view transformers from keypoints.
    
    Args:
        keypoints: Dictionary of keypoints (frame_reference_points and pitch_reference_points)
        
    Returns:
        Tuple of (frame-to-pitch transformer, pitch-to-frame transformer)
    """
    if keypoints is None or 'frame_reference_points' not in keypoints:
        return None, None
    
    frame_points = keypoints['frame_reference_points']
    pitch_points = keypoints['pitch_reference_points']
    
    if len(frame_points) < 4 or len(pitch_points) < 4:
        return None, None
    
    # For player position mapping (frame to pitch)
    transformer = ViewTransformer(source=frame_points, target=pitch_points)
    
    # For pitch visualization (pitch to frame)
    viz_transformer = ViewTransformer(source=pitch_points, target=frame_points)
    
    return transformer, viz_transformer


def determine_frames_needing_detection(batch_frames, frame_count, batch_size, 
                                      last_keypoints, last_frame_for_comparison,
                                      field_detection_interval, change_threshold):
    """
    Determine which frames in a batch need field detection.
    
    Args:
        batch_frames: List of frames in the current batch
        frame_count: Total number of frames processed so far
        batch_size: Size of the batch
        last_keypoints: Previously detected keypoints (if any)
        last_frame_for_comparison: Previous frame used for comparison
        field_detection_interval: Interval for regular field detection
        change_threshold: Threshold for significant frame difference
        
    Returns:
        Tuple of (list of indices needing detection, updated comparison frame)
    """
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
    
    return frames_needing_field_detection, last_frame_for_comparison


def process_batch_field_detection(batch_frames, frames_needing_field_detection, 
                                field_model, last_keypoints, config):
    """
    Run field detection on selected frames in a batch.
    
    Args:
        batch_frames: List of frames in the current batch
        frames_needing_field_detection: List of indices needing field detection
        field_model: YOLO model for field detection
        last_keypoints: Previously detected keypoints (if any)
        config: Pitch configuration
        
    Returns:
        Tuple of (results list, updated keypoints)
    """
    # Initialize results with None for all frames
    pitch_results = [None] * len(batch_frames)
    
    # Process only frames that need field detection
    if frames_needing_field_detection:
        field_detection_frames = [batch_frames[i] for i in frames_needing_field_detection]
        field_detection_results = field_model.predict(field_detection_frames, verbose=False)
        
        # Assign results to proper indices
        for i, result_idx in enumerate(frames_needing_field_detection):
            pitch_results[result_idx] = field_detection_results[i]
            
            # Process and cache keypoints for this frame
            updated_keypoints, _ = process_field_detection(
                frame=batch_frames[result_idx],
                field_detection_model=field_model, 
                last_keypoints=last_keypoints,
                config=config
            )
            
            # Only update if we got valid keypoints and they're different from last ones
            if updated_keypoints is not None:
                if last_keypoints is None:
                    last_keypoints = updated_keypoints
                else:
                    # Check if arrays have the same shape
                    if (updated_keypoints['frame_reference_points'].shape != last_keypoints['frame_reference_points'].shape or
                        updated_keypoints['pitch_reference_points'].shape != last_keypoints['pitch_reference_points'].shape or
                        updated_keypoints['confidence'].shape != last_keypoints['confidence'].shape):
                        # If shapes are different, update the keypoints
                        last_keypoints = updated_keypoints
                    else:
                        # Only compare if shapes match
                        frame_points_diff = np.any(updated_keypoints['frame_reference_points'] != last_keypoints['frame_reference_points'])
                        pitch_points_diff = np.any(updated_keypoints['pitch_reference_points'] != last_keypoints['pitch_reference_points'])
                        confidence_diff = np.any(updated_keypoints['confidence'] != last_keypoints['confidence'])
                        
                        if frame_points_diff or pitch_points_diff or confidence_diff:
                            last_keypoints = updated_keypoints
    
    return pitch_results, last_keypoints 