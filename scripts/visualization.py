import supervision as sv
import numpy as np
import cv2

from scripts.football_pitch import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram, draw_pitch_voronoi_diagram_2
from scripts.annotators import CONFIG, vertex_annotator, edge_annotator, vertex_annotator_2
from scripts.annotators import ellipse_annotator, label_annotator, triangle_annotator, box_annotator
from scripts.utils import resolve_goalkeepers_team_id
from scripts.field_detection import create_transformers
from scripts.view import ViewTransformer


def add_pitch_mapping_visualization(frame, keypoints, viz_transformer, config):
    """
    Add pitch mapping visualization to the frame.
    
    Args:
        frame: Input frame to annotate
        keypoints: Keypoints for the pitch
        viz_transformer: ViewTransformer for visualization
        config: Pitch configuration
        
    Returns:
        Annotated frame with pitch mapping visualization
    """
    if not keypoints or viz_transformer is None:
        return frame
    
    # Get all pitch points and transform them to frame coordinates
    pitch_all_points = np.array(config.vertices)
    frame_all_points = viz_transformer.transform_points(points=pitch_all_points)
    frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])
    
    # Draw detected/reference keypoints
    frame_reference_key_points = sv.KeyPoints(xy=keypoints['frame_reference_points'][np.newaxis, ...])
    annotated_frame = vertex_annotator.annotate(scene=frame.copy(), key_points=frame_reference_key_points)
    
    # Draw all pitch lines and points
    annotated_frame = edge_annotator.annotate(scene=annotated_frame, key_points=frame_all_key_points)
    annotated_frame = vertex_annotator_2.annotate(scene=annotated_frame, key_points=frame_all_key_points)
    
    return annotated_frame


def process_frame_visualizations(frame_data, video_writers, last_keypoints, map_pitch, class_ids, config, enable_voronoi=False):
    """
    Process and visualize each frame in the batch.
    
    Args:
        frame_data: List of dictionaries containing detection results for each frame
        video_writers: Dictionary of video writers for different output videos
        last_keypoints: Cached keypoints from previous frames
        map_pitch: Whether to enable pitch mapping visualization
        class_ids: Dictionary mapping class names to IDs
        config: Pitch configuration
        enable_voronoi: Whether to enable voronoi visualization
    """
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
                    pitch_reference_points = np.array(config.vertices)[filter_indices]
                    
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
                            annotated_frame = add_pitch_mapping_visualization(
                                frame=annotated_frame,
                                keypoints={'frame_reference_points': frame_reference_points},
                                viz_transformer=viz_transformer,
                                config=config
                            )
        
        elif last_keypoints is not None:
            # Use cached keypoints from the last detected frame
            transformer, viz_transformer = create_transformers(last_keypoints)
            
            # Add pitch mapping visualization if enabled
            if map_pitch and viz_transformer is not None:
                annotated_frame = add_pitch_mapping_visualization(
                    frame=annotated_frame,
                    keypoints=last_keypoints,
                    viz_transformer=viz_transformer,
                    config=config
                )

        video_writers["annotated"].write(annotated_frame)
        
        # Only process voronoi visualizations if enabled
        if enable_voronoi and transformer is not None:
            # Create pitch visualization
            visualize_pitch(
                data=data, 
                players_detections=players_detections,
                transformer=transformer,
                video_writers=video_writers,
                config=config
            )
        else:
            # If voronoi is enabled but no transformer is available, write blank frames
            if enable_voronoi and "voronoi" in video_writers:
                blank_frame = np.zeros((data['frame'].shape[0], data['frame'].shape[1], 3), dtype=np.uint8)
                video_writers["voronoi"].write(blank_frame)


def visualize_pitch(data, players_detections, transformer, video_writers, config):
    """
    Create and write pitch visualization with player and ball positions.
    
    Args:
        data: Detection data for the current frame
        players_detections: Merged player detections
        transformer: ViewTransformer for coordinate mapping
        video_writers: Dictionary of video writers
        config: Pitch configuration
    """
    # Only process if voronoi writer exists
    if "voronoi" not in video_writers:
        return
        
    # Create pitch visualization
    pitch_frame = draw_pitch(config)

    # Transform and draw ball positions if any detected
    pitch_ball_xy = None
    if len(data['ball_detections']) > 0:
        frame_ball_xy = data['ball_detections'].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)
        pitch_frame = draw_points_on_pitch(
            config=config,
            xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=pitch_frame)

    # Transform and draw team positions
    pitch_players_xy = None
    if len(players_detections) > 0:
        players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_players_xy = transformer.transform_points(points=players_xy)

        # Draw team 1 players (class_id == 0)
        team1_indices = players_detections.class_id == 0
        if np.any(team1_indices):
            pitch_frame = draw_points_on_pitch(
                config=config,
                xy=pitch_players_xy[team1_indices],
                face_color=sv.Color.from_hex('00BFFF'),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=pitch_frame)

        # Draw team 2 players (class_id == 1)
        team2_indices = players_detections.class_id == 1
        if np.any(team2_indices):
            pitch_frame = draw_points_on_pitch(
                config=config,
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
            config=config,
            xy=pitch_referees_xy,
            face_color=sv.Color.from_hex('FFD700'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=pitch_frame)

    # Create Voronoi diagram if both teams have players detected
    if (pitch_players_xy is not None and len(players_detections) > 0 and 
            np.any(players_detections.class_id == 0) and 
            np.any(players_detections.class_id == 1)):
        
        voronoi_frame = draw_pitch_voronoi_diagram_2(
            config=config,
            team_1_xy=pitch_players_xy[players_detections.class_id == 0],
            team_2_xy=pitch_players_xy[players_detections.class_id == 1],
            team_1_color=sv.Color.from_hex('00BFFF'),
            team_2_color=sv.Color.from_hex('FF1493'),
            opacity=0.5,
            pitch=pitch_frame
        )
        
        # Resize to match original frame dimensions
        voronoi_frame = cv2.resize(voronoi_frame, (data['frame'].shape[1], data['frame'].shape[0]))
        video_writers["voronoi"].write(voronoi_frame)
    else:
        # If we don't have both teams, just write the basic pitch with players
        pitch_frame = cv2.resize(pitch_frame, (data['frame'].shape[1], data['frame'].shape[0]))
        video_writers["voronoi"].write(pitch_frame) 