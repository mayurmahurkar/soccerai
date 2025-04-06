import cv2
import numpy as np
import os
import supervision as sv


def get_video_info(video_path):
    """Extract video metadata."""
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.release()
    return frame_width, frame_height, fps, total_frames


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


def get_video_frames_batch(video_path, batch_size, callback=None):
    """
    Generator to yield batches of frames from a video file.
    
    Args:
        video_path: Path to the video file
        batch_size: Number of frames to include in each batch
        callback: Optional callback function to call after each batch
        
    Yields:
        List of frames in the current batch
    """
    frame_generator = sv.get_video_frames_generator(video_path)
    batch_frames = []
    frame_count = 0
    
    for frame in frame_generator:
        batch_frames.append(frame)
        frame_count += 1
        
        if len(batch_frames) == batch_size:
            yield batch_frames, frame_count
            if callback:
                callback(len(batch_frames))
            batch_frames = []
    
    # Yield any remaining frames
    if batch_frames:
        yield batch_frames, frame_count
        if callback:
            callback(len(batch_frames)) 