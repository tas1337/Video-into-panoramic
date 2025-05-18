import cv2
import os
import numpy as np

def extract_keyframes(video_path, output_folder, frame_interval=5):
    """
    Extracts keyframes from a video at a given interval and saves them as images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_paths = []
    frame_id = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            count += 1

        frame_id += 1

    cap.release()
    return frame_paths

def crop_black_borders(image):
    """
    Crops black borders from the stitched panorama.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    x, y, w, h = cv2.boundingRect(thresh)
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

def stitch_images_with_opencv(image_paths):
    """
    Uses OpenCV's built-in Stitcher class to create a panoramic image.
    """
    images = [cv2.imread(img) for img in image_paths]
    
    # Check if images were loaded properly
    if any(img is None for img in images):
        print("Error: Some images were not loaded correctly.")
        return None

    # Initialize stitcher
    stitcher = cv2.Stitcher_create()
    status, panorama = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        print(f"Error stitching images: {status}")
        return None

    panorama = crop_black_borders(panorama)
    panorama_resized = cv2.resize(panorama, (1024, 512), interpolation=cv2.INTER_AREA)

    return panorama_resized

def create_panorama(video_path, output_image, frame_interval=5):
    """
    Extracts keyframes from a video and creates a panoramic image.
    """
    frames_folder = "extracted_frames"
    
    print("Extracting keyframes...")
    keyframe_paths = extract_keyframes(video_path, frames_folder, frame_interval)

    if not keyframe_paths:
        print("No keyframes found.")
        return

    print("Stitching images...")
    panorama = stitch_images_with_opencv(keyframe_paths)

    if panorama is not None:
        # Convert to HDR format (EXR) and save
        hdr_image = np.float32(panorama) / 255.0
        cv2.imwrite(output_image, hdr_image)

        print(f"Panoramic HDR image saved as {output_image}")
    else:
        print("Panorama stitching failed.")

if __name__ == "__main__":
    video_file = "input.mp4"  # Change this to your video file
    output_panorama = "panorama.hdr"  # Use EXR format for HDR
    create_panorama(video_file, output_panorama, frame_interval=5)
