import yt_dlp
import os
import cv2

# Replace this URL with the YouTube video URL you want to download
video_url = "https://www.youtube.com/watch?v=eO6GRU4RfC4&ab_channel=RVILtdRemoteVisualInspections"
download_path = "downloaded_video.mp4"  # Specify the filename to save the video

# Download video using yt-dlp
ydl_opts = {
    'format': 'best',
    'outtmpl': download_path,  # Save video with this filename
}

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    print(f"Video downloaded and saved at: {download_path}")
except Exception as e:
    print(f"An error occurred during download: {e}")

# Frame extraction part
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

# Load the downloaded video
video_capture = cv2.VideoCapture(download_path)

# Frame extraction settings
frame_rate = 1  # Extract one frame every second
success, frame_count = True, 0

while success:
    # Move to the next frame based on the specified frame rate
    video_capture.set(cv2.CAP_PROP_POS_MSEC, frame_count * 1000 * frame_rate)
    success, frame = video_capture.read()

    if success:
        # Save frame as an image file
        frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Saved: {frame_path}")
        frame_count += 1

video_capture.release()
print("Frame extraction completed.")
