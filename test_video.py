#!/usr/bin/env python3
"""
Video file diagnostic tool for debugging .mov and other video file issues.
"""

import cv2
import sys
import os

def test_video_file(video_path):
    """Test if a video file can be opened and read by OpenCV."""
    print(f"Testing video file: {video_path}")
    print("-" * 50)
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"❌ ERROR: File does not exist: {video_path}")
        return False
    
    print(f"✅ File exists: {os.path.getsize(video_path)} bytes")
    
    # Try to open with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ERROR: OpenCV cannot open this video file")
        print("Possible causes:")
        print("  - Unsupported codec")
        print("  - Corrupted file")
        print("  - Missing codec libraries")
        return False
    
    print("✅ OpenCV can open the file")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    print(f"📊 Video Properties:")
    print(f"   Dimensions: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Codec (FourCC): {fourcc} ({chr(fourcc & 0xFF)}{chr((fourcc >> 8) & 0xFF)}{chr((fourcc >> 16) & 0xFF)}{chr((fourcc >> 24) & 0xFF)})")
    
    # Validate properties
    issues = []
    if width <= 0 or height <= 0:
        issues.append(f"Invalid dimensions: {width}x{height}")
    if fps <= 0:
        issues.append(f"Invalid FPS: {fps}")
    if total_frames <= 0:
        issues.append(f"Invalid frame count: {total_frames}")
    
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    
    # Try to read first frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ ERROR: Cannot read first frame")
        cap.release()
        return False
    
    print(f"✅ Successfully read first frame: {frame.shape}")
    
    # Try to read a few more frames
    frames_read = 1
    for i in range(min(10, total_frames - 1)):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames_read += 1
        else:
            break
    
    print(f"✅ Successfully read {frames_read} frames")
    
    cap.release()
    
    if issues:
        print("\n⚠️  Video has metadata issues but may still be processable")
        return False
    else:
        print("\n✅ Video file appears to be fully compatible!")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_video.py <video_file_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    success = test_video_file(video_path)
    sys.exit(0 if success else 1)
