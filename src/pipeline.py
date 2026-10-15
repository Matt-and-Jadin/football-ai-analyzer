"""
Main pipeline for football film analysis
Combines player detection and formation analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import yaml
from tqdm import tqdm
from src.detection.player_detector import PlayerDetector
from src.analysis.formation_analyzer import FormationAnalyzer


def run_full_pipeline(video_path, config_path="config.yaml"):
    """
    Run the full analysis pipeline: detection + formation analysis

    Args:
        video_path: Path to the video file
        config_path: Path to the configuration YAML file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize detector
    print("Initializing player detector...")
    detector = PlayerDetector(config_path)

    # Initialize formation analyzer
    print("Initializing formation analyzer...")
    analyzer = FormationAnalyzer(config_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {width}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, {total_frames} frames")

    # Collect detection results
    detection_results = []
    frame_count = 0

    with tqdm(total=total_frames, desc="Running detection") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            results = detector.model(
                frame,
                conf=detector.config['model']['confidence_threshold'],
                iou=detector.config['model']['iou_threshold'],
                device=detector.device,
                verbose=False
            )

            # Get boxes
            boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([x1, y1, x2, y2])

            detection_results.append(boxes)
            frame_count += 1
            pbar.update(1)

    cap.release()

    # Run formation analysis
    print("\nRunning formation analysis...")
    formations = analyzer.analyze_video(detection_results, width)

    # Print summary
    summary = analyzer.get_formation_summary(formations)
    print("\n" + "="*50)
    print("FORMATION ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total frames analyzed: {summary['total_frames']}")
    print(f"\nFormation types:")
    print(f"  - Tight: {summary['formation_types']['tight']} frames")
    print(f"  - Spread: {summary['formation_types']['spread']} frames")
    print(f"\nFormation balance:")
    print(f"  - Balanced: {summary['balances']['balanced']} frames")
    print(f"  - Left-heavy: {summary['balances']['left_heavy']} frames")
    print(f"  - Right-heavy: {summary['balances']['right_heavy']} frames")
    print(f"\nAverage players per frame: {summary['avg_players']:.1f}")
    print(f"Average formation width: {summary['avg_formation_width']:.1f} pixels")
    print("="*50)

    return formations


if __name__ == "__main__":
    # Load config to get video path
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    video_path = config['video']['input_path']
    run_full_pipeline(video_path)
