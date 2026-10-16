"""
Main pipeline for football film analysis

Where this fits in the overall flow:
    video file -> OpenCV reads frames -> YOLOv8 finds people (bounding boxes)
    -> FormationAnalyzer turns boxes into per-frame stats
    -> outputs JSON / CSV / annotated MP4

This file wires the whole thing together. Run it from the repo root:
    python src/pipeline.py
It reads the video, collects YOLO boxes per frame, runs formation
analysis, prints a summary, and writes the annotated MP4 + CSV.
"""

import sys
import os
# Add the repo root to Python's import search path so that
# `from src.detection...` works when running `python src/pipeline.py`
# directly (without this, Python wouldn't find the `src` package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import yaml
import numpy as np
from tqdm import tqdm
from src.detection.player_detector import PlayerDetector
from src.analysis.formation_analyzer import FormationAnalyzer


def run_full_pipeline(video_path, config_path="config.yaml", create_annotated_video=True):
    """
    Run the full analysis pipeline: detection + formation analysis

    All file paths (config.yaml, data/...) are relative, so run this from
    the repo root directory.

    Args:
        video_path: Path to the video file
        config_path: Path to the configuration YAML file
        create_annotated_video: Whether to create a video with formation overlays
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
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {width}x{height}, {total_frames} frames")

    # Collect detection results and frames
    # NOTE: every frame is kept in memory in `frames`, so long videos can
    # run out of RAM.
    detection_results = []
    frames = []
    frame_count = 0

    # NOTE: this loop ignores the processing.resize_* and skip_frames
    # settings from config.yaml — unlike player_detector.detect_players,
    # it processes every frame at full size.
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

            # Get boxes: xyxy pixel coordinates for every detection.
            # NOTE: these boxes are NOT filtered to the "person" class —
            # any COCO class YOLO detects (e.g. a sports ball) is passed to
            # the analyzer and counted as a player.
            boxes = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([x1, y1, x2, y2])

            detection_results.append(boxes)
            frames.append(frame)
            frame_count += 1
            pbar.update(1)

    cap.release()

    # Run formation analysis: turns the per-frame box lists into
    # per-frame FormationData rows (and saves them to JSON).
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

    # Create annotated video if requested
    # NOTE: output paths are hardcoded to data/processed/play1_annotated.mp4
    # and formation_analysis.csv regardless of which input video you use.
    if create_annotated_video:
        print("\nCreating annotated video with formation overlays...")
        create_annotated_video_output(frames, formations, detector, config)
        print("Annotated video saved to: data/processed/play1_annotated.mp4")

    # Export to CSV
    print("\nExporting formation data to CSV...")
    export_to_csv(formations, "data/processed/formation_analysis.csv")
    print("CSV saved to: data/processed/formation_analysis.csv")

    return formations


def create_annotated_video_output(frames, formations, detector, config):
    """Create a video with formation information overlaid"""
    # Setup video writer. fourcc is a 4-character codec code ("mp4v" for .mp4).
    output_path = "data/processed/play1_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*config['video']['fourcc'])
    fps = config['video']['fps']

    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Track formation changes: when the formation type flips between
    # frames, the overlay gets highlighted so the change stands out.
    last_formation_type = None

    # Process each frame
    # NOTE: detection runs a SECOND time here (per frame) just to draw the
    # boxes — the first pass above already found them. That makes the
    # pipeline roughly 2x slower than it needs to be.
    for frame_idx, (frame, formation) in enumerate(zip(frames, formations)):
        # Draw field zones
        frame = detector._draw_field_zones(frame, width)

        # Draw player boxes
        # NOTE: this check is always true here — frames and formations
        # have equal length by construction.
        if frame_idx < len(formations):
            # Run detection on this frame
            results = detector.model(
                frame,
                conf=detector.config['model']['confidence_threshold'],
                iou=detector.config['model']['iou_threshold'],
                device=detector.device,
                verbose=False
            )
            frame = detector._draw_detections(frame, results[0])

            # Check for formation change
            formation_changed = (formation.formation_type != last_formation_type)
            if formation_changed:
                last_formation_type = formation.formation_type

            # Draw formation overlay
            overlay_text = {
                'Formation': formation.formation_type.upper(),
                'Balance': formation.balance.upper(),
                'Players': formation.total_players,
                'Frame': frame_idx
            }
            frame = detector._draw_overlay(frame, overlay_text, highlight=formation_changed)

        out.write(frame)

    out.release()


def export_to_csv(formations, output_path):
    """Export formation data to CSV file: one row per frame."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Frame', 'Players', 'Left', 'Center', 'Right',
            'Formation_Width', 'Formation_Type', 'Balance', 'Confidence'
        ])

        for formation in formations:
            writer.writerow([
                formation.frame_number,
                formation.total_players,
                formation.left_side_players,
                formation.center_players,
                formation.right_side_players,
                f"{formation.formation_width:.2f}",
                formation.formation_type,
                formation.balance,
                f"{formation.confidence:.2f}"
            ])


if __name__ == "__main__":
    # `if __name__ == "__main__":` means this block only runs when the file
    # is executed directly (python src/pipeline.py), not when it's imported.
    # Load config to get video path
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    video_path = config['video']['input_path']
    run_full_pipeline(video_path)
