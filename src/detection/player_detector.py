"""
Player Detection Module
Uses YOLOv8 to detect players in football game film
"""

import cv2
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


class PlayerDetector:
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the player detector with configuration

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.model = self._load_model()
        self.device = self._get_device()

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _load_model(self):
        """Load YOLO model"""
        model_name = self.config['model']['name']
        print(f"Loading YOLO model: {model_name}")
        model = YOLO(model_name)
        return model

    def _get_device(self):
        """Determine the device to use (GPU or CPU)"""
        device = self.config['model']['device']
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            print(f"Using device: {device}")
            if device == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(0)}")
        return device

    def detect_players(self, video_path, output_path=None, overlay_text=None):
        """
        Detect players in a video file

        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional, uses config if not provided)
            overlay_text: Optional text to overlay on each frame (e.g., formation info)

        Returns:
            Path to the output video file
        """
        if output_path is None:
            output_path = self.config['video']['output_path']

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Apply resize if configured
        resize_width = self.config['processing']['resize_width']
        resize_height = self.config['processing']['resize_height']
        if resize_width > 0 and resize_height > 0:
            width = resize_width
            height = resize_height

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config['video']['fourcc'])
        out = cv2.VideoWriter(output_path, fourcc, self.config['video']['fps'], (width, height))

        # Process frames
        frame_count = 0
        skip_frames = self.config['processing']['skip_frames']

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if configured
                if frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    pbar.update(1)
                    continue

                # Resize if configured
                if resize_width > 0 and resize_height > 0:
                    frame = cv2.resize(frame, (resize_width, resize_height))

                # Run detection
                results = self.model(
                    frame,
                    conf=self.config['model']['confidence_threshold'],
                    iou=self.config['model']['iou_threshold'],
                    device=self.device,
                    verbose=False
                )

                # Draw detections
                frame = self._draw_detections(frame, results[0])

                # Draw overlay text if provided
                if overlay_text and frame_count < len(overlay_text):
                    frame = self._draw_overlay(frame, overlay_text[frame_count])

                # Write frame
                out.write(frame)

                frame_count += 1
                pbar.update(1)

        # Cleanup
        cap.release()
        out.release()

        print(f"Output saved to: {output_path}")
        return output_path

    def _draw_detections(self, frame, result):
        """
        Draw bounding boxes and labels on the frame

        Args:
            frame: The image frame
            result: YOLO detection result

        Returns:
            Frame with drawn detections
        """
        if not self.config['detection']['draw_boxes']:
            return frame

        boxes = result.boxes
        box_color = tuple(self.config['detection']['box_color'])
        thickness = self.config['detection']['box_thickness']

        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get confidence
            conf = float(box.conf[0].cpu().numpy())

            # Get class
            cls = int(box.cls[0].cpu().numpy())
            class_name = self.model.names[cls]

            # Only draw if it's a person
            if class_name in self.config['detection']['target_classes']:
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

                # Draw label if configured
                if self.config['detection']['draw_labels']:
                    label = f"{class_name} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                 (x1 + label_size[0], y1), box_color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame

    def _draw_overlay(self, frame, text_info):
        """
        Draw overlay text on the frame

        Args:
            frame: The image frame
            text_info: Dictionary with text to display

        Returns:
            Frame with overlay text
        """
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Draw text
        y_offset = 30
        for key, value in text_info.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (20, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        return frame


def main():
    """Main function to run player detection"""
    detector = PlayerDetector()

    # Update config with your video path
    video_path = detector.config['video']['input_path']

    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        print("Please place your video file in data/raw/ and update config.yaml")
        return

    # Run detection
    detector.detect_players(video_path)


if __name__ == "__main__":
    main()
