"""
Player Detection Module

Where this fits in the overall flow:
    video file -> OpenCV reads frames -> YOLOv8 finds people (bounding boxes)
    -> FormationAnalyzer turns boxes into per-frame stats
    -> outputs JSON / CSV / annotated MP4

This file is the "YOLOv8 finds people" step. It wraps the YOLO object
detector from the `ultralytics` package, feeds it video frames one at a
time via OpenCV (cv2), and draws bounding boxes plus labels onto an
annotated output video.

YOLO means "You Only Look Once": a detector that finds all objects in an
image in a single pass, which is what makes it fast enough for video.
YOLOv8 comes in sizes yolov8n / s / m / l / x (nano to extra-large);
bigger models are more accurate but slower. `YOLO("yolov8s.pt")`
downloads the pre-trained weights file the first time it runs.

The weights were trained on COCO, a dataset of 80 everyday object
classes ("person" is one of them). The model was NOT trained on
football specifically — it just knows what a generic person looks like.
"""

import cv2  # OpenCV: reads/writes video, draws shapes and text on frames
import yaml  # Parses config.yaml into a Python dict
import torch  # PyTorch: used here only to check if a GPU is available
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm  # Progress bar for the frame-by-frame loop


class PlayerDetector:
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the player detector with configuration

        Loads config.yaml, loads the YOLO model, and picks the compute
        device (GPU if possible, CPU otherwise).

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.model = self._load_model()
        self.device = self._get_device()

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        # The leading underscore on _load_config is a Python convention
        # meaning "internal helper" — callers shouldn't need it directly.
        # (In this project, pipeline.py still calls some _helpers directly.)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _load_model(self):
        """Load YOLO model"""
        model_name = self.config['model']['name']
        print(f"Loading YOLO model: {model_name}")
        # YOLO() downloads the weights file (e.g. yolov8s.pt) automatically
        # the first time it runs, then caches it for later runs.
        model = YOLO(model_name)
        return model

    def _get_device(self):
        """Determine the device to use (GPU or CPU)"""
        # "cuda" is the name for NVIDIA GPUs in PyTorch. If the machine has
        # no NVIDIA GPU (or torch was installed without CUDA support),
        # torch.cuda.is_available() is False and we fall back to the CPU.
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

        Reads the input video frame by frame, runs YOLO on each kept
        frame, draws boxes/labels (and optional overlay text), and writes
        an annotated .mp4.

        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional, uses config if not provided)
            overlay_text: Optional text to overlay on each frame (e.g., formation info)

        Returns:
            Path to the output video file
        """
        if output_path is None:
            output_path = self.config['video']['output_path']

        # Open video: cv2.VideoCapture lets us pull frames out of a video
        # file one at a time instead of loading the whole video at once.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties: resolution, frames-per-second, and how
        # many frames the file contains (for the progress bar).
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Apply resize if configured: smaller frames run through YOLO faster.
        resize_width = self.config['processing']['resize_width']
        resize_height = self.config['processing']['resize_height']
        if resize_width > 0 and resize_height > 0:
            width = resize_width
            height = resize_height

        # Setup video writer. "fourcc" is a 4-character code naming the
        # video codec; "mp4v" produces an .mp4 file.
        fourcc = cv2.VideoWriter_fourcc(*self.config['video']['fourcc'])
        out = cv2.VideoWriter(output_path, fourcc, self.config['video']['fps'], (width, height))

        # Process frames
        frame_count = 0
        skip_frames = self.config['processing']['skip_frames']

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # ret is False once the video is out of frames

                # Skip frames if configured: `frame_count % (skip_frames + 1) != 0`
                # means we process only every (skip_frames + 1)th frame.
                # NOTE: skipped frames are dropped entirely — they are not
                # written to the output — yet the output fps still comes
                # from the config, so with skip_frames > 0 the output video
                # plays shorter/faster than the input.
                if frame_count % (skip_frames + 1) != 0:
                    frame_count += 1
                    pbar.update(1)
                    continue

                # Resize if configured
                if resize_width > 0 and resize_height > 0:
                    frame = cv2.resize(frame, (resize_width, resize_height))

                # Run detection. `conf` discards detections scoring below
                # confidence_threshold. `iou` feeds NMS (non-maximum
                # suppression): when two predicted boxes overlap more than
                # the IoU threshold, only the higher-scoring box is kept,
                # so one player doesn't get boxed twice.
                results = self.model(
                    frame,
                    conf=self.config['model']['confidence_threshold'],
                    iou=self.config['model']['iou_threshold'],
                    device=self.device,
                    verbose=False
                )

                # Draw detections. results is a list with one entry per
                # image passed in; results[0] is the result for our single
                # frame.
                frame = self._draw_detections(frame, results[0])

                # Draw overlay text if provided
                if overlay_text and frame_count < len(overlay_text):
                    frame = self._draw_overlay(frame, overlay_text[frame_count])

                # Write frame
                out.write(frame)

                frame_count += 1
                pbar.update(1)

        # Cleanup: releasing cap/out flushes the video file and frees it.
        cap.release()
        out.release()

        print(f"Output saved to: {output_path}")
        return output_path

    def _draw_detections(self, frame, result):
        """
        Draw bounding boxes and labels on the frame

        Only draws detections whose class is listed in target_classes in
        the config (just "person" here), so e.g. a COCO "sports ball"
        detection is ignored.

        Args:
            frame: The image frame
            result: YOLO detection result (results[0] for a single frame)

        Returns:
            Frame with drawn detections
        """
        if not self.config['detection']['draw_boxes']:
            return frame

        boxes = result.boxes
        # OpenCV colors are BGR, not RGB — so [0, 255, 0] is green and
        # the numbers mean (blue, green, red).
        box_color = tuple(self.config['detection']['box_color'])
        thickness = self.config['detection']['box_thickness']

        for box in boxes:
            # Get box coordinates: box.xyxy[0] is (x1, y1) top-left and
            # (x2, y2) bottom-right, in pixels. .cpu().numpy() moves the
            # PyTorch tensor off the GPU into a plain NumPy array so we can
            # do normal math on it.
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get confidence: a score from 0 to 1 for how sure the model is
            conf = float(box.conf[0].cpu().numpy())

            # Get class: box.cls is the numeric class index;
            # model.names maps that index to a name like "person"
            cls = int(box.cls[0].cpu().numpy())
            class_name = self.model.names[cls]

            # Only draw if it's a person
            if class_name in self.config['detection']['target_classes']:
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

                # Draw label if configured: a filled rectangle behind the
                # text so the label stays readable over the video.
                if self.config['detection']['draw_labels']:
                    label = f"{class_name} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                 (x1 + label_size[0], y1), box_color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame

    def _draw_overlay(self, frame, text_info, highlight=False):
        """
        Draw overlay text on the frame

        Args:
            frame: The image frame
            text_info: Dictionary with text to display
            highlight: Whether to highlight the overlay (for formation changes)

        Returns:
            Frame with overlay text
        """
        # Background for text: OpenCV colors are BGR, so (255, 0, 0) is
        # BLUE, not red.
        overlay = frame.copy()
        bg_color = (255, 0, 0) if highlight else (0, 0, 0)  # Blue if highlight, black otherwise
        cv2.rectangle(overlay, (10, 10), (450, 130), bg_color, -1)
        # addWeighted blends the drawn-on copy back over the original
        # frame: overlay * 0.6 + frame * 0.4 -> a semi-transparent box.
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text: one line per key/value pair, stacked 25px apart
        y_offset = 30
        for key, value in text_info.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (20, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        return frame

    def _draw_field_zones(self, frame, width):
        """
        Draw field zone dividers (left, center, right thirds)

        Args:
            frame: The image frame
            width: Width of the frame

        Returns:
            Frame with zone dividers
        """
        height = frame.shape[0]

        # Draw vertical lines for field thirds
        left_third = int(width * 0.33)
        right_third = int(width * 0.67)

        # Draw lines with transparency: same addWeighted trick as the
        # overlay. (255, 255, 0) is CYAN in BGR.
        overlay = frame.copy()
        cv2.line(overlay, (left_third, 0), (left_third, height), (255, 255, 0), 2)
        cv2.line(overlay, (right_third, 0), (right_third, height), (255, 255, 0), 2)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Add labels
        cv2.putText(frame, "LEFT", (left_third - 50, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, "CENTER", (width // 2 - 40, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, "RIGHT", (right_third + 20, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

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
