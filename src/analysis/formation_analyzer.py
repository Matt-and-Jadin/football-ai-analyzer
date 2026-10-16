"""
Formation Analysis Module

Where this fits in the overall flow:
    video file -> OpenCV reads frames -> YOLOv8 finds people (bounding boxes)
    -> FormationAnalyzer turns boxes into per-frame stats
    -> outputs JSON / CSV / annotated MP4

This file is the "turns boxes into stats" step. For each frame it takes
the list of bounding boxes YOLO found, works out where each person's
center is, and produces simple stats: how many people are on the
left/center/right of the screen, how wide the group spreads, and whether
the formation looks "tight" or "spread".

Important context: the "zones" are thirds of the SCREEN divided by
pixel x-position (33% / 67% of the frame width), not actual field
markings — the result depends on the camera angle and zoom. Also, every
detected person is counted: both teams, referees, and anyone on the
sidelines. There is no team separation yet.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import json


@dataclass
class FormationData:
    """Data structure for formation analysis results

    @dataclass makes Python auto-generate __init__ (and a few other
    methods) so this class works as a simple data-holder: one
    FormationData object = one row of results for one frame.
    """
    frame_number: int
    total_players: int
    left_side_players: int
    right_side_players: int
    center_players: int
    formation_width: float
    formation_type: str
    balance: str
    confidence: float


class FormationAnalyzer:
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the formation analyzer

        Args:
            config_path: Path to the configuration YAML file
        """
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get formation analysis settings
        self.analysis_config = self.config.get('formation_analysis', {})

    def analyze_formation(self, boxes, frame_number, video_width):
        """
        Analyze formation based on player bounding boxes

        Args:
            boxes: List of bounding boxes (x1, y1, x2, y2)
            frame_number: Current frame number
            video_width: Width of the video frame

        Returns:
            FormationData object with analysis results
        """
        # No detections in this frame: return an "empty" result row.
        if len(boxes) == 0:
            return FormationData(
                frame_number=frame_number,
                total_players=0,
                left_side_players=0,
                right_side_players=0,
                center_players=0,
                formation_width=0.0,
                formation_type="no_players",
                balance="unknown",
                confidence=0.0
            )

        # Calculate player centers: the midpoint of each bounding box.
        # We only use the x-position below, so "where on screen" means
        # horizontal position.
        player_centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            player_centers.append((center_x, center_y))

        player_centers = np.array(player_centers)

        # Divide field into thirds of the screen width (pixels, not
        # real field markings)
        left_threshold = video_width * 0.33
        right_threshold = video_width * 0.67

        # Count players in each zone. np.sum on a boolean array counts
        # how many entries are True, i.e. how many centers fall in the zone.
        left_side = np.sum(player_centers[:, 0] < left_threshold)
        right_side = np.sum(player_centers[:, 0] > right_threshold)
        center = len(player_centers) - left_side - right_side

        # Calculate formation width: horizontal distance between the
        # leftmost and rightmost person centers, in pixels.
        min_x = np.min(player_centers[:, 0])
        max_x = np.max(player_centers[:, 0])
        formation_width = max_x - min_x

        # Determine formation type based on width: if the group spans less
        # than width_threshold (50%) of the frame width it's "tight",
        # otherwise "spread".
        width_threshold = video_width * self.analysis_config.get('width_threshold', 0.5)
        if formation_width < width_threshold:
            formation_type = "tight"
        else:
            formation_type = "spread"

        # Determine balance: "balanced" when left and right counts are
        # within 2 of each other, otherwise which side has more.
        if abs(left_side - right_side) <= 2:
            balance = "balanced"
        elif left_side > right_side:
            balance = "left_heavy"
        else:
            balance = "right_heavy"

        # Calculate confidence based on player count. This is NOT the
        # model's detection confidence — it's just total_players /
        # min_players capped at 1.0, so it only says "did we see at least
        # min_players people". min_players is used only here: the config
        # comment says "minimum players to consider formation valid", but
        # no formation is ever rejected for falling below it.
        total_players = len(player_centers)
        min_players = self.analysis_config.get('min_players', 10)
        confidence = min(1.0, total_players / min_players)

        return FormationData(
            frame_number=frame_number,
            total_players=total_players,
            left_side_players=left_side,
            right_side_players=right_side,
            center_players=center,
            formation_width=formation_width,
            formation_type=formation_type,
            balance=balance,
            confidence=confidence
        )

    def analyze_video(self, detection_results, video_width, output_path=None):
        """
        Analyze formations from detection results

        Args:
            detection_results: List of detection results per frame (each is a list of boxes)
            video_width: Width of the video frame
            output_path: Path to save analysis results (JSON)

        Returns:
            List of FormationData objects
        """
        formations = []
        for frame_idx, boxes in enumerate(detection_results):
            formation = self.analyze_formation(boxes, frame_idx, video_width)
            formations.append(formation)

        # Save results
        if output_path is None:
            output_path = self.analysis_config.get('output_path', 'data/processed/formation_analysis.json')

        self._save_formations(formations, output_path)
        print(f"Formation analysis saved to: {output_path}")

        return formations

    def _save_formations(self, formations, output_path):
        """Save formation analysis to JSON file"""
        # The zone counts/widths above are NumPy integers/floats, and
        # json can't serialize NumPy types — hence the int()/float()
        # wrappers converting everything to plain Python numbers.
        data = [
            {
                'frame_number': int(f.frame_number),
                'total_players': int(f.total_players),
                'left_side_players': int(f.left_side_players),
                'right_side_players': int(f.right_side_players),
                'center_players': int(f.center_players),
                'formation_width': float(f.formation_width),
                'formation_type': f.formation_type,
                'balance': f.balance,
                'confidence': float(f.confidence)
            }
            for f in formations
        ]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_formation_summary(self, formations):
        """
        Get a summary of formation analysis

        Args:
            formations: List of FormationData objects

        Returns:
            Dictionary with summary statistics
        """
        if not formations:
            return {}

        formation_types = [f.formation_type for f in formations]
        balances = [f.balance for f in formations]

        return {
            'total_frames': len(formations),
            'formation_types': {
                'tight': formation_types.count('tight'),
                'spread': formation_types.count('spread')
            },
            'balances': {
                'balanced': balances.count('balanced'),
                'left_heavy': balances.count('left_heavy'),
                'right_heavy': balances.count('right_heavy')
            },
            'avg_players': float(np.mean([f.total_players for f in formations])),
            'avg_formation_width': float(np.mean([f.formation_width for f in formations]))
        }


def main():
    """Main function to run formation analysis"""
    # NOTE: this main() is broken — analyze_video() expects a list of
    # per-frame boxes plus video_width, but here it's called with a path
    # string (and no width), which raises TypeError. Use src/pipeline.py
    # to run the analysis instead.
    analyzer = FormationAnalyzer()

    # Analyze the video
    video_path = analyzer.config['video']['input_path']
    formations = analyzer.analyze_video(video_path)

    # Print summary
    summary = analyzer.get_formation_summary(formations)
    print("\nFormation Analysis Summary:")
    print(f"Total frames analyzed: {summary['total_frames']}")
    print(f"Formation types: {summary['formation_types']}")
    print(f"Balances: {summary['balances']}")
    print(f"Average players per frame: {summary['avg_players']:.1f}")


if __name__ == "__main__":
    main()
