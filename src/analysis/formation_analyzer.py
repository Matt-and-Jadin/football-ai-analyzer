"""
Formation Analysis Module
Analyzes player positions to identify football formations
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import json


@dataclass
class FormationData:
    """Data structure for formation analysis results"""
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

        # Calculate player centers
        player_centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            player_centers.append((center_x, center_y))

        player_centers = np.array(player_centers)

        # Divide field into thirds
        left_threshold = video_width * 0.33
        right_threshold = video_width * 0.67

        # Count players in each zone
        left_side = np.sum(player_centers[:, 0] < left_threshold)
        right_side = np.sum(player_centers[:, 0] > right_threshold)
        center = len(player_centers) - left_side - right_side

        # Calculate formation width
        min_x = np.min(player_centers[:, 0])
        max_x = np.max(player_centers[:, 0])
        formation_width = max_x - min_x

        # Determine formation type based on width
        width_threshold = video_width * self.analysis_config.get('width_threshold', 0.5)
        if formation_width < width_threshold:
            formation_type = "tight"
        else:
            formation_type = "spread"

        # Determine balance
        if abs(left_side - right_side) <= 2:
            balance = "balanced"
        elif left_side > right_side:
            balance = "left_heavy"
        else:
            balance = "right_heavy"

        # Calculate confidence based on player count
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
