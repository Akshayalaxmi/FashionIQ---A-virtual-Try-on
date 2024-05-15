import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

SOURCE_IMG = "jeans.jpeg"

from segmentation.cloth_extractor import run_visualization


run_visualization(SOURCE_IMG)