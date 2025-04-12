import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytest
from pathlib import Path
from src.models.dmpr import DirectionalPointDetector