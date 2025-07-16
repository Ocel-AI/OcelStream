
import sys
sys.path.append("deepstream/app") 
from deepstream import DynamicRTSPPipeline
from deepstream import SpotManager

pipeline = DynamicRTSPPipeline(max_sources=15)