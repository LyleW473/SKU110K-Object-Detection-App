import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)

# Add grandparent directory to the path to access "src" module
sys.path.append(grandparent_dir)
os.chdir(grandparent_dir)