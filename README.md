# l5kit_demo_project

Demo multi-agent prediction (simulate or use Lyft .zarr).

## Setup
pip install -r requirements.txt

## Run
# 1. Visualize synth scene:
python main.py --mode visualize

# 2. Animate synth scene and save:
python main.py --mode animate --scene-idx 0 --save out.mp4

# 3. Train baseline (on synthetic data):
python main.py --mode train --epochs 10

# 4. Use L5Kit dataset (optional)
python main.py --mode visualize --use-l5kit --l5kit-data /path/to/lyftdata

