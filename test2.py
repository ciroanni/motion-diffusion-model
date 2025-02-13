from Motion.InverseKinematics import animation_from_positions
from Motion import BVH
import numpy as np
import argparse
from visualize.motions2hik import motions2hik


# get input from the user
# Set up argument parser
parser = argparse.ArgumentParser(description='Process motion data from npy file.')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input npy file')
parser.add_argument('--output_dir', type=str, required=False, help='Path to the output bvh files')
args = parser.parse_args()

# Load data from the provided file path
npy_file = args.input_file

#npy_file = 'results.npy'
data = np.load(npy_file, allow_pickle=True)
# Extract the motion component
motion_component = data.item().get('motion')

animation = motions2hik(motion_component)
BVH.save(animation, args.output_dir)