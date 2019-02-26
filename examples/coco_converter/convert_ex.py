from utils import create_cocodata_json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='The directory containing images/ and segmentations/ subdirectories', required=True)
parser.add_argument('-j', '--json_path', help='The result json file name', required=True)
args = parser.parse_args()

image_dir = os.path.join(args.path, 'images')
label_dir = os.path.join(args.path, 'semantic')
assert os.path.exists(image_dir), 'No such directory: {}'.format(image_dir)
assert os.path.exists(label_dir), 'No such directory: {}'.format(label_dir)

create_cocodata_json(args.json_path, image_dir, label_dir)
