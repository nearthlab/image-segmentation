# This file contains (modified) parts of codes from the following blog post:
# http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

from PIL import Image
import numpy as np
from skimage import measure
from shapely.geometry import Polygon
import os
from datetime import datetime
from collections import OrderedDict
import json
from coco_config import category_ids, categories
from ast import literal_eval

target_pixels = list(map(literal_eval, list(category_ids.keys())))

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))
            if not isinstance(pixel, int):
                pixel = pixel[:3]
            else:
                pixel = (pixel, pixel, pixel)

            # If the pixel is one of the target pixels...
            if pixel in target_pixels:
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks


def create_sub_mask_annotations(sub_mask, image_id, category_id, annotation_id, is_crowd):
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    annotations = []

    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        if hasattr(poly, 'exterior') and poly.exterior is not None:
            x, y, max_x, max_y = poly.bounds
            width = max_x - x
            height = max_y - y
            bbox = (x, y, width, height)
            area = poly.area
            if width <= 50 and height <= 50:
                continue
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            annotation = OrderedDict(
                [
                    ('segmentation', [segmentation]),
                    ('iscrowd', is_crowd),
                    ('image_id', image_id),
                    ('category_id', category_id),
                    ('id', annotation_id),
                    ('bbox', bbox),
                    ('area', area)
                ]
            )
            annotations.append(annotation)
            annotation_id += 1

            # print(segmentation)
            # clone = Image.new('1', (sub_mask.width, sub_mask.height))
            # draw = ImageDraw.Draw(clone)
            # for i in range(len(segmentation) // 2):
            #     x = segmentation[2 * i]
            #     y = segmentation[2 * i + 1]
            #     draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=255)
            # clone.show('asdf')

    return annotations

def create_header(version='1.0'):
    date = datetime.now()
    header = OrderedDict(
        [
            ("description", "Blade segmentation dataset created by Nearthlab"),
            ("url", "http://nearthlab.com/en"),
            ("version", version),
            ("year", str(date.year)),
            ("contributor", "Nearthlab Inc."),
            ("date_created", date.strftime('%Y-%m-%d %H:%M:%S.%f'))
        ]
    )
    return header

from tqdm import tqdm

def create_cocodata_json(json_filename, image_dir, label_dir, version='1.0'):
    assert(os.path.isdir(image_dir) and os.path.isdir(label_dir))
    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)

    assert(len(image_files) == len(label_files))
    num = len(image_files)
    image_files.sort()
    label_files.sort()

    images = []
    annotations = []
    annotation_id = 1
    for i in tqdm(range(num)):
        image_file = os.path.join(image_dir, image_files[i])
        label_file = os.path.join(label_dir, label_files[i])

        image_name = os.path.basename(image_file)
        label_name = os.path.basename(label_file)

        if image_name[:-4] == label_name[:-4]:
            img = Image.open(image_file)
            lab = Image.open(label_file)
            images.append(OrderedDict(
                [
                    ("license", 1),
                    ("file_name", image_name),
                    ("coco_url", "N/A"),
                    ("height", img.height),
                    ("width", img.width),
                    ("date_captured", "N/A"),
                    ("flickr_url", 'N/A'),
                    ("id", i + 1)
                ]
            ))

            sub_masks = create_sub_masks(lab)
            for color, sub_mask in sub_masks.items():
                category_id = category_ids[color]
                annos = create_sub_mask_annotations(sub_mask, i + 1, category_id, annotation_id, 0)
                annotations += annos
                annotation_id += len(annos)

        else:
            raise RuntimeError('filename mismatch: image_name: {}, label_name: {}'.format(image_name, label_name))

    fp = open(json_filename, 'w')

    json.dump(OrderedDict(
        [
            ('info', create_header(version)),
            ('images', images),
            ('annotations', annotations),
            ('categories', categories)
        ]
    ), fp=fp)

    fp.close()

