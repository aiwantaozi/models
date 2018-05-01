from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import tensorflow as tf
import json
import pandas as pd
from PIL import Image, ImageDraw

from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the Json input')
flags.DEFINE_string('image_dir', '', 'Path to the image')
flags.DEFINE_string('label', '', 'label name')
FLAGS = flags.FLAGS


def drawrectangle(group, expect_label_name, draw):
    for index, row in group.object.iterrows():
        if row['class'] == expect_label_name:
            data = (row['xmin'], row['ymin'] , row['xmax'], row['ymax'])
            draw.rectangle(data, fill=(0,0,0,128))

def draw_single_image(group, label, filename):
    im01 = Image.open(filename)
    im01 = im01.convert("RGBA")

    tmp = Image.new('RGBA', im01.size, (0,0,0,0))
    draw = ImageDraw.Draw(tmp)

    drawrectangle(group, label, draw)

    im01 = Image.alpha_composite(im01, tmp)
    im01.show()

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def main(_):

    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        filename = group.filename.encode('utf8')
        image_dir = FLAGS.image_dir.encode("utf-8") # python3
        file_path = os.path.join(image_dir, filename)
        draw_single_image(group, FLAGS.label, file_path)
if __name__ == '__main__':
    tf.app.run()
