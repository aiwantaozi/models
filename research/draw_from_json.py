from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import tensorflow as tf
import json
from PIL import Image, ImageDraw

from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('json_input', '', 'Path to the Json input')
flags.DEFINE_string('image_dir', '', 'Path to the image')
flags.DEFINE_string('label', '', 'label name')
FLAGS = flags.FLAGS


def drawrectangle(group, expect_label_name, draw):
    # filename = group["External ID"].encode('utf8')
    # basename = os.path.basename(FLAGS.image_path)
    # if filename == basename:
    for labelname, labelvalue in group["Label"].iteritems():
        if labelname == expect_label_name:
            for row in labelvalue:
                xmin = min(row, key=lambda x: x['x'] )
                xmax = max(row, key=lambda x: x['x'] )
                ymin = min(row, key=lambda x: x['y'] )
                ymax = max(row, key=lambda x: x['y'] )
                # print(xmin['x'], ymin['y'],  xmax['x'], ymax['y'])
                draw.rectangle((xmin['x'], ymin['y'] , xmax['x'], ymax['y']), fill=(0,0,0,128))

def draw_single_image(group, label, filename):
    im01 = Image.open(filename)
    im01 = im01.convert("RGBA")

    tmp = Image.new('RGBA', im01.size, (0,0,0,0))
    draw = ImageDraw.Draw(tmp)

    # with open(FLAGS.json_input) as data_file:    
    #     data = json.load(data_file)
    # for group in data:
    #     filename = group["External ID"].encode('utf8')
    drawrectangle(group, label, draw)

    im01 = Image.alpha_composite(im01, tmp)
    im01.show()

def main(_):

    # for filename in os.listdir(FLAGS.image_dir):
    #     draw_single_image(filename)
    # imgfile = open(FLAGS.image_path, 'wb')
    # im01 = Image.open("/Users/fengcaixiao/Desktop/tempwork/src/github.com/tensorflow/kevinyaoooooo/gridswitchrecognitiononpc/tools/docker/3_dataset/raccoon_dataset/images/321520090639_.pic_hd.jpg")
    # im01 = Image.open(FLAGS.image_path)
    # im01 = im01.convert("RGBA")

    # tmp = Image.new('RGBA', im01.size, (0,0,0,0))
    # draw = ImageDraw.Draw(tmp)

    with open(FLAGS.json_input) as data_file:    
        data = json.load(data_file)
    for group in data:
        filename = group["External ID"].encode('utf8')
        file_path = os.path.join(FLAGS.image_dir, filename)
        draw_single_image(group, FLAGS.label, file_path)

    # im01 = Image.alpha_composite(im01, tmp)
    # im01.show()

if __name__ == '__main__':
    tf.app.run()
