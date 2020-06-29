"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils
import cv2

FLAGS = tf.flags.FLAGS

test_data="/data/ydeng1/cycle_gan/MR_test_data/"
output="/data/ydeng1/cycle_gan/MR_test_predict/"

tf.flags.DEFINE_string('model', '', 'model path (.pb)')
tf.flags.DEFINE_string('input', 'input_sample.png', 'input image path (.png)')
#tf.flags.DEFINE_string('output', 'output_sample.png', 'output image path (.png)')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')


def inference(input_image_name,output_path):
  graph = tf.Graph()

  with graph.as_default():
    with tf.gfile.FastGFile(input_image_name, 'rb') as f:
      image_data = f.read()
      input_image = tf.image.decode_jpeg(image_data, channels=3)
      input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
      input_image = utils.convert2float(input_image)
      input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())
    [output_image] = tf.import_graph_def(graph_def,
                          input_map={'input_image': input_image},
                          return_elements=['output_image:0'],
                          name='output')

  with tf.Session(graph=graph) as sess:
    generated = output_image.eval()
    filename= output_path+input_image_name[:-11] 
    
    with open(filename, 'wb') as f:
       f.write(generated)

def main(unused_argv):
    #input image dir
  path="/data/ydeng1/cycle_gan/MR_test_data/"
  outpath="/data/ydeng1/cycle_gan/MR_test_predict/"
  for file_name in os.listdir(path):
    filename= path+file_name      
    print(filename,'%%%%%%%%%%%')
     
    inference(filename,outpath )

if __name__ == '__main__':
  tf.app.run()
