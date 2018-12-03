# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
import cv2

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models.ckpt", "")
tf.app.flags.DEFINE_string("video_file", "a.mp4", "")
tf.app.flags.DEFINE_string("out_video_file", "test.mp4", "")

FLAGS = tf.app.flags.FLAGS

with tf.Session() as sess:

def main(_):
    video_file=cv2.VideoCapture(FLAGS.video_file)
    nub=0
    # Get image's height and width.
    height = 0
    width = 0
    ret,image=video_file.read()
    height = image.shape[0]
    width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))
    while ret:
        nub=nub+1
        
        # with open(image_file, 'rb') as img:
        #     with tf.Session().as_default() as sess:
        #         if image_file.lower().endswith('png'):
        #             image = sess.run(tf.image.decode_png(img.read()))
        #         else:
        #             image = sess.run(tf.image.decode_jpeg(img.read()))
        #         height = image.shape[0]
        #         width = image.shape[1]
        

        with tf.Graph().as_default():
            with tf.Session().as_default() as sess:

                # # Read image data.
                # image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                #     FLAGS.loss_model,
                #     is_training=False)
                # image = reader.get_image(image_file, height, width, image_preprocessing_fn)

                image = tf.image.convert_image_dtype(image,tf.float32)
                # Add batch dimension
                image = tf.expand_dims(image, 0)

                generated = model.net(image, training=False)
                generated = tf.cast(generated, tf.uint8)

                # Remove batch dimension
                generated = tf.squeeze(generated, [0])

                # Restore model variables.
                saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                # Use absolute path
                FLAGS.model_file = os.path.abspath(FLAGS.model_file)
                saver.restore(sess, FLAGS.model_file)

                # Make sure 'generated' directory exists.
                generated_file = 'generated/'+str(nub)+'.jpg'
                if os.path.exists('generated') is False:
                    os.makedirs('generated')

                # Generate and write image data to file.
                with open(generated_file, 'wb') as img:
                    start_time = time.time()
                    img.write(sess.run(tf.image.encode_jpeg(generated)))
                    end_time = time.time()
                    tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                    # tf.logging.info('Done. Please check %s.' % generated_file)
        ret,image=video_file.read()
    video_file.release()

    fps = 24   #视频帧率
    fourcc = cv2.cv.CV_FOURCC('M','J','P','G')  
    videoWriter = cv2.VideoWriter(FLAGS.out_video_file, fourcc, fps, (width,height))   #视频大小
    while (nub<0):
        img12 =cv2.imread(str(nub)+'.jpg')
        videoWriter.write(img12)
    videoWriter.release()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
