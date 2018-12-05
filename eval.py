# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
import cv2
import numpy as np

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models.ckpt", "")
tf.app.flags.DEFINE_string("image_file", "a.jpg", "")
#分割次数
tf.app.flags.DEFINE_string("fen", "1", "")
#偏移量
tf.app.flags.DEFINE_string("d","0","")

FLAGS = tf.app.flags.FLAGS

#图像分割
#res        图像矩阵
#ii         分割次数
#deviation  像素偏移
deviation = int(FLAGS.d)
def fen(res,ii):
    leth = len(res)
    ioio = (int)(leth/ii)
    ret=[]
    if deviation > ioio:
        print("error:偏移量比每段分割大")
        return ret
    head=0
    for i in range(ii):
        tail = head + ioio
        if i == 0:
            ret.append(res[ head : tail + deviation])
        elif i+1 == ii:
            ret.append(res[ head - deviation : tail ])
        else:
            ret.append(res[ head - deviation : tail + deviation ])
        head = head + ioio
    return ret

def run(f,i,ii):
    with tf.Graph().as_default():
            with tf.Session().as_default() as sess:
                # Read image data.
                # image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                #     FLAGS.loss_model,
                #     is_training=False)
                # image = reader.get_image(FLAGS.image_file, height, width, image_preprocessing_fn)
                
                image = f[i]
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
                generated_file = 'generated/t.jpg'
                if os.path.exists('generated') is False:
                    os.makedirs('generated') 
                # Generate and write image data to file.
                with open(generated_file,'wb') as img:
                    start_time = time.time()
                    img.write(sess.run(tf.image.encode_jpeg(generated)))
                    end_time = time.time()

                    readimage = cv2.imread('generated/t.jpg')
                    imageout = cv2.imread('generated/res.jpg')
                    if i == 0:
                        if ii > 1:
                            imageout = readimage[ 0 : len(readimage) - deviation]
                        else:
                            imageout = readimage
                    elif i + 1 ==ii:
                        imageout = np.vstack((imageout,readimage[ deviation : len(readimage)]))
                    else:
                        imageout = np.vstack((imageout,readimage[ deviation :len(readimage)-deviation]))
                    tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
                    cv2.imwrite('generated/res.jpg',imageout)
                    

def main(_):


    # with open(FLAGS.image_file, 'rb') as img:
    #     with tf.Session().as_default() as sess:
    #         if FLAGS.image_file.lower().endswith('png'):
    #             image = sess.run(tf.image.decode_png(img.read()))
    #         else:
    #             image = sess.run(tf.image.decode_jpeg(img.read()))
    #         height = image.shape[0]
    #         width = image.shape[1]

    image = cv2.imread(FLAGS.image_file)

    # Get image's height and width.
    height = len(image)
    width = len(image[0])
    
    tf.logging.info('Image size: %dx%d' % (width, height))
    image_ten=tf.Variable(image,name="n")
    with tf.Session().as_default() as sess:
        sess.run(tf.global_variables_initializer())
        image=sess.run(image_ten)
    f = []
    ii = 0
    if int(FLAGS.fen) > 1:
        f = fen(image,int(FLAGS.fen))
        ii = len(f)
    else:
        f.append(image)
        ii = 1
    
    for i in range(ii):
        v=tf.Variable(f[i],name="image")
        run(f,i,ii)
    tf.logging.info('Done. ')
    


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
