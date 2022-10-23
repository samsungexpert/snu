import tensorflow as tf


lst = tf.io.gfile.glob('*.py')
print(lst)
print(type(lst))