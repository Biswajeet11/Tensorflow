import tensorflow as tf
a=tf.constant(6)
b=tf.constant(7)
result=tf.multiply(a,b)

sess=tf.Session()
print(sess.run(result))
