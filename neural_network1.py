import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("/tmp/data/", one_hot = True)
classes=10
node_hiddenl1=500
node_hiddenl2=300
node_hiddenl3=500

batch_size=100

x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

def neural_network(input):
    hiddenlayer1= {'weight': tf.Variable(tf.random_normal([784,node_hiddenl1])),
                   'bias': tf.Variable(tf.random_normal([node_hiddenl1]))
    }
    hiddenlayer2= {'weight': tf.Variable(tf.random_normal([node_hiddenl1,node_hiddenl2])),
                   'bias': tf.Variable(tf.random_normal([node_hiddenl2]))
    }
    hiddenlayer3= {'weight': tf.Variable(tf.random_normal([node_hiddenl2,node_hiddenl3])),
                   'bias': tf.Variable(tf.random_normal([node_hiddenl3]))
    }
    outputlayer ={'weight':tf.Variable(tf.random_normal([node_hiddenl3,classes])),
                  'bias':tf.Variable(tf.random_normal([classes]))}

    l1= tf.add(tf.matmul(input,hiddenlayer1['weight']),hiddenlayer1['bias'])
    l1=tf.nn.relu(l1)
    l2= tf.add(tf.matmul(l1,hiddenlayer2['weight']),hiddenlayer2['bias'])
    l2=tf.nn.relu(l2)
    l3= tf.add(tf.matmul(l2,hiddenlayer3['weight']),hiddenlayer3['bias'])
    l3=tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3,outputlayer['weight']),outputlayer['bias'])
    return output

def neural_training(output):
    n_epochs=5
    predict=neural_network(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=y))
    optimizer=tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs+1):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y=mnist.train.next_batch(batch_size)
                _,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss+=c

            print('Epoch',epoch,'completed out of',n_epochs,'loss',epoch_loss)

        correct=tf.equal(tf.argmax(predict,1),tf.argmax(y,1))
        acurracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',acurracy.eval({x:mnist.test.images, y:mnist.test.labels}))

neural_training(x)
print((mnist.train.num_examples))
