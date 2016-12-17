# -*- coding: utf-8 -*-
import csv
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing

class TrainNN:

  def __init__(self) : 
    # Parameters
    self.learning_rate = 0.01     # 学習率 高いとcostの収束が早まる
    self.training_epochs = 50     # 学習全体をこのエポック数で区切り、区切りごとにcostを表示する

    self.batch_choice = 300
    self.batch_size = 0           # 学習1回ごと( sess.run()ごと )に訓練データをいくつ利用するか
    self.display_step = 1         # 1なら毎エポックごとにcostを表示
    self.train_size = 500         # 全データの中でいくつ訓練データに回すか
    self.step_size =  500         # 何ステップ学習するか

    # Network Parameters
    self.n_hidden_1 = 64      # 隠れ層1のユニットの数
    self.n_hidden_2 = 64      # 隠れ層2のユニットの数

    self.n_input = 27          # 与える変数の数
    self.n_classes = 0        # 分類するクラスの数

  def load_csv(self):
    file_name = "data/jra_race_resultNN.csv"
    df = pd.read_csv(file_name)
    ## 文字列の数値化
    labelEncoder = preprocessing.LabelEncoder()
    df['area_name'] = labelEncoder.fit_transform(df['area_name'])
    df['race_name'] = labelEncoder.fit_transform(df['race_name'])
    df['track'] = labelEncoder.fit_transform(df['track'])
    df['run_direction'] = labelEncoder.fit_transform(df['run_direction'])
    df['track_condition'] = labelEncoder.fit_transform(df['track_condition'])
    df['horse_name'] = labelEncoder.fit_transform(df['horse_name'])
    df['horse_sex'] = labelEncoder.fit_transform(df['horse_sex'])
    df['jockey_name'] = labelEncoder.fit_transform(df['jockey_name'])
    df['margin'] = labelEncoder.fit_transform(df['margin'])
    df['is_blinkers'] = labelEncoder.fit_transform(df['is_blinkers'])
    df['trainer_name'] = labelEncoder.fit_transform(df['trainer_name'])
    df['comments_by_trainer'] = labelEncoder.fit_transform(df['comments_by_trainer'])
    df['evaluation_by_trainer'] = labelEncoder.fit_transform(df['evaluation_by_trainer'])
    df['dhorse_weight'] = labelEncoder.fit_transform(df['dhorse_weight'])
    x_np = np.array(df[['area_name', 'race_number', 'race_name', 'track', 'run_direction',
                       'distance', 'track_condition', 'purse', 'heads_count', 
                       'post_position', 'horse_number', 'horse_name', 'horse_sex', 'horse_age', 
                       'jockey_name', 'time', 'margin', 'time3F', 
                       'load_weight', 'horse_weight', 'dhorse_weight', 'odds_order', 
                       'odds', 'is_blinkers', 'trainer_name', 'comments_by_trainer', 
                        'evaluation_by_trainer'
    ]].fillna(0))
    # 結果
    d = df[['finish_order']].to_dict('record')
    self.vectorizer = DictVectorizer(sparse=False)
    y_np = self.vectorizer.fit_transform(d)
    self.n_classes = len(self.vectorizer.get_feature_names())
    self.train_size =   int(len(df[['finish_order']]) / 5)
    self.batch_size = self.train_size

    # データを訓練データとテストデータに分ける
    [self.x_train, self.x_test] = np.vsplit(x_np, [self.train_size]) 
    [self.y_train, self.y_test] = np.vsplit(y_np, [self.train_size])

  # Create model
  def multilayer_perceptron(self, x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

  def train(self) :
    # tf Graph input
    x = tf.placeholder("float", [None, self.n_input])
    y = tf.placeholder("float", [None, self.n_classes])
    
    # Store layers weight & bias
    weights = {
      'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]), name="h1"),
      'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]), name="h2"),
      'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes]), name="wout")
    }
    # バイアスの設定
    biases = {
      'b1': tf.Variable(tf.random_normal([self.n_hidden_1]), name="b1"),
      'b2': tf.Variable(tf.random_normal([self.n_hidden_2]), name="b2"),
      'out': tf.Variable(tf.random_normal([self.n_classes]), name="bout")
    }

    # Construct model
    pred = self.multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Logistic Regression  AdamOptimizer GradientDescentOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    # Launch the graph
    
    with tf.Session() as sess:
      tf.scalar_summary("cost", cost)
      tf.scalar_summary("accuracy", accuracy)
      merged = tf.merge_all_summaries()
      writer = tf.train.SummaryWriter("logs/tensorflow_log", sess.graph_def)    

      sess.run(init)
      # Training cycle
      for epoch in range(self.training_epochs):
        avg_cost = 0.
        # Loop over step_size
        for i in range(self.step_size):
          # 訓練データから batch_size で指定した数をランダムに取得
          ind = np.random.choice(self.batch_size, self.batch_choice)
          x_train_batch = self.x_train[ind]
          y_train_batch = self.y_train[ind]
          # Run optimization op (backprop) and loss op (to get loss value)
          _, c = sess.run([optimizer, cost], feed_dict={x: x_train_batch, y: y_train_batch})
          avg_cost += c / self.step_size

        # Display logs per epoch step
        if epoch % self.display_step == 0:
          print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
          summary_str, acc = sess.run([merged, accuracy], feed_dict={x: x_train_batch, y: y_train_batch})
          writer.add_summary(summary_str, epoch)
          print "Accuracy:", acc
          ## modelの保存
          # name_model_file = 'model_epoch_' + str(epoch+1) + '.ckpt'
          # save_path = saver.save(sess, 'model/tensorflow/' + name_model_file)
            
      # Test model
      correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
      # Calculate accuracy
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      print "Accuracy:", accuracy.eval(session=sess, feed_dict={x: self.x_test, y: self.y_test})

if __name__ == "__main__":
  trainNN = TrainNN()
  trainNN.load_csv()
  trainNN.train()
