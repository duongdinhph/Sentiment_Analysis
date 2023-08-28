# Sentiment_Analysis
Implement a Deep Neural Network based on Bidirectional LSTM Structure for Social media Comments Sentiment Classification
## The problem
Social media comments sentiment analysis helps you discover hidden gems of business intelligence. It helps you set competitor benchmarks, 
track your performance based on customer satisfaction, get product insights, and manage your brand reputation. This project solves this problem 
by implementing a deep neural network model to classify Social media comments sentiments.
## Data
The dataset used is the set of reviews on "Foody.vn" containing 27,000 samples labeled.
It can be downloaded at:
  ```sh
  https://streetcodevn.com/blog/dataset
  ```
## Pre-Processing Data
* Firstly, we do Word Embedding, using Pretrained-Embedding Set:
  ```sh
  https://github.com/sonvx/word2vecVN
  ```
* Randomly split Dataset into Data train, validation and test with the ratio 8:1:1

## Model
  ```sh
  model = Sequential()
  
  model.add(Bidirectional(LSTM(128, return_sequences=True),
                          input_shape=(200, 400)))
  model.add(Dropout(0.3))
  model.add(Bidirectional(LSTM(128, return_sequences=False)))                     
  model.add(Dropout(0.3))
  
  model.add(Dense(units=128, activation='relu'))
  model.add(Dense(units=1, activation='sigmoid'))
  
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.summary()
  ```
* Bidirectional LSTM
  ![image](https://github.com/duongdinhph/Sentiment_Analysis/assets/56771011/0da99c7b-793c-495b-aab0-7cab9981d221)


![image](https://github.com/duongdinhph/Sentiment_Analysis/assets/56771011/0020b39f-f354-438c-b5ed-e5f3b5f76c81)
## Result
  ```sh
  Epoch1/10: 21600/21600 [==============================] - 255s 12ms/step - loss: 0.5280 - acc: 0.7462 - val_loss: 0.4350 - val_acc: 0.8259 
  Epoch 2/10 21600/21600 [==============================] - 250s 12ms/step - loss: 0.3597 - acc: 0.8501 - val_loss: 0.3235 - val_acc: 0.8596 
  Epoch 3/10 21600/21600 [==============================] - 249s 12ms/step - loss: 0.3000 - acc: 0.8771 - val_loss: 0.3156 - val_acc: 0.8652 
  Epoch 4/10 21600/21600 [==============================] - 248s 12ms/step - loss: 0.2627 - acc: 0.8945 - val_loss: 0.2718 - val_acc: 0.8859 
  Epoch 5/10 21600/21600 [==============================] - 243s 11ms/step - loss: 0.2340 - acc: 0.9069 - val_loss: 0.2769 - val_acc: 0.8811 
  Epoch 6/10 21600/21600 [==============================] - 246s 11ms/step - loss: 0.2122 - acc: 0.9175 - val_loss: 0.2846 - val_acc: 0.8896 
  Epoch 7/10 21600/21600 [==============================] - 247s 11ms/step - loss: 0.1802 - acc: 0.9313 - val_loss: 0.3045 - val_acc: 0.8852 
  Epoch 8/10 21600/21600 [==============================] - 247s 11ms/step - loss: 0.1476 - acc: 0.9455 - val_loss: 0.3312 - val_acc: 0.8793 
  Epoch 9/10 21600/21600 [==============================] - 247s 11ms/step - loss: 0.1293 - acc: 0.9532 - val_loss: 0.2995 - val_acc: 0.8922 
  Epoch 10/10 21600/21600 [==============================] - 246s 11ms/step - loss: 0.1045 - acc: 0.9619 - val_loss: 0.3487 - val_acc: 0.8867
  
  scores = model.evaluate(_test_x, test_y, verbose=1)
  print('Test loss:', scores[0])
  print('Test accuracy:', scores[1])
  =>
  2700/2700 [==============================] - 46s 17ms/step Test loss: 0.3845756830744169 Test accuracy: 0.8733333333333333
  ```
