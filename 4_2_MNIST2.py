# 텐서플로우에서 fit 대신 직접 훈련 시키는 방법을 알아봅시다. 
# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch?hl=ko

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# mnist 데이터를 불러와봅시다. 
# 1차원으로 변경하여 mlp에 입력하겠습니다. 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# validation 데이터를 나눕니다. 
# 데이터는 섞여 있어서 shuffle을 안하고 진행
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]


# 모델 작성
model = tf.keras.models.Sequential([
    layers.Flatten(input_shape= (28,28,)),
    layers.Dense(64, activation = 'relu'), 
    layers.Dense(64, activation = 'relu'), 
    layers.Dense(10, activation = 'softmax')
    ])


# 학습 관련 설정
epochs = 2
batch_size = 64
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# epoch 반복: 전체 데이터 1회씩 학습
for epoch in range(epochs):
    print(f'epoch: {epoch}')
    
    # 데이터를 batch_size 만큼씩 학습
    for step in range(0, len(x_train), batch_size):
        
        with tf.GradientTape() as tape:            
            pred = model(x_train[step:step+batch_size], training = True)
            loss = loss_fn(y_train[step:step+batch_size], pred)
            
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # 100 batch 마다 출력
        if step % 100 == 0:
            print(f'{step}: loss: {loss}')

# 평가하기        
pred = model(x_test)
pred_y = tf.math.argmax(pred, axis = 1)
m = tf.keras.metrics.Accuracy() #accuracy 평가
m.update_state(pred_y, y_test)
print(f'accuracy: {m.result().numpy():.1%}')
        
        
    
