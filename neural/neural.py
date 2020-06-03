import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

PATH = '../train.csv'
TRAIN_SPLIT = 1461


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    print(len(labels))
    return np.array(data), np.array(labels)


def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  plt.savefig(f'{title}.png')
  plt.clf()
  return plt

def baseline(history):
  return np.mean(history)

if __name__ == '__main__':
    df = pd.read_csv(PATH)
    df = df[(df['item'] == 1)]
    df = df.groupby('date')['sales'].sum().to_frame().reset_index()

    # df['date'] = pd.Timestamp(df['date'])

    uni_data = df['sales']
    uni_data.index = df['date']
    uni_data = uni_data.values
    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()

    uni_data = (uni_data - uni_train_mean) / uni_train_std
    univariate_past_history = 20
    univariate_future_target = 0

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)

    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                           univariate_past_history,
                                           univariate_future_target)

    show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
    show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
              'Baseline Prediction Example')

    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])

    simple_lstm_model.compile(optimizer='adam', loss='mae')

    #for x, y in val_univariate.take(1):
    #    print(simple_lstm_model.predict(x).shape)

    EPOCHS = 10
    EVALUATION_INTERVAL = 200

    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, validation_steps=50)

    i = 0
    for x, y in val_univariate.take(3):
        plot = show_plot([x[0].numpy(), y[0].numpy(),
                          simple_lstm_model.predict(x)[0]], 0, f'Simple LSTM model{i}')
        i += 1
        plot.show()

