import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

# Download and process the data
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir=",", cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

# Remove unnecessary files in train
#dir_path = '/tmp/.keras/aclImdb/train'
#items = os.listdir(dir_path)
#for item in items:
#  item_path = os.path.join(dir_path, item)
#  if item not in ['neg', 'pos']:
#    if os.path.isfile(item_path):
#      os.remove(item_path)
#    elif os.path.isdir(item_path):
#      shutil.rmtree(item_path)

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

for file in items:
  print(file)

# Split validation data set
batch_size = 32
seed = 42
raw_train_ds = tf.keras.utils.text_dataset_from_directory('/tmp/.keras/aclImdb/train',
                              batch_size=batch_size,
                              validation_split=0.2,
                              subset='training',
                              seed=seed)
#for test_batch, label_batch in raw_train_ds.take(1):
#  for i in range(3):
#    print("Review", test_batch.numpy()[i])
#    print("Label", label_batch.numpy()[i])
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

raw_val_ds = tf.keras.utils.text_dataset_from_directory('/tmp/.keras/aclImdb/train',
                             batch_size=batch_size,
                             validation_split=0.2,
                             subset='validation',
                             seed=seed)
raw_test_ds = tf.keras.utils.text_dataset_from_directory('/tmp/.keras/aclImdb/train',
                              batch_size=batch_size)

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


max_features = 10000
sequence_length = 250
vectorize_layer = layers.TextVectorization(standardize=custom_standardization,
                       max_tokens=max_features,
                       output_mode='int',
                       output_sequence_length=sequence_length)
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16
model = tf.keras.Sequential([
  layers.Embedding(max_features, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1, activation='sigmoid')])

model.summary()

model.compile(loss=losses.BinaryCrossentropy(),
              optimizer='adam',
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)])

epochs = 20
history = model.fit(train_ds,
           validation_data=val_ds,
           epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Draw the model graph
history_dict = history.history
history_dict.keys()
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

examples = tf.constant([
  "The movie was excellent, great, fantastic! I would watch it with my girlfriend again",
  "The movie was okay.",
  "worst"
])

export_model.predict(examples)
