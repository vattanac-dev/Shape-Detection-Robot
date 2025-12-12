import tensorflow as tf

model = tf.keras.models.load_model("models/shape_mobilenet.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("models/shape_mobilenet.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved models/shape_mobilenet.tflite")
