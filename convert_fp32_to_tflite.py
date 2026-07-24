import tensorflow as tf


print("Loading model...")


model = tf.keras.models.load_model(

    "training/models/model_fp32.keras"

)


print("Converting to TFLite...")


converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()


output_file = "training/models/model_fp32.tflite"


with open(output_file, "wb") as f:

    f.write(tflite_model)


print(f"Created: {output_file}")