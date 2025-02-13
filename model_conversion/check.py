import tensorflow as tf

# Load the YOLOv2 model
model = tf.keras.models.load_model("model_conversion\yolov2-food100.h5")

# Print model input shape
print("Model's actual input shape:", model.input.shape)
