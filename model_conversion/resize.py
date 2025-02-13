import tensorflow as tf

# Load your model (adjust path if needed)
model = tf.keras.models.load_model("model_conversion\yolov2-food100.h5")

# Force fixed input shape
fixed_input = tf.keras.Input(shape=(416, 416, 3))  # Change to your expected YOLO input shape
new_model = tf.keras.Model(inputs=fixed_input, outputs=model(fixed_input))

# Save the corrected model
new_model.save("fixed_model.h5")
