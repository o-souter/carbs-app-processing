import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model("model_conversion\yolov2-food100.h5")

# Resize model input to (416, 416, 3) for consistency
input_shape = [1, 416, 416, 3]  # Set a standard input size
model.build(input_shape)  # Rebuild the model with new input shape

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
tflite_filename = "yolov2-food100.tflite"
with open(tflite_filename, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved as '{tflite_filename}'")

# === 🔍 Step 2: Verify the Converted Model ===
try:
    interpreter = tf.lite.Interpreter(model_path=tflite_filename)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"📥 Model Input Shape: {input_details[0]['shape']}")
    print(f"📤 Model Output Shape: {output_details[0]['shape']}")

    # Compare with expected shape
    if tuple(input_details[0]['shape']) == tuple(input_shape):
        print("✅ TFLite model has the expected input shape!")
    else:
        print("⚠️ Warning: TFLite model input shape is different than expected!")

except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")