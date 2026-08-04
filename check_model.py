from tensorflow.keras.models import load_model

model = load_model("models/mri_model.h5")

print("Input Shape:", model.input_shape)
print("\nModel Summary:\n")
model.summary()