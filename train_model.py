import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt
import pandas as pd


# ==========================
# LOAD DATA
# ==========================

train = tf.keras.utils.image_dataset_from_directory(

    "dataset/mri_images",

    validation_split=0.2,

    subset="training",

    seed=123,

    image_size=(128,128),

    batch_size=8

)


val = tf.keras.utils.image_dataset_from_directory(

    "dataset/mri_images",

    validation_split=0.2,

    subset="validation",

    seed=123,

    image_size=(128,128),

    batch_size=8

)


print(

"Classes:",

train.class_names

)


# ==========================
# CNN + LSTM
# ==========================

model = models.Sequential([

layers.Rescaling(

1./255,

input_shape=(128,128,3)

),

layers.Conv2D(

32,

3,

activation="relu"

),

layers.MaxPooling2D(),


layers.Conv2D(

64,

3,

activation="relu"

),

layers.MaxPooling2D(),


layers.Conv2D(

128,

3,

activation="relu"

),

layers.MaxPooling2D(),


layers.Reshape(

(196,128)

),


layers.LSTM(

64

),


layers.Dropout(

0.3

),


layers.Dense(

64,

activation="relu"

),


layers.Dense(

2,

activation="softmax"

)

])


model.compile(

optimizer="adam",

loss="sparse_categorical_crossentropy",

metrics=[

"accuracy"

]

)


model.summary()


# ==========================
# TRAIN
# ==========================

history = model.fit(

train,

validation_data=val,

epochs=10

)


# ==========================
# SAVE MODEL
# ==========================

model.save(

"mri_model.keras"

)


print(

"\nMRI CNN-LSTM MODEL SAVED"

)


# ==========================
# SAVE ACCURACY
# ==========================

final_accuracy = round(

history.history[

"val_accuracy"

][-1]*100,

2

)


with open(

"accuracy.txt",

"w"

) as f:

    f.write(

        str(

            final_accuracy

        )

    )


print(

"Accuracy Saved:",

final_accuracy

)


# ==========================
# SAVE CSV
# ==========================

pd.DataFrame(

history.history

).to_csv(

"training_history.csv",

index=False

)


# ==========================
# SAVE GRAPH
# ==========================

plt.figure(

figsize=(8,5)

)

plt.plot(

history.history["accuracy"]

)

plt.plot(

history.history["val_accuracy"]

)

plt.title(

"Training Accuracy"

)

plt.xlabel(

"Epoch"

)

plt.ylabel(

"Accuracy"

)

plt.legend(

[

"Train",

"Validation"

]

)

plt.savefig(

"training_graph.png"

)


print(

"Graph Saved"

)