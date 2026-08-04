import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# ======================
# SETTINGS
# ======================

IMG_SIZE = (224,224)

BATCH_SIZE = 16

EPOCHS = 60


# ======================
# LOAD DATA
# ======================

train = tf.keras.utils.image_dataset_from_directory(

    "dataset/mri_images",

    validation_split=0.2,

    subset="training",

    seed=123,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE

)

val = tf.keras.utils.image_dataset_from_directory(

    "dataset/mri_images",

    validation_split=0.2,

    subset="validation",

    seed=123,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE

)

CLASS_NAMES = train.class_names

print("\nClasses:", CLASS_NAMES)


# ======================
# DATA PIPELINE
# ======================

AUTOTUNE = tf.data.AUTOTUNE

train = (

train

.cache()

.shuffle(200)

.prefetch(AUTOTUNE)

)

val = (

val

.cache()

.prefetch(AUTOTUNE)

)


# ======================
# AUGMENTATION
# ======================

augmentation = tf.keras.Sequential([

layers.RandomFlip(

"horizontal"

),

layers.RandomRotation(

0.15

),

layers.RandomZoom(

0.15

),

layers.RandomContrast(

0.15

)

])


# ======================
# MODEL
# ======================

model = models.Sequential([

layers.Input(

shape=(224,224,3)

),

augmentation,

layers.Rescaling(

1./255

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

(676,128)

),


layers.LSTM(

128

),


layers.Dropout(

0.5

),


layers.Dense(

128,

activation="relu"

),


layers.Dense(

len(CLASS_NAMES),

activation="softmax"

)

])


# ======================
# COMPILE
# ======================

model.compile(

optimizer="adam",

loss="sparse_categorical_crossentropy",

metrics=["accuracy"]

)


model.summary()


# ======================
# CALLBACKS
# ======================

early = tf.keras.callbacks.EarlyStopping(

monitor="val_loss",

patience=8,

restore_best_weights=True

)

checkpoint = tf.keras.callbacks.ModelCheckpoint(

"mri_model.keras",

save_best_only=True

)


# ======================
# TRAIN
# ======================

history = model.fit(

train,

validation_data=val,

epochs=EPOCHS,

callbacks=[

early,

checkpoint

]

)


# ======================
# SAVE ACCURACY
# ======================

accuracy = round(

max(

history.history[
"val_accuracy"
]

)*100,

2

)


with open(

"accuracy.txt",

"w"

) as f:

    f.write(
        str(
            accuracy
        )
    )


print(
"\nAccuracy Saved:",
accuracy,
"%"
)


# ======================
# SAVE HISTORY
# ======================

pd.DataFrame(

history.history

).to_csv(

"training_history.csv",

index=False

)


# ======================
# TRAIN GRAPH
# ======================

plt.figure(

figsize=(8,5)

)

plt.plot(

history.history[
"accuracy"
]

)

plt.plot(

history.history[
"val_accuracy"
]

)

plt.legend(

[
"Train",
"Validation"
]

)

plt.title(

"Training Accuracy"

)

plt.savefig(

"training_graph.png"

)

plt.close()


# ======================
# LOSS GRAPH
# ======================

plt.figure(

figsize=(8,5)

)

plt.plot(

history.history[
"loss"
]

)

plt.plot(

history.history[
"val_loss"
]

)

plt.legend(

[
"Train",
"Validation"
]

)

plt.title(

"Loss Graph"

)

plt.savefig(

"loss_graph.png"

)

plt.close()


# ======================
# EVALUATION
# ======================

y_true=[]

y_pred=[]


for images, labels in val:

    pred = model.predict(

        images,

        verbose=0

    )

    y_true.extend(

        labels.numpy()

    )

    y_pred.extend(

        np.argmax(
            pred,
            axis=1
        )

    )


report = classification_report(

y_true,

y_pred,

target_names=

CLASS_NAMES

)


with open(

"classification_report.txt",

"w"

) as f:

    f.write(
        report
    )


cm = confusion_matrix(

y_true,

y_pred

)


plt.figure(

figsize=(6,6)

)

plt.imshow(

cm,

cmap="Blues"

)

plt.colorbar()

plt.title(

"Confusion Matrix"

)

plt.savefig(

"confusion_matrix.png"

)

plt.close()


print(
"\nTRAINING COMPLETE"
)