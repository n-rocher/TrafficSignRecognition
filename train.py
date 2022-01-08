import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

import wandb
from tensorflow import keras
from wandb.keras import WandbCallback

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

#######################################################
#                   HYPERPARAMETERS                   #
#######################################################
EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
IMG_SIZE = (75, 75)

TRAINING_CLASS = [1, 100, 107, 108, 109, 11, 12, 125, 13, 140, 15, 16, 17, 2, 23, 24, 25, 3, 32, 35, 36, 37, 38, 39, 4, 40, 41, 51, 52, 53, 54, 55, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 7, 80, 81, 82, 83, 84, 85, 86, 87, 88]

FOLDER = r"F:\European Traffic Sign Dataset\PNG-"

#######################################################
#                        MODEL                        #
#######################################################
def TrafficSignClassifier(img_size, classes):
    x_in = layers.Input(img_size + (3,))
    x = layers.Conv2D(8, (3, 3), activation="relu")(x_in)
    # x = layers.Conv2D(8, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(16, (3, 3), activation="relu")(x)
    # x = layers.Conv2D(16, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    # x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    # x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)

    x = layers.Flatten()(x)
    # x = layers.BatchNormalization()(x)

    x = layers.Dense(256, activation="relu")(x)
    # x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(classes, activation="softmax")(x)

    model = models.Model(inputs=[x_in], outputs=[x], name="TrafficSignClassifier")

    return model

#######################################################
#                 LOAD DATA FUNCTION                  #
#######################################################

def load_data(folder, class_ids):
    data_image = []
    data_label = []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    print("[INFO] " + str(len(class_ids)) + " classes à charger dans le dossier " + FOLDER + folder)

    for index, class_id in enumerate(class_ids):

        # On créé le path du dossier de la classe
        class_path = FOLDER + folder + "/" + str(class_id).zfill(3)

        try:
            # On récupère toutes les images
            images = os.listdir(class_path)
            images = images + images
            images = images[:350]

            compteur_image_ajoutee = 0

            for image in images:

                # On créé le path de l'image
                image_path = class_path + "/" + image

                try:
                    # On charge l'image
                    image = cv2.resize(cv2.imread(image_path), IMG_SIZE)

                    if image is None : 
                        print("NONENONENONENONENONENONENONENONENONENONENONENONENONENONENONENONENONENONENONENONENONENONENONENONE")
                        continue

                    image_equalized = np.zeros_like(image)

                    # Adaptative Treshold
                    for i in range(3):
                        image_equalized[:,:,i] = clahe.apply(image[:,:,i])
                                        
                    # On ajoute dans la mémoire
                    data_image.append(image / 255.)
                    data_image.append(image_equalized / 255.)

                    data_label.append(index)  # ON UTILISE L'INDEX POUR LE ONE-HOT ENCODER
                    data_label.append(index)  # ON UTILISE L'INDEX POUR LE ONE-HOT ENCODER

                    compteur_image_ajoutee += 1

                except KeyboardInterrupt:
                    print("[SHUTING DOWN] Fin du programme...")
                    exit()
                except Exception as err:
                    print(err)
                    print("[ERROR] Impossible d'ouvrir l'image suivante : " + image_path)

            print("[INFO] Classe " + str(class_id) + " chargée (" + str(compteur_image_ajoutee) + " images)")

        except Exception as err:
            print(err)
            print("[ERROR] Impossible d'ouvrir le dossier suivant : " + class_path)

    data_image = np.array(data_image, dtype=np.float16)
    data_label = np.array(data_label, dtype=np.uint8)

    return (data_image, data_label)

#######################################################
#                      PROGRAM                        #
#######################################################


if __name__ == '__main__':

    # Weights & Biases
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = wandb.init(project="Traffic Sign recognition", entity="nrocher", config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMG_SIZE,
        "dataset": "European Traffic Sign Dataset",
        "model": "TrafficSignClassifier"
    })

    # On charge le fichier des classes à utiliser
    # class_file = pd.read_csv(CSV_CLASS)
    # class_ids = class_file['Class'].tolist()

    class_ids = TRAINING_CLASS

    (trainX, trainY) = load_data("Training", class_ids)
    (testX, testY) = load_data("Testing", class_ids)


    # One-Hot Encoding data
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    # Image Augmentation
    data_aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        vertical_flip=False)

    # Création du modèle
    model = TrafficSignClassifier(IMG_SIZE, classes=len(class_ids))
    optimizer = Adam(learning_rate=LEARNING_RATE)#, decay=LEARNING_RATE / EPOCHS)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint("models/" + now_str + "/" + model.name + "_" + str(IMG_SIZE[0]) + "-" + str(IMG_SIZE[1]) + "_epoch-{epoch:02d}_loss-{val_loss:.2f}_acc_{val_accuracy:.2f}.h5"),
        keras.callbacks.TensorBoard(log_dir="models/" + now_str + "/logs/", histogram_freq=1),
        WandbCallback()
    ]

    fit = model.fit(
        data_aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(testX, testY),
        # use_multiprocessing=True,
        # workers=6,
        callbacks=callbacks)

    # Weights & Biases - END
    run.finish()
