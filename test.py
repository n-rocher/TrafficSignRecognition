import os
import cv2
import numpy as np
from tensorflow import keras

#######################################################
#                      PARAMETRES                     #
#######################################################
MODEL_NAME = r"J:\PROJET\IA\European Traffic Sign Dataset\models\20211212-020838\TrafficSignClassifier_75-75_epoch-19_loss-2.55_acc_0.74.h5"
TEST_FILE = "data/"

if __name__ == "__main__":

    #######################################################
    #                        MODEL                        #
    #######################################################
    model = keras.models.load_model(MODEL_NAME)


    #######################################################
    #                      PROGRAM                        #
    #######################################################

    # On charge les images à tester
    for image_name in os.listdir(TEST_FILE):

        if ".png" in image_name:

            image = cv2.imread(TEST_FILE + image_name)
            image = cv2.resize(image, (75, 75), interpolation = cv2.INTER_AREA)

            image = np.array([image / 255.])

            result = model.predict(image)[0]

            max_index_col = np.argmax(result, axis=0)

            print("\n[INFO] Resultat pour " + image_name + " : '" + max_index_col + "' P=" + str(result[max_index_col]))