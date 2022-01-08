<!-- <p align="center">

![Demo of BiSeNet V2](image/bisenetv2.gif)

</p> -->

# Traffic Sign Classification
The goal of this project is to create an AI able to classify in real time traffic signs.

# Model
The model is a Convolutional Neural Network of 1M params.

I trained the model using an I7-7700K with a 6GB GTX 1060 and 28GB of RAM.

# Categories
Those are the 53 categories trained to be classified by the AI :

    1: Virage à droite
    100: Sens unique (droit)
    107: Zone 30
    108: Fin zone 30
    109: Passage pour piétons
    11: Ralentisseur simple
    12: Ralentisseur double
    125: Ralentisseur
    13: Route glissante
    140: Direction
    15: Chute de pierres
    16: Passage pour piétons
    17: Enfants (école)
    2: Virage à gauche
    23: Intersection
    24: Intersection avec une route
    25: Rond-point
    3: Double virage (gauche)
    32: Autres dangers
    35: Céder le passage
    36: Stop
    37: Route prioritaire
    38: Fin route prioritaire
    39: Priorité au trafic en sens inverse
    4: Double virage (droite)
    40: Priorité au trafic en sens inverse
    41: Sens interdit
    51: Virage à gauche interdit
    52: Virage à droite interdit
    53: Demi-tour interdit
    54: Dépassement interdit
    55: Dépassement interdit aux véhicules de transport de marchandises
    57: Vitesse maximale 20
    59: Vitesse maximale 30
    60: Vitesse maximale 40
    61: Vitesse maximale 50
    62: Vitesse maximale 60
    63: Vitesse maximale 70
    64: Vitesse maximale 80
    65: Vitesse maximale 90
    66: Vitesse maximale 100
    67: Vitesse maximale 110
    68: Vitesse maximale 120
    7: Rétrécissement de la chaussée
    80: Direction - Tout droit
    81: Direction - Droite
    82: Direction - Gauche
    83: Direction - Tout droit ou à droite
    84: Direction - Tout droit ou à gauche
    85: Direction - Tourner à droite
    86: Direction - Tourner à gauche
    87: Passer à droite
    88: Passer à gauche

# Datasets
The AI was trained using a mix of those two datasets :
1. [European Traffic Sign Dataset](https://ieeexplore.ieee.org/abstract/document/8558481) 
2. [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/)


# Tools
List of tools I used :
1. [Keras](https://keras.io/)
2. [OpenCV](https://opencv.org/)
3. [Weights & Biases](https://wandb.ai/)