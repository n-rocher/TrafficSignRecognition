import os
import pandas as pd
import matplotlib.pyplot as plt


######################
#    CONFIGURATION   #
######################

CSV_CLASS = "ClassesAEntrainer.csv"

ANALYSING_FOLDER = r"F:\European Traffic Sign Dataset\PNG-Training"

#####################################################

classes = pd.read_csv(CSV_CLASS)

classes["Values"] = [ 0 ] * len(classes)

classes["Title"] = classes["Name"] + classes["Class"].astype(str) 

for index, row in classes.iterrows():
    
    try:
        files = os.listdir(ANALYSING_FOLDER + "/" + str(row[1]).zfill(3))
        classes["Values"][index] = len(files)
    except:
        pass


classes = classes.sort_values("Values", ascending=False)
print(classes)

classes.plot.bar(x='Title', y='Values', rot=90)
plt.show()