from constants import CLASSES_EMOTIONS

import matplotlib.pyplot as plt
import seaborn as sns


CLASSES_EMOTIONS_DICT = {}
for i in range(len(CLASSES_EMOTIONS)):
    CLASSES_EMOTIONS_DICT[CLASSES_EMOTIONS[i]] = i

def get_emotion_class(label):
    return CLASSES_EMOTIONS_DICT[label]

def get_emotion_label(clas):
    return CLASSES_EMOTIONS[clas]

def show_distribution(series, title = ""):
    sns.set(font_scale=1.2)
    series.sort_values().value_counts(sort = False).plot(kind="bar", figsize=(7, 6), rot=0, title = title, xlabel = 'Emotions', ylabel = 'Count')
    plt.show()
