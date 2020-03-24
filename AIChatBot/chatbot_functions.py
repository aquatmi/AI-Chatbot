import wikipediaapi
import tkinter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from keras.models import model_from_json
from tkinter.filedialog import askopenfilename
from scipy import stats


def save(model):
    with open('input/trained_model/model.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights('input/trained_model/model_weights.h5')


def graph(model):
    plt.rcParams['figure.figsize'] = (20.0, 5.0)  # Setting default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['font.family'] = 'Times New Roman'

    fig = plt.figure()
    plt.plot(model.history['accuracy'], '-o', linewidth=3.0)
    plt.plot(model.history['val_accuracy'], '-o', linewidth=3.0)
    plt.title('training', fontsize=22)
    plt.legend(['train', 'validation'], loc='upper left', fontsize='xx-large')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.tick_params(labelsize=18)

    plt.show()

    fig.savefig('training.png')
    plt.close()
# END OF plot()


def get_labels(file):
    label_list = []
    r = pd.read_csv(file)
    for name in r['SignName']:
        label_list.append(name)
    return label_list
# END OF get_labels()


def get_document(page):
    wiki = wikipediaapi.Wikipedia('en')  # set up variable and language of wikipedia
    document = wiki.page(page)
    if document.exists():
        return str(document.summary)
    else:
        return 'null'
# END OF get_document()


def lemmatize(tokens):
    return [WordNetLemmatizer().lemmatize(token) for token in tokens]
# END OF lemmatize()


def get_image():
    win = tkinter.Tk()
    win.withdraw()
    filename = askopenfilename()
    if not (filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg')):
        return None
    else:
        image_array = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        image_array = cv2.resize(image_array, (32, 32), interpolation=cv2.INTER_NEAREST)
        image_array = image_array.reshape(1, 32, 32, 3)
        return image_array
# END OF get_image()


def load_model():
    with open('input/trained_model/model.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights('input/trained_model/model_weights.h5')
    return model
# END OF load_models()


def predict(model, image):
    score = model.predict(image)
    prediction = np.argmax(score)
    # print('ClassId:', prediction)
    labels = get_labels('input/data_set/signnames.csv')
    print('I think that is a', labels[prediction], 'sign.')
    print('Is there anything else you want to know?')
# END OF predict()


