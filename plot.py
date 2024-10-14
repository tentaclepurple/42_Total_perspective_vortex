import matplotlib.pyplot as plt
import mne
import sys
import pickle


path = 'data/raw_data.pickle'


if __name__ == "__main__":

    #open pickle file
    with open(path, 'rb') as f:
        data = pickle.load(f)


    if isinstance(data, mne.io.BaseRaw):
            # Graficar los datos
            fig = data.plot()
            plt.show()
    else:
        print("El archivo no contiene datos MNE Raw.")