import matplotlib.pyplot as plt
import mne
import sys
import pickle


if __name__ == "__main__":

    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        print("Usage: python3 plot.py <path>")
        sys.exit(1)

    #open pickle file
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, mne.io.BaseRaw):
            # Graficar los datos
            fig = data.plot()
            plt.show()
    else:
        print("File does not contain MNE Raw")
