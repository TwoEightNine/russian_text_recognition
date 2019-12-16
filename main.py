from PIL import Image, ImageFilter
import processing as prc
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.filters import threshold_otsu
from skimage.color import label2rgb

if __name__ == "__main__":
    im = Image.open('sample.png').convert('L')

    rotated = prc.get_pretty_rotated(im)
    hist = prc.get_histogram(rotated)
    # plt.figure(1)
    # plt.plot(hist)
    # plt.show()

    lines_positions = prc.get_lines_positions(rotated)
    lines = prc.get_lines_from_positions(rotated, lines_positions)
    # plt.figure(1)
    # for i in range(len(lines)):
    #     plt.subplot(len(lines), 1, i + 1)
    #     plt.imshow(np.array(lines[i]), cmap='gray')
    # plt.show()

    line = lines[3]
    hist = prc.get_histogram(line, horiz=True)
    # plt.figure(1)
    # plt.plot(hist)
    # plt.show()

    words_positions = prc.get_words_positions(line)
    words = prc.get_words_from_position(line, words_positions)

    # plt.figure(1)
    # for i in range(len(words)):
    #     plt.subplot(int(len(words) ** .5) + 1, int(len(words) ** .5) + 1, i + 1)
    #     plt.imshow(np.array(words[i]), cmap='gray')
    # plt.show()

    word = words[0]
    hist = prc.get_histogram(word, horiz=True)
    print(hist)
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(hist)
    # plt.subplot(212)
    # plt.imshow(word, cmap='gray')
    # plt.show()

    word = np.array(word)
    threshold = threshold_otsu(word)
    print(threshold)
    bw_word = closing(word < 200, square(1))
    plt.figure(1)
    plt.imshow(bw_word, cmap='gray')
    plt.show()
    labels = label(bw_word)
    image_labels = label2rgb(labels)
    plt.figure(1)
    plt.imshow(image_labels)
    plt.show()
    print(labels)
