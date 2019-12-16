import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import processing as prc

if __name__ == "__main__":
    im = Image.open('sample.png').convert('L')

    rotated = prc.get_pretty_rotated(im)

    lines_positions = prc.get_lines_positions(rotated)
    lines = prc.get_lines_from_positions(rotated, lines_positions)
    plt.figure(1)
    for i in range(len(lines)):
        plt.subplot(len(lines), 1, i + 1)
        plt.imshow(np.array(lines[i]), cmap='gray')
    plt.show()

    for line in lines:

        words_positions = prc.get_words_positions(line)
        words = prc.get_words_from_position(line, words_positions)

        plt.figure(1)
        for i in range(len(words)):
            plt.subplot(int(len(words) ** .5) + 1, int(len(words) ** .5) + 1, i + 1)
            plt.imshow(np.array(words[i]), cmap='gray')
        plt.show()

        for word in words:
            letters_bounds = prc.get_letters_bounds(word)
            letters = prc.get_letters_from_bounds(word, letters_bounds)

            plt.figure(1)
            for i in range(len(letters)):
                plt.subplot(int(len(letters) ** .5) + 1, int(len(letters) ** .5) + 1, i + 1)
                plt.imshow(np.array(letters[i]), cmap='gray')
            plt.show()
