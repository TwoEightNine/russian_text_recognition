import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import cnn
import dataset
import processing as prc

if __name__ == "__main__":

    model = cnn.CNN()
    im = Image.open('sample.png').convert('L')
    rotated = prc.get_pretty_rotated(im)

    lines_positions = prc.get_lines_positions(rotated)
    lines = prc.get_lines_from_positions(rotated, lines_positions)

    result_letters = []
    for line in lines:

        words_positions = prc.get_words_positions(line)
        words = prc.get_words_from_position(line, words_positions)

        for word in words:
            letters_bounds = prc.get_letters_bounds(word)
            letters = prc.get_letters_from_bounds(word, letters_bounds)

            for letter in letters:
                guessed_letter, _ = model.predict(letter)
                result_letters.append(guessed_letter)
        result_letters.append(' ')

    print(''.join(result_letters))
