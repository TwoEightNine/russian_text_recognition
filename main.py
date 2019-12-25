import time

import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from skimage.exposure import histogram

import cnn
import dataset
import processing as prc


def create_dataset(image_name):
    im = Image.open(image_name).convert('L')
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
            result_letters += letters

    plt.figure(1)
    sps = int(len(result_letters) ** .5) + 1
    for i in range(len(result_letters)):
        plt.subplot(sps, sps, i + 1)
        plt.imshow(result_letters[i], cmap='gray')
    plt.show()

    ds = dataset.Dataset()
    real_letters = input('>> ')
    for i in range(len(real_letters)):
        ds.add_letter(result_letters[i], real_letters[i])


def recognize_text(image_name):
    model = cnn.CNN()
    im = Image.open(image_name).convert('L')
    im = prc.get_prepared(im)

    rotated = prc.get_pretty_rotated(im)

    lines_positions = prc.get_lines_positions(rotated)
    lines = prc.get_lines_from_positions(rotated, lines_positions)

    result_letters = []
    for line in lines:

        pretty_line = prc.get_pretty_sloped(line)
        # plt.figure(1)
        # plt.subplot(211)
        # plt.imshow(line, cmap='gray')
        # plt.subplot(212)
        # plt.imshow(pretty_line, cmap='gray')
        # plt.show()
        words_positions = prc.get_words_positions(pretty_line)
        words = prc.get_words_from_position(pretty_line, words_positions)

        for word in words:
            letters_bounds = prc.get_letters_bounds(word)
            letters = prc.get_letters_from_bounds(word, letters_bounds)

            # plt.figure(1)
            # i = 1
            # l = int(len(letters) ** .5) + 1
            # for letter in letters:
            #     plt.subplot(l, l, i)
            #     i += 1
            #     plt.imshow(letter)
            # plt.show()

            for letter in letters:
                guessed_letter, prob = model.predict(letter, return_probs=True)
                result_letters.append(guessed_letter)
                # plt.figure(1, figsize=(12, 6))
                # plt.subplot(211)
                # plt.imshow(letter, cmap='gray')
                # plt.subplot(212)
                # for letter, prob in prob.items():
                #     if prob > .005:
                #         plt.bar(letter, prob)
                # plt.show()
            result_letters.append(' ')
        result_letters.append('\n')

    text = ''.join(result_letters)
    text = text.replace('ь|', 'ы')
    text = text.replace('ъ|', 'ы')
    return text


def retrain():
    model = cnn.CNN()
    ds = dataset.Dataset()
    images, letters = ds.get_data()
    model.train_and_save(images, letters)


if __name__ == "__main__":
    for file in ['sample.png']:
        print('------------------', file, '---------------------')
        start_time = time.time()
        print(recognize_text(file))
        print('\n\ntook = %.3f sec' % (time.time() - start_time))
