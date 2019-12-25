import math
import random

import numpy as np
from PIL import Image, ImageFilter
from scipy.optimize import basinhopping
import PIL.ImageOps
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from matplotlib import pyplot as plt

import utils


def contrast(im):
    colors = np.array(im).flatten()
    values = np.percentile(colors, [1, 99])
    value_from, value_to = int(round(values[0])), int(round(values[1]))
    tga = 255 / (value_to - value_from)
    w, h = im.size
    for i in range(w):
        for j in range(h):
            pixel = im.getpixel((i, j))

            pixel = (pixel - value_from) * tga
            pixel = int(round(pixel))
            if pixel < 0:
                pixel = 0
            if pixel > 255:
                pixel = 255
            im.putpixel((i, j), pixel)
    return im


def as_white_background(im):
    colors = np.array(im).flatten()
    median = np.percentile(colors, 50)
    if median > 128:
        return im
    else:
        return PIL.ImageOps.invert(im)


def get_prepared(im):
    return as_white_background(contrast(im))


def get_edges(im):
    return im.filter(ImageFilter.FIND_EDGES)


def get_histogram(im, horiz=False):
    if horiz:
        return np.array(im).mean(axis=0)
    else:
        return np.array(im).mean(axis=1)


def get_pretty_rotated(im):
    im_edges = get_edges(im)

    # we gonna maximize variance
    def to_minimize(angle):
        return -get_histogram(im_edges.rotate(angle))[4:-4].var()

    needed_angle = utils.find_optimum(to_minimize, 0, 10, 0.25) % 360
    return im.rotate(needed_angle, resample=Image.BILINEAR, fillcolor='white')


def get_lines_positions(im, threshold=.002):
    hist = get_histogram(get_edges(im))
    threshold = hist.min() + threshold * (hist.max() - hist.min())

    height = hist.shape[0]
    lines = []

    pos = 0
    while True:
        while pos < height and hist[pos] < threshold: pos += 1
        if pos == height:
            break

        line_start = pos
        while pos < height and hist[pos] >= threshold: pos += 1
        line_end = pos
        lines.append((line_start, line_end))
        if pos == height:
            break

    lines = [i for i in lines if abs(i[0] - i[1]) > 5]  # remove errors
    return lines


def get_lines_from_positions(im, lines_positions):
    w, h = im.size
    lines = []
    for line in lines_positions:
        lines.append(im.crop((0, line[0], w - 1, line[1])))
    return lines


def get_words_positions(line, threshold=.002, space_threshold=.15):
    hist = get_histogram(get_edges(line), horiz=True)
    threshold = hist.min() + threshold * (hist.max() - hist.min())

    h_hist = get_histogram(line)
    letters_height = h_hist / h_hist.mean() / 2
    letters_height = 1 - np.round(letters_height)
    letters_height = letters_height.sum()
    space_threshold = int(round(letters_height * space_threshold))
    width = hist.shape[0]
    words = []

    pos = 0
    while True:
        while pos < width and hist[pos] < threshold: pos += 1
        if pos == width:
            break

        word_start = pos
        while pos < width and hist[pos] >= threshold: pos += 1
        word_end = pos
        new_word = (word_start, word_end)
        if len(words) != 0:
            prev_words = words[-1]
            if word_start - prev_words[1] < space_threshold:
                words[-1] = (words[-1][0], word_end)
            else:
                words.append(new_word)
        else:
            words.append(new_word)

        if pos == width:
            break

    words = [i for i in words if abs(i[0] - i[1]) > 5]  # remove errors
    return words


def get_words_from_position(im, words_positions):
    w, h = im.size
    words = []
    for word in words_positions:
        words.append(im.crop((word[0], 0, word[1], h - 1)))
    return words


def get_letters_bounds(im_word):
    word = np.array(im_word)
    threshold = threshold_otsu(word)
    bw_word = closing(word < threshold, square(1))
    labels = label(bw_word)
    # plt.figure(1)
    # plt.imshow(labels)
    # plt.show()

    letters = []
    for region in regionprops(labels):
        # take regions with large enough areas
        if region.area >= 20:
            # draw rectangle around segmented coins
            _, letter_start, _, letter_end = region.bbox
            letters.append((letter_start, letter_end))

    letters.sort(key=lambda x: x[0])
    processed_letters = []
    for letter in processed_letters:
        if len(processed_letters) != 0:
            prev_letter = processed_letters[-1]
            if letter[0] - prev_letter[1] < -7:
                processed_letters[-1] = (min(letter[0], prev_letter[0]), max(letter[1], prev_letter[1]))
            else:
                processed_letters.append(letter)
        else:
            processed_letters.append(letter)
    return letters


def get_letters_from_bounds(word, letters_bounds):
    w, h = word.size
    letters = []
    for bounds in letters_bounds:
        letters.append(get_unified(word.crop((bounds[0], 0, bounds[1], h - 1))))
    return letters


def trim(im, background=255, thres=.99):
    w, h = im.size
    image = np.array(im)
    left = 0
    while image[left, :].sum() > w * background * thres:
        left += 1
    right = h - 1
    while image[right, :].sum() > w * background * thres:
        right -= 1
    up = 0
    while image[:, up].sum() > h * background * thres:
        up += 1
    down = w - 1
    while image[:, down].sum() > h * background * thres:
        down -= 1
    return im.crop((up, left, down, right))


def get_unified(im):
    return trim(im).resize((32, 32), resample=Image.BILINEAR)


def slope(im, angle):
    w, h = im.size

    left_line = w // 2 - h // 10
    right_line = w // 2 + h // 10
    delta = int(round(math.tan(math.pi * angle / 180) * h))

    start_points = [(left_line, 0), (left_line, h - 1), (right_line, h - 1), (right_line, 0)]
    end_points = [(left_line + delta, 0), (left_line, h - 1), (right_line, h - 1), (right_line + delta, 0)]
    coeffs = find_coeffs(end_points, start_points)

    return im.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor='white')


def find_coeffs(pa, pb):
    """Find coefficients for perspective transformation. From http://stackoverflow.com/a/14178717/4414003."""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def get_pretty_sloped(im):
    im_edges = get_edges(im)

    # we gonna maximize variance
    def to_minimize(angle):
        return -get_histogram(slope(im_edges, angle), horiz=True).var()

    needed_angle = utils.find_optimum(to_minimize, 0, 10, 0.25)
    return slope(im, needed_angle)
