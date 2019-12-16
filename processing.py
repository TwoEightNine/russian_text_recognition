from PIL import Image, ImageFilter
import numpy as np
from scipy.optimize import basinhopping
from skimage.filters import threshold_otsu


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
    def to_minimize(angle_arr):
        return -get_histogram(im_edges.rotate(angle_arr[0]))[4:-4].var()

    result = basinhopping(to_minimize, [0], stepsize=3)
    needed_angle = int(result.x[0]) % 360
    return im.rotate(needed_angle, fillcolor='white')


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


def get_words_positions(line, threshold=.002):
    hist = get_histogram(get_edges(line), horiz=True)
    threshold = hist.min() + threshold * (hist.max() - hist.min())
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
        words.append((word_start, word_end))
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
