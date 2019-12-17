import numpy as np
from PIL import Image, ImageFilter
from scipy.optimize import basinhopping
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from matplotlib import pyplot as plt


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
    return im.rotate(needed_angle, resample=Image.BILINEAR)


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


def get_words_positions(line, threshold=.002, space_threshold=.2):
    hist = get_histogram(get_edges(line), horiz=True)
    threshold = hist.min() + threshold * (hist.max() - hist.min())
    space_threshold = int(round(line.size[1] * space_threshold))
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

    letters = []
    for region in regionprops(labels):
        # take regions with large enough areas
        if region.area >= 30:
            # draw rectangle around segmented coins
            _, letter_start, _, letter_end = region.bbox
            new_letter = (letter_start, letter_end)
            if len(letters) != 0:
                prev_letter = letters[-1]
                if new_letter[0] < prev_letter[1]:
                    letters[-1] = (min(new_letter[0], prev_letter[0]), max(new_letter[1], prev_letter[1]))
                else:
                    letters.append(new_letter)
            else:
                letters.append(new_letter)

    letters.sort(key=lambda x: x[0])
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
