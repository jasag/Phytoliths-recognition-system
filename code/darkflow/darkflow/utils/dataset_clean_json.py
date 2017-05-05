"""
parse JSON annotations
"""

import os
import sys
import glob
from skimage import io
import json


def _pp(l):  # pretty printing
    for i in l: print('{}: {}'.format(i, l[i]))


def json_parser(ANN, pick, exclusive=False):
    print('Parsing for {} {}'.format(
        pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(cur_dir)
    annotations = os.listdir('.')
    annotations = glob.glob(str(annotations) + '*.json')
    size = len(annotations)

    for i, file in enumerate(annotations):
        # progress bar
        sys.stdout.write('\r')
        percentage = 1. * (i + 1) / size
        progress = int(percentage * 20)
        bar_arg = [progress * '=', ' ' * (19 - progress), percentage * 100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()

        # actual parsing
        with open(file) as jsonfile:

            image_name = file.split(".")[0] + ".jpg"

            image = io.imread(image_name)

            coords_dict = json.load(jsonfile)[image_name]

            h, w = image.shape[:2]

            all = list()

            # Adding coords
            for k, _ in coords_dict.items():

                for rect_y, rect_y2, rect_x, rect_x2 in coords_dict[k]:
                    current = [k, rect_x, rect_y, rect_x2, rect_y2]
                    all += [current]

                add = [[jpg, [w, h, all]]]
                dumps += add

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]] += 1
                else:
                    stat[current[0]] = 1

    print('\nStatistics:')
    _pp(stat)
    print('Dataset size: {}'.format(len(dumps)))

    os.chdir(cur_dir)
    return dumps
