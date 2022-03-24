import numpy as np
import cv2 as cv
import glob
import logging
import argparse
import json
from tqdm import tqdm
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(name)s::%(levelname)-8s: %(message)s')


class VerbosityParsor(argparse.Action):
    """ accept debug, info, ... or theirs corresponding integer value formatted as string."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:  # in case it represent an int, directly get it
            values = int(values)
        except ValueError:  # else ask logging to sort it out
            assert isinstance(values, str)
            values = logging.getLevelName(values.upper())
        setattr(namespace, self.dest, values)


def main():
    parser = argparse.ArgumentParser(description='Apply un-distortion from intrinsic calibration.')
    parser_verbosity = parser.add_mutually_exclusive_group()
    parser_verbosity.add_argument(
        '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO, action=VerbosityParsor,
        help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
    parser_verbosity.add_argument(
        '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
    parser.add_argument('-d', '--display', default=None, const='', nargs='?',
                        help='display target (or output to video if <filename> given).')
    parser.add_argument('-i', '--input', required=True, help='path to input directory')
    parser.add_argument('-c', '--calib', required=True, help='path to calib file')
    parser.add_argument('-o', '--output', default='calib.txt', help='output text file with full calib')

    args = parser.parse_args()

    try:
        # args.video_out_filename = args.display or None
        # args.display = args.display is not None

        if args.verbose:
            logging.getLogger().setLevel(logging.INFO)
        if args.verbose > 1:
            logging.getLogger().setLevel(logging.DEBUG)

        # undistort
        with open(args.calib, 'rt') as f:
            calib_json = json.load(f)
            print(calib_json)
        # mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    except Exception as e:
        logger.critical(e)
        if args.verbose <= logging.DEBUG:
            raise


if __name__ == '__main__':
    main()
