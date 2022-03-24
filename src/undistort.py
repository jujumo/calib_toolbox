import os

import numpy as np
import cv2
from glob import glob
import logging
import argparse
import json
import os.path as path
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
    try:
        parser = argparse.ArgumentParser(description='Apply un-distortion from intrinsic calibration.')
        parser_verbosity = parser.add_mutually_exclusive_group()
        parser_verbosity.add_argument(
            '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO, action=VerbosityParsor,
            help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
        parser_verbosity.add_argument(
            '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
        parser.add_argument('-d', '--display', default=False, action='store_true',
                            help='display target (or output to video if <filename> given).')
        parser.add_argument('-i', '--input', required=True,
                            help='path to input image directory or video file.')
        parser.add_argument('-c', '--calib', required=True,
                            help='path to calib file.')
        parser.add_argument('-o', '--output',
                            help='optional output directory path.')

        args = parser.parse_args()

        # args.video_out_filename = args.display or None
        # args.display = args.display is not None

        if args.verbose:
            logging.getLogger().setLevel(logging.INFO)
        if args.verbose > 1:
            logging.getLogger().setLevel(logging.DEBUG)

        # load calib
        with open(args.calib, 'rt') as f:
            calib_json = json.load(f)
            distortion = np.array(calib_json['distortion'])
            camera_mat = np.array(calib_json['cameraMatrix'])
            # logger.debug(f'calibration loaded: {path.basename(args.calib)}')

        if args.output:
            os.makedirs(args.output, exist_ok=True)

        image_size = (0, 0)
        mapx, mapy = None, None
        # open images/video
        path_list = sorted(glob(path.join(args.input, '*.*')))
        images = {f: cv2.imread(f) for f in path_list}
        images = {f: i for f, i in images.items() if i is not None}
        for input_filepath, image in tqdm(images.items()):
            image_size_current = (image.shape[1], image.shape[0])
            if image_size_current != image_size:
                image_size = image_size_current
                logger.debug(f'computing distortion maps for {image_size}')
                newcameramtx = None
                mapx, mapy = cv2.initUndistortRectifyMap(camera_mat, distortion, None, newcameramtx, image_size, 5)
                # TODO: write out newcameramtx
            # apply disto
            assert mapx is not None
            image_undist = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

            if args.output:
                output_filepath = path.join(args.output, path.basename(input_filepath))
                if output_filepath == input_filepath: # avoid collision
                    output_filepath = '_undistort'.join(path.splitext(output_filepath))
                logger.debug(f'writing {output_filepath}')
                cv2.imwrite(output_filepath, image_undist)

            if args.display:
                cv2.imshow('image', image_undist)
                k = cv2.waitKey(0)
                if k == 27:  # esc = stop display
                    args.display = False

    except Exception as e:
        logger.critical(e)
        if args.verbose <= logging.DEBUG:
            raise


if __name__ == '__main__':
    main()
