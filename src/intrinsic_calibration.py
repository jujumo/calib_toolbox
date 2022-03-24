#!/usr/bin/env python3
import argparse
import logging
import numpy as np
import cv2
from glob import glob
import os.path
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(name)s::%(levelname)-8s: %(message)s')

# inspired from
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
font = cv2.FONT_HERSHEY_SIMPLEX


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
        parser = argparse.ArgumentParser(description='Do intrinsic calibration from of images of the chessboard.')
        parser_verbosity = parser.add_mutually_exclusive_group()
        parser_verbosity.add_argument(
            '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO, action=VerbosityParsor,
            help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
        parser_verbosity.add_argument(
            '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
        parser.add_argument('-d', '--display', default=None, const='', nargs='?',
                            help='display target (or output to video if <filename> given).')
        parser.add_argument('-i', '--input', required=True, help='path to input directory')
        parser.add_argument('-o', '--output', default='calib.txt', help='output text file with full calib')
        parser.add_argument('--sfm', default='calib.csv', help='output csv file with intrinsic calib in sfm format.')
        parser.add_argument('-x', '--chess_cols', default=17, type=int,
                            help='Number of corners per row (not squares).')
        parser.add_argument('-y', '--chess_rows', default=28, type=int,
                            help='Number of corners per column (not squares).')
        parser.add_argument('--scale', type=float, const=0.01, nargs='?',
                            help='scale (size) of a square of the chessboard (used for position estimation).')

        args = parser.parse_args()

        args.video_out_filename = args.display or None
        args.display = args.display is not None

        if args.verbose:
            logging.getLogger().setLevel(logging.INFO)
        if args.verbose > 1:
            logging.getLogger().setLevel(logging.DEBUG)

        # retrieve all images from input dir
        logging.info('populating files, may take a while ....')
        path_list = sorted(glob(os.path.join(args.input, '*.*')))
        images = {f: cv2.imread(f) for f in path_list}
        images = {f: i for f, i in images.items() if i is not None}
        chessboard_size = (args.chess_cols, args.chess_rows)

        logging.info('listed images : {}'.format(len(images)))

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        corners_3d = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
        corners_3d[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        # if the chess square size is given, then apply a scale factor to ob
        if args.scale:
            corners_3d *= args.scale

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        if args.video_out_filename:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # just probe one of the input image to get the image size
            images
            video_out = cv2.VideoWriter(args.video_out_filename, fourcc, 20.0, (640, 480))
        else:
            video_out = None

        for filename, image in images.items():
            logging.info(f'processing {filename} ...')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # save image to file, if pattern found
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, cv2.CALIB_CB_FAST_CHECK)
            if not ret:
                logging.warning('chess not found in {}'.format(filename))
            else:
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                objpoints.append(corners_3d)

            # draw
            if args.display:
                if not ret:
                    cv2.putText(image, 'not found', (10, 50), font, 3, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.drawChessboardCorners(image, chessboard_size, corners, ret)

                cv2.imshow('grid', image)
                cv2.waitKey(00)

                if video_out:
                    video_out.write(image)

        if video_out:
            video_out.release()

        logging.info('computing calib, may take a while.')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # Distortioncoefficients = (k1 k2 p1 p2 k3)
        calib = dict(
            ret=ret,
            cameraMatrix=mtx.tolist(),
            distortion=dist.tolist(),
            rotations=[r.flatten().tolist() for r in rvecs],
            translations=[t.flatten().tolist() for t in tvecs]
        )

        # write the full calib file
        if args.output:
            enc = json.dumps(calib, indent=4)
            with open(args.output, 'w') as fout:
                fout.write(enc)

        # write the CSV intrinsic file
        if args.sfm:
            M = np.matrix(calib['cameraMatrix'])
            disto = calib['distortion'][0]
            intrinsics = {
                'fx': M[0, 0],
                'fy': M[1, 1],
                'cx': M[0, 2],
                'cy': M[1, 2],
                'k1': disto[0],
                'k2': disto[1],
                'p1': disto[2],
                'p2': disto[3],
                'k3': disto[4],
            }
            calib_str = ', '.join(f'{k}={v}' for k, v in intrinsics.items())
            logging.debug(calib_str)

        if args.display:
            cv2.destroyAllWindows()

    except Exception as e:
        logger.critical(e)
        if args.verbose <= logging.DEBUG:
            raise


if __name__ == '__main__':
    main()
