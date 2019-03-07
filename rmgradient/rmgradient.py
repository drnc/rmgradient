#!/usr/bin/env python
# rmgradient.py

# Copyright 2019 Guillaume Duranceau
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Substract a background gradient from a TIFF file image

This can be used to remove light pollution gradient from astrophotography
images for example.

For command line usage, run:
    python rmgradient.py --help

:Author:
    Guillaume Duranceau
"""

import argparse
import logging
import math
import numpy
import os.path
import scipy.interpolate
import scipy.ndimage.filters
import tifffile

logger = logging.getLogger('rmgradient')

class TiffReader:
    """Read image data from a TIFF file."""

    def __init__(self, filename):
        """Create the file reader object.

        Parameters
        ----------
        filename : str
            Path of the file
        """
        self._filename = filename
        self._data = None
        self._tags = []

    def name(self):
        """Return the reader's file name"""
        return self._filename

    def is_valid(self):
        """Check that this is a valid and readable file"""
        if not os.path.isfile(self._filename):
            logger.error("invalid file [{}]".format(self._filename))
            return False

        if not os.access(self._filename, os.R_OK):
            logger.error(
                "cannot read file [{}] (no permission)".format(self._filename))
            return False

        return True

    def load(self):
        """Load file image from the TIFF file as numpy array.

        Returns
        -------
        data : numpy.ndarray
            Image data content
        """
        logger.info("loading image [{}]".format(self._filename));
        try:
            with tifffile.TiffFile(self._filename) as tif:
                self._data = tif.asarray(out='memmap')
                self._extract_tags(tif.pages[0].tags)
            #self._data = tifffile.imread(self._filename, out='memmap')
        except (OSError, IOError, ValueError) as err:
            logger.error(
                "couldn't load image [{}]: {}".format(self._filename, err))
            return None

        # make sure that a monochrome image has 3 dimension as this is assumed
        # everywhere in this file
        if self._data.ndim == 2:
            self._data = numpy.expand_dims(self._data, axis=2)
        return self._data

    def is_2d(self):
        """Check if this is a 2 dimensions image."""
        if self._data is None:
            if self.load() is None:
                return False

        if self._data.ndim != 3:
            logger.error("image [{}] is not a 2D image".format(self._filename))
            return False
        return True

    def points_in_middle(self, points, offset):
        """Check if a list of points are in the middle of the image.

        The points must be separated from the image boundaries by a minimum
        offset.

        Parameters
        ----------
        points : sequence
            List of point coordinates as [x, y] tuples
        offset : int
            Minimum offset from the image boundaries
        """
        if not self.is_2d():
            return False

        xmin = offset
        xmax = self._data.shape[1] - 1 - offset
        ymin = offset
        ymax = self._data.shape[0] - 1 - offset

        for point in points:
            if point[0] >= 0 and point[1] >= 0 and \
               (point[0] < xmin or point[0] > xmax or \
                point[1] < ymin or point[1] > ymax):
                logger.error(
                    "point {} is too close to image boundaries".format(point))
                return False

        return True

    def tags(self):
        """Return interesting tags from the input image."""
        return self._tags

    # Extract the following tags from the image, if present:
    # - DateTime
    # - InterColorProfile
    # - Orientation
    def _extract_tags(self, tags):
        for tagname in ['DateTime', 'InterColorProfile', 'Orientation']:
            if tagname in tags:
                tag = tags[tagname]
                if tag.dtype.startswith('1'):
                    tag.dtype = tag.dtype[1:]
                self._tags.append(
                    (tag.code, tag.dtype, tag.count, tag.value, True))

class TiffWriter:
    """Write image data in a TIFF file."""

    def __init__(self, filename):
        """Create the file writer object.

        Parameters
        ----------
        filename : str
            Path of the file
        """
        self._filename = filename

    def is_valid(self):
        """Check that the file doesn't already exist and can be created"""
        if os.path.exists(self._filename):
            logger.error("file [{}] already exists".format(self._filename))
            return False

        try:
            out = open(self._filename, 'a')
        except Exception as err:
            logger.error(
                "cannot open file [{}]: {}".format(self._filename, err))
            return False
        out.close()
        os.remove(self._filename)

        return True

    def write(self, data, datatype = None, compress = 0, tags = []):
        """Write image data in the TIFF file.

        Parameters
        ----------
        data : numpy.ndarray
            Image data,
        datatype : numpy.dtype
            TIFF data type. None to use data datatype.
        compress : int
            Zlib compression level for TIFF data written.
        tags : sequence of tuple
            Extra tags to write in image, as defined in tifffile.TiffWriter.save
        """
        logger.info("writing image [{}]".format(self._filename));
        if datatype is not None:
            data = data.astype(datatype)
        try:
            tifffile.imwrite(
                self._filename, data, compress=compress, extratags=tags)
        except ValueError as err:
            logger.error(
                "couldn't write image [{}]: {}".format(self._filename, err))
            return False
        return True

class BlurImage:
    """Blur selected points of an image"""

    def __init__(self, image, sigma):
        """Create the blurring image object.

        Parameters
        ----------
        image : numpy.ndarray
            Image data.
        sigma : float
            Standard deviation of the Gaussian distribution used to blur image
            data
        """
        self._image = image

        # generate the convolution kernel to apply to selected image points to
        # blur them. this is simply a Gaussian filter of a 2D dirac function.
        self._center_offset = math.ceil(3 * sigma)
        kernel_size = 2 * self._center_offset + 1
        dirac = numpy.zeros((kernel_size, kernel_size))
        dirac[self._center_offset, self._center_offset] = 1
        self._kernel = scipy.ndimage.filters.gaussian_filter(dirac, sigma)

        # only compute the blur on the 3 first channels
        self._channels = min(self._image.shape[2], 3)

    def run(self, point):
        """Apply a Gaussion blur on a point value in the image.

        Parameters
        ----------
        point : sequence
            List of 2 integrate coordinates on the image [x, y].
            The point is assumed to not be too close to the image boundaries.
            More precisely, it should be at a distance of at least 3 times the
            sigma value from any boundary.

        Returns
        -------
        value : sequence
            Channel values of the blurred image at the input point coordinates
        """
        # we just need to compute the convolution in a single point.
        # we extract a square from the image around the point with the same
        # size than the kernel.
        xmin = point[0] - self._center_offset
        xmax = point[0] + self._center_offset
        ymin = point[1] - self._center_offset
        ymax = point[1] + self._center_offset
        square = self._image[ymin:ymax+1, xmin:xmax+1]

        value = []
        for i in range(0, self._channels):
            value.append(numpy.sum(square[...,i] * self._kernel))
        return value

class BackgroundModel:
    """Generate a model of an image background gradient"""

    def __init__(self, image, sigma, smooth = 0.1, rows = 500):
        """Create the background model object.

        Parameters
        ----------
        image : numpy.ndarray
            Image data.
        sigma : float
            Standard deviation of the Gaussian distribution used to blur image
            data prior to generating background model
        smooth : float
            Smoothness of the approximation, as used by scipy.interpolate.Rbf
        rows : int
            Compute the result of the background model on an image grid by packs
            of rows. Reducing this number reduces memory consumption
        """
        self._image = image
        self._smooth = smooth
        self._rows = rows
        self._blur = BlurImage(self._image, sigma)

    def run(self, bgpoints):
        """Generate the background model.

        Parameters
        ----------
        bgpoints : sequence
            List of background points ([x, y] coordinates) in the image from
            which the model is generated. Those points are assumed to not be too
            close to the image boundaries. More precisely, they should be at a
            distance  sigma value from any boundary.
            If a point has negative coordinates, its absolute positive
            coordinates are used instead, but its color is assumed to the be
            same than the previous point in the list.

        Returns
        -------
        data : numpy.ndarray
            Blurred image data
        """
        xvalues = abs(numpy.array(bgpoints)[...,0]).tolist()
        yvalues = abs(numpy.array(bgpoints)[...,1]).tolist()

        logger.info("background model generation: "\
            "blurring selected image points")
        blur_values = []
        for point in bgpoints:
            if point[0] >= 0 and point[1] >= 0:
                blur_values.append(self._blur.run(point))
            else:
                # point with negative coordinate:
                # => assume same value than previous one
                blur_values.append(blur_values[-1])
        blur_values = numpy.array(blur_values)

        model = numpy.empty(self._image.shape, numpy.float64)

        # only compute the background model on the 3 first channels
        channels = min(self._image.shape[2], 3)

        for i in range(0, channels):
            logger.info("background model generation: "\
                "compute interpolation function on channel {}".format(i))
            bgfunction = scipy.interpolate.Rbf(
                xvalues, yvalues, blur_values[...,i].tolist(),
                function='multiquadric', smooth=self._smooth)

            logger.info("background model generation: "\
                "apply interpolation function on grid {}x{} [channel {}]"\
                .format(self._image.shape[0], self._image.shape[1], i))
            channel_model = self._apply_function_on_grid(bgfunction)
            model[...,i] = numpy.transpose(channel_model)

        return model

    # apply a function of (x, y) coordinates on a grid with same size than the
    # main image
    def _apply_function_on_grid(self, func):
        for row in range(0, self._image.shape[0], self._rows):
            row_max = min(row + self._rows, self._image.shape[0])
            logger.info( "background model generation: "\
                "... rows {} to {}...".format(row, row_max))
            xaxis, yaxis = numpy.mgrid[
                0:self._image.shape[1], row:row_max]
            if row == 0:
                res = func(xaxis, yaxis)
            else:
                res = numpy.concatenate((res, func(xaxis, yaxis)), axis=1)
        return res

class GradientRemove:
    """Substract a model of an image background gradient to this image"""

    def __init__(self, image, background):
        """Create the gradient remover object.

        Parameters
        ----------
        image : numpy.ndarray
            Image data.
        background : numpy.ndarray
            Background image data, extracted with BackgroundModel
        """
        self._image = image
        self._channels = self._image.shape[2]

        # adjust the background for each channel, so that its minimum value is
        # 0. this allows to retain the highest possible dymanic range from the
        # image by substracting the lowest possible values to even the image.
        self._background = background - \
            [background[...,i].min() for i in range(0, self._channels)]

    def run(self):
        """Substract the image background gradient.

        This substracts the background data from the image, and stretch the
        results so that the maximum possible value of an image point is not
        changed with the operation.

        Returns
        -------
        data : numpy.ndarray
            Resulting image data.
        """
        logger.info("remove background and stretch")

        # substract the adusted background and clip negative values
        res = self._image.astype(numpy.float64)
        for i in range(0, self._channels):
            res[...,i] -= self._background[...,i]
            res[...,i] = numpy.where(res[...,i] < 0, 0, res[...,i])

        # for a given point of coordinate P (x, y, channel), its value is in the
        # the range [0, max_value - background(P)].
        # We want to linearly stretch this interval to [0, max_value].
        # The transformation to apply is:
        # result(P) *= max_value / (max_value - background(P))

        max_val = self._max_value()
        stretch = max_val / (max_val - self._background)
        for i in range(0, self._channels):
            res[...,i] *= stretch[...,i]

        return res

    # maximum possible value of an image point
    def _max_value(self):
        dtype = self._image.dtype
        if numpy.issubdtype(dtype, numpy.integer):
            return numpy.iinfo(dtype).max
        else:
            return 1.0

def parse_args():
    help_description =\
        "Substract a background gradient from a TIFF file image"

    help_points =\
        "background point coordinates in the image. "\
        "those points should be chosen carefully to be of expected neutral "\
        "color once background gradient has been removed. points must be "\
        "provided as a space separated list of x, y coordinates."
    help_sigma =\
        "sigma value used to apply graussion blur on reference background "\
        "points. points up to a radius of 3 times the sigma value are used "\
        "for blurred value computation. (default=8.0)"
    help_background =\
        "background image output file. background is not dumped in file if "\
        "this option isn't specified."
    help_compress =\
        "zlib compression level for output file (default=0)"
    help_rows =\
        "number of rows on which background model is computed "\
        "simultaneously. high values require more memory (default=500)"
    help_smooth =\
        "Smoothness of the approximation. 0 is for an exact interpolation "\
        "greater values will smooth the approximation (default=0.1)"

    parser = argparse.ArgumentParser(description=help_description)
    parser.add_argument('-i', '--input', help='input file', action='store', required=True)
    parser.add_argument('-o', '--output', help='output file', action='store', required=True)
    parser.add_argument('-p', '--points', help=help_points, action='store', type=int, required=True, nargs='+')
    parser.add_argument('-s', '--sigma', help=help_sigma, action='store', type=float, default=8.0)
    parser.add_argument('-b', '--background', help=help_background, action='store')
    parser.add_argument('-c', '--compress', help=help_compress, action='store', type=int, default=0)
    parser.add_argument('-r', '--rows', help=help_rows, action='store', type=int, default=500)
    parser.add_argument('-m', '--smooth', help=help_smooth, action='store', type=float, default=0.1)
    parser.add_argument('-q', '--quiet', help='mute message output on stdout', action='store_true')
    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)

    if len(args.points) % 2 != 0:
        logger.error("odd number of point coordinates (should be even number)")
        return None

    args.bgpoints = [args.points[i:i+2] for i in range(0, len(args.points), 2)]

    args.inputfile = TiffReader(args.input)
    args.outputfile = TiffWriter(args.output)

    if not args.inputfile.is_valid() or not args.outputfile.is_valid():
        return None

    if args.background:
        args.bgfile = TiffWriter(args.background)
        if not args.bgfile.is_valid():
            return None
    else:
        args.bgfile = None

    args.inputdata = args.inputfile.load()
    if args.inputdata is None:
        return None
    if not args.inputfile.is_2d():
        return None
    kernel_limit = math.ceil(3 * args.sigma)
    if not args.inputfile.points_in_middle(args.bgpoints, kernel_limit):
        return None

    return args

def main():
    args = parse_args()
    if not args:
        return 1

    tags = args.inputfile.tags()

    bgmodel = BackgroundModel(
        args.inputdata, args.sigma, args.smooth, args.rows)
    bg = bgmodel.run(args.bgpoints)
    if args.bgfile:
        args.bgfile.write(bg, args.inputdata.dtype, args.compress, tags)

    rmgradient = GradientRemove(args.inputdata, bg)
    res = rmgradient.run()
    args.outputfile.write(res, args.inputdata.dtype, args.compress, tags)

    return 0

if __name__ == '__main__':
    main()
