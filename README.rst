==========
rmgradient
==========

Copyright 2019, Guillaume Duranceau

https://github.com/drnc/rmgradient

*rmgradient* is a **background gradient remover from TIFF files**,
delivered as a **Python** script.
It builds a **background model** of the image from input points,
and substracts it.
This can be used to
**remove light pollution gradient from astrophotography images**
for example.

Background model
================

The background model is built from
input points coordinates on the image,
provided by the user.
Those points are assumed to all have the **same color**
once the background is removed.
When processing astrophotography pictures,
these points are typically chosen
in starless dark sky areas of the image.

The background model is built as
a multiquadratic `radial basis function`_ **interpolation**
of the color values for the chosen points in the image,
computed with scipy_.

The colors used for the interpolation
aren't directly the color values of the input points.
A `Gaussian blur`_ on the image is applied
at the input points coordinates,
so as to smooth input data.

Substracting the computed background model from the image
would result in a completely dark background on the output,
but would also very likely clip information from the input image.
Instead, each channel of the background model is **darkened**
so that its minimum value accross the whole image is zero,
before being substracted from the input image.
This allows to **level the background** of the input image,
while **preserving as much information as possible**
and giving more freedom in post processing.

Input background points
=======================

A few dozen of background points are usually enough
to get a quite precise background model
on typical astrophotography images.

Note that those points cannot be chosen
**too close to the borders** of the image.
The convolution kernel size
for the Gaussian blur applied on those points
is 3 times the standard deviation of the Gaussian distribution.
Using the default standard deviation of 8 (``--sigma`` option),
this means that background points should be at least
24 pixels away from the borders.

It's very common that large parts of the image
cannot be used as background dark points
(like when framing the Milky way or a nebula in an astrophotography).
If the background gradient pattern across the image
is simple and regular
(which is often the case with light pollution),
It's possible to help the interpolation process
by specifying **a series of points on an isoline**
(which are assumed to have the same background color).
The coordinates of those points should be given
as **negative values**::

    100 250    <-- background point of coordinate (100, 250)
    750 600    <-- another background point
    200 400    <-- background point P, starting a series on an isoline
    -300 -360  |
    -400 -310  |    points (300, 360), (400, 310), (500, 250)
    -500 -250  |<-- (550, 200) and (600, 140) are assumed to have
    -550 -200  |    the same background value than P (200, 400)
    -600 -140  |
    800 720    <-- back to a standard background point

Note that this can be used
to reference points close to image borders.

Features and options
====================

*rmgradient* performs all computation with 64 bits floating point values.
It writes the resulting TIFF image file
with the same type than the input image
(if input image is 16 bits TIFF files,
output image is a 16 bits TIFF file).

*rmgradient* can optionally compress the output TIFF file.

The background model can optionally be written
in a separate TIFF file.

The standard deviation (sigma) of the Gaussian distribution
used to blur the input background points
can be specified.
As described above,
this impacts the minimum distance to the borders of the input points.

Applying the interpolation function to the whole image
is a memory consuming operation.
To **limit memory consumption**,
*rmgradient* **applies the interpolation
by group of limited number of rows**
(100 by default,
but this can lowered
to accomodate a user environment with low memory constraints).

The smoothness of the background model approximation
can be controlled.

How to run
==========

rmgradient.py_ is a Python 3 script,
depending on numpy_, scipy_ and tifffile_ packages.

To run it and display the usage and options::

    python rmgradient.py -h

Usually, coordinates of input points chosen as reference for the background
are written in a file::

    x1 y1
    x2 y2
    ...

This file can then used as an input this way::

   cat input_points | xargs python rmgradient.py -i image.tif -o output.tif [options] -p

An example image and input points are available
in the `test data folder`_,
together with a test_sky_image.sh_ script,
to demonstrate how to use *rmgradient*.

.. _radial basis function: https://en.wikipedia.org/wiki/Radial_basis_function
.. _Gaussian blur: https://en.wikipedia.org/wiki/Gaussian_blur
.. _numpy: http://www.numpy.org/
.. _scipy: https://www.scipy.org/
.. _tifffile: http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html
.. _rmgradient.py: https://github.com/drnc/rmgradient/blob/master/rmgradient/rmgradient.py
.. _test data folder: https://github.com/drnc/rmgradient/tree/master/rmgradient/test_data
.. _test_sky_image.sh: https://github.com/drnc/rmgradient/blob/master/rmgradient/test_sky_image.sh
