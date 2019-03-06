import logging
import numpy
import rmgradient
import unittest
import os
import tifffile

def datafile(filename):
    return os.path.join('test_data', filename)

def filereader(filename):
    return rmgradient.TiffReader(datafile(filename))

def filewriter(filename):
    return rmgradient.TiffWriter(datafile(filename))

image = rmgradient.TiffReader(datafile('image.tif')).load()

image_points = [
    [3, 3], [-3, 0], [7, 5], [13, 4],
    [5, 8], [15, 8], [8, 11], [12, 12],
    [9, 4], [10, 9], [9, 12], [9, 16],
    [3, 16], [-3, -19], [8, 15], [16, 16],
    [-16, -19], [-16, 0]]

# Maximum difference between image.tif and its background in the 10x10 middle square
def image_background_middle_max_diff(sigma, smooth):
    model = rmgradient.BackgroundModel(image, sigma, smooth)
    bg = model.run(image_points).astype(numpy.uint16)
    return abs(image.astype(float)[5:15, 5:15] - bg.astype(float)[5:15, 5:15]).max()

class TestTiffReader(unittest.TestCase):

    def test_name(self):
        self.assertEqual(rmgradient.TiffReader(datafile('name')).name(), datafile('name'))

    def test_invalid(self):
        self.assertFalse(rmgradient.TiffReader(datafile('unexisting-file')).is_valid())

    def test_load(self):
        self.assertEqual(rmgradient.TiffReader(datafile('1d.tif')).load().tolist(), [10, 100, 1000, 10000])
        self.assertEqual(rmgradient.TiffReader(datafile('image.tif')).load()[5, 3].tolist(), [4162, 7908, 2037])

    def test_is_2d(self):
        self.assertFalse(rmgradient.TiffReader(datafile('1d.tif')).is_2d())
        self.assertTrue(rmgradient.TiffReader(datafile('image.tif')).is_2d())

    def test_points_in_middle(self):
        # check single point
        self.assertFalse(rmgradient.TiffReader(datafile('1d.tif')).points_in_middle([[2, 2]], 1))
        self.assertFalse(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[0, 0]], 1))
        self.assertFalse(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[19, 19]], 2))
        self.assertFalse(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[0, 18]], 2))
        self.assertFalse(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[17, 18]], 3))
        self.assertFalse(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[16, 6]], 4))
        self.assertFalse(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[4, 3]], 4))
        self.assertTrue(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[17, 18]], 1))
        self.assertTrue(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[16, 16]], 3))
        self.assertTrue(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[19, 19]], 0))
        self.assertTrue(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[4, 3]], 3))
        self.assertTrue(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[10, 10]], 8))

        # check multiple points
        self.assertFalse(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[17, 18], [16, 16], [4, 3]], 3))
        self.assertFalse(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[4, 3], [16, 16], [17, 18]], 3))
        self.assertTrue(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[4, 3], [16, 16], [17, 18]], 1))
        self.assertTrue(rmgradient.TiffReader(datafile('image.tif')).points_in_middle([[4, 3], [16, 16]], 3))

class TestTiffWriter(unittest.TestCase):

    def test_invalid(self):
        self.assertFalse(rmgradient.TiffWriter(datafile('unexisting-dir/out.tif')).is_valid())

    def test_write(self):
        out_filename = datafile('out.tif')
        out = rmgradient.TiffWriter(out_filename)
        out.write(image, numpy.uint16)
        out_read = rmgradient.TiffReader(out_filename).load()
        self.assertEqual(out_read.tolist(), image.tolist())
        self.assertEqual(out_read.dtype, numpy.uint16)
        os.remove(out_filename)

    def test_write_compressed(self):
        out_filename = datafile('out.tif')
        out = rmgradient.TiffWriter(out_filename)
        out.write(image, numpy.uint16, 5)
        out_read = rmgradient.TiffReader(out_filename).load()
        self.assertEqual(out_read.tolist(), image.tolist())
        self.assertEqual(out_read.dtype, numpy.uint16)
        os.remove(out_filename)

class TestBlurImage(unittest.TestCase):

    def test_blur_points_sigma_1(self):
        blur = rmgradient.BlurImage(image, 1.0)
        self.assertEqual([int(i) for i in blur.run([5, 5])], [5970, 12054, 3034])
        self.assertEqual([int(i) for i in blur.run([15, 15])], [16006, 32023, 8143])
        self.assertEqual([int(i) for i in blur.run([10, 10])], [10996, 22022, 5516])
        self.assertEqual([int(i) for i in blur.run([12, 4])], [12816, 25887, 6659])
        self.assertEqual([int(i) for i in blur.run([11, 8])], [12040, 23925, 6106])

    def test_blur_points_sigma_25(self):
        blur = rmgradient.BlurImage(image, 2.5)
        self.assertEqual([int(i) for i in blur.run([10, 10])], [10993, 22028, 5531])
        self.assertEqual([int(i) for i in blur.run([11, 8])], [11999, 23990, 6065])

class TestBackgroundModel(unittest.TestCase):

    def test_bgmodel(self):
        self.assertTrue(image_background_middle_max_diff(sigma=1.0, smooth=0) < 1500)
        self.assertTrue(image_background_middle_max_diff(sigma=1.0, smooth=0.1) < 1500)
        self.assertTrue(image_background_middle_max_diff(sigma=1.0, smooth=1.0) < 1500)
        self.assertTrue(image_background_middle_max_diff(sigma=0.5, smooth=0) < 1500)
        self.assertTrue(image_background_middle_max_diff(sigma=0.5, smooth=0.1) < 1500)
        self.assertTrue(image_background_middle_max_diff(sigma=0.5, smooth=1.0) < 1500)

    def test_bgmodel_rows(self):
        models = [
            rmgradient.BackgroundModel(image, 1.0, 0.1),
            rmgradient.BackgroundModel(image, 1.0, 0.1, rows=10),
            rmgradient.BackgroundModel(image, 1.0, 0.1, rows=2),
            rmgradient.BackgroundModel(image, 1.0, 0.1, rows=1)]

        bg = [model.run(image_points).tolist() for model in models]
        self.assertEqual(bg[0], bg[1])
        self.assertEqual(bg[0], bg[2])
        self.assertEqual(bg[0], bg[3])

class TestGradientRemove(unittest.TestCase):

    def test_rmgradient(self):
        model = rmgradient.BackgroundModel(image, 1.0)
        bg = model.run(image_points)
        res = rmgradient.GradientRemove(image, bg).run()
        self.assertTrue(res.min() > 0) # no clipped data
        # no real check on the output

if __name__ == '__main__':
    logging.basicConfig(level=100) # deactivate logging
    unittest.main()
