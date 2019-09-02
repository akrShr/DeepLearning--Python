import unittest

import numpy as np


class TestCheckers(unittest.TestCase):
    def setUp(self):
        self.reference_img = np.load('reference_arrays/checker.npy')
        self.reference_img2 = np.load('reference_arrays/checker2.npy')

    def testPattern(self):
        import pattern
        c = pattern.Checker(250, 25)
        c.draw()
        np.testing.assert_almost_equal(c.output, self.reference_img)

    def testPatternDifferentSize(self):
        import pattern
        c = pattern.Checker(100, 25)
        c.draw()
        np.testing.assert_almost_equal(c.output, self.reference_img2)

    def testReturnCopy(self):
        import pattern
        c = pattern.Checker(100, 25)
        res = c.draw()
        res[:] = 0
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, res, c.output, "draw() did not return a copy!")


class TestSpectrum(unittest.TestCase):
    def setUp(self):
        self.reference_img = np.load('reference_arrays/spectrum.npy')
        self.reference_img2 = np.load('reference_arrays/spectrum2.npy')

    def testPattern(self):
        import pattern
        s = pattern.Spectrum(255)
        spec = s.draw()
        np.testing.assert_almost_equal(spec, self.reference_img)

    def testPatternDifferentSize(self):
        import pattern
        s = pattern.Spectrum(100)
        spec = s.draw()
        np.testing.assert_almost_equal(spec, self.reference_img2)

    def testReturnCopy(self):
        import pattern
        c = pattern.Spectrum(100)
        res = c.draw()
        res[:] = 0
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, res, c.output, "draw() did not return a copy!")


class TestCircle(unittest.TestCase):
    def setUp(self):
        self.reference_img = np.load('reference_arrays/circle.npy')
        self.reference_img2 = np.load('reference_arrays/circle2.npy')

    
    def testPattern(self):
        import pattern
        c = pattern.Circle(1024, 200, (512, 256))
        circ = c.draw()
        np.testing.assert_almost_equal(circ, self.reference_img)

    def testPatternDifferentSize(self):
        import pattern
        c = pattern.Circle(512, 20, (50, 50))
        circ = c.draw()
        np.testing.assert_almost_equal(circ, self.reference_img2)

    def testReturnCopy(self):
        import pattern
        c = pattern.Circle(512, 20, (50, 50))
        res = c.draw()
        res[:] = 0
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, res, c.output, "draw() did not return a copy!")


class TestGen(unittest.TestCase):
    def setUp(self):
        self.json_path = './Labels.json'
        self.file_path = './exercise_data/'

    def _get_corner_points(self, image):
        # expects batch of image - expected shape is [s,x,y,c]
        return image[:, [0, -1], :, :][:, : , [0, -1], :]

    def testInit(self):
        from generator import ImageGenerator

        gen = ImageGenerator(self.file_path, self.json_path, 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
        gen2 = ImageGenerator(self.file_path, self.json_path, 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
        np.testing.assert_almost_equal(gen.next()[0], gen2.next()[0])
        np.testing.assert_almost_equal(gen.next()[1], gen2.next()[1])

    def testResetIndex(self):
        from generator import ImageGenerator

        # Data contains 100 image samples, create two overlapping batches
        gen = ImageGenerator(self.file_path, self.json_path, 60, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
        b1 = gen.next()[0]
        b2 = gen.next()[0]
        np.testing.assert_almost_equal(b1[:20], b2[40:])

    def testShuffle(self):
        from generator import ImageGenerator

        gen = ImageGenerator(self.file_path, self.json_path, 10, [32, 32, 3], rotation=False, mirroring=False, shuffle=True)
        gen2 = ImageGenerator(self.file_path, self.json_path, 10, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, gen.next()[0], gen2.next()[0])
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, gen.next()[1], gen2.next()[1])

    def testRotation(self):
        from generator import ImageGenerator

        batch1 = ImageGenerator(self.file_path, self.json_path, 100, [32, 32, 3], rotation=False, mirroring=False, shuffle=False).next()[0]
        batch2 = ImageGenerator(self.file_path, self.json_path, 100, [32, 32, 3], rotation=True, mirroring=False, shuffle=False).next()[0]

        # determine the images which were augmented
        augmented_images_indices = np.sum(np.abs(batch1 - batch2), axis=(1, 2, 3)).astype(np.bool_)

        # extract corner points for each sample and reduce augmented to one characateristic row
        # this row is also inverted for simpler computations
        augmented_corners = self._get_corner_points(batch2[augmented_images_indices])
        characteristic_corners = augmented_corners[:, 0, ::-1]
        original_corners = self._get_corner_points(batch1[augmented_images_indices])

        # subtract characteristic corners to original corners. after summing spacial and channel dimensions, 0 are expected
        # e.g. if sample 2 is rotated by 90, we have a 0 in rot1 at the 2nd posistion see similiar to testMirroring
        rot1 = np.sum(original_corners[:, :, 0] - characteristic_corners, axis=(1, 2))
        rot2 = np.sum(original_corners[:, 1, :] - characteristic_corners, axis=(1, 2))
        rot3 = np.sum(original_corners[:, ::-1, 1] - characteristic_corners, axis=(1, 2))

        # assumption is that augmented images are either rotated by 90 (rot1), 180 (ro2) or 270 (rot3) degrees, thus
        # their elementwise product must be zero
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, batch1, batch2)
        np.testing.assert_almost_equal(np.sum(rot1 * rot2 * rot3), 0)

    def testMirroring(self):
        from generator import ImageGenerator

        batch1 = ImageGenerator(self.file_path, self.json_path, 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False).next()[0]
        batch2 = ImageGenerator(self.file_path, self.json_path, 12, [32, 32, 3], rotation=False, mirroring=True, shuffle=False).next()[0]

        # determine the images which were augmented
        augmented_images_indices = np.sum(np.abs(batch1 - batch2), axis=(1, 2, 3)).astype(np.bool_)

        # extract corner points for each sample
        augmented_corners = self._get_corner_points(batch2[augmented_images_indices])
        original_corners = self._get_corner_points(batch1[augmented_images_indices])

        # pick vertical ref points and compare them with opposites in original corners
        # compute then the diff and sum over vertical axis and channels. those images which were flipped vertically
        # contain now zeros
        vertical_augmented_corners = augmented_corners[:, :, 0]
        vertical = np.sum(original_corners[:, :, 1, :] - vertical_augmented_corners, axis=(1, 2))

        # pick horizontal ref points and compare them with opposites in original corners
        # compute then the diff and sum over horizontal axis and channels. those images which were flipped horizontally
        # contain now zeros
        horizontal_augmented_corners = augmented_corners[:, 0, :]
        horizontal = np.sum(original_corners[:, 1, :, :] - horizontal_augmented_corners, axis=(1, 2))

        # pick top left corner and bottom right corner (diagonals are flipped if double mirror)
        vertical_horizontal_augmented_corners = np.stack(
            list(zip(augmented_corners[:, 0, 0, :], augmented_corners[:, 1,  1, :])))
        original_corner_diagonals = np.stack(
            list(zip(original_corners[:, 1, 1, :], original_corners[:, 0, 0, :])))
        horizontal_vertical = np.sum(original_corner_diagonals - vertical_horizontal_augmented_corners, axis=(1, 2))


        # the elementwise product of horizontal and vertical must be zero
        # since the images can only be augmented with vertical or horizontal mirroring
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, batch1, batch2)
        np.testing.assert_almost_equal(np.sum(vertical * horizontal * horizontal_vertical), 0)

    def testResize(self):
        from generator import ImageGenerator

        batch = ImageGenerator(self.file_path, self.json_path, 12, [50, 50, 3], rotation=False, mirroring=False, shuffle=False).next()[0]
        self.assertEqual(batch.shape, (12, 50, 50, 3))

if __name__ == '__main__':
    unittest.main()