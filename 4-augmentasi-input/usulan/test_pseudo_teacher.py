import unittest
import numpy as np
from pseudo_teacher import promote


class TeacherTests(unittest.TestCase):
    def test_promotes_without_mutating_inputs_and_handles_both_targets(self):
        x = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.]], dtype=np.float32)
        u = np.array([[.2, 1.2], [.8, 1.8], [1.2, 2.2]], dtype=np.float32)
        original = u.copy()
        for classification, y in ((True, np.array([0, 0, 1, 1])),
                                   (False, np.array([0., 1., 2., 3.], dtype=np.float32))):
            xp, yp, info = promote(x, y, u, classification)
            self.assertEqual((len(xp), len(yp), info['promoted']), (6, 6, 2))
            np.testing.assert_array_equal(u, original)
            self.assertTrue(np.isfinite(yp).all())


if __name__ == '__main__':
    unittest.main()
