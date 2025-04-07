'''
Here, I have taken the cases that were given in the einops repo for testing
'''

import unittest
import numpy as np

class TestRearrangePatterns(unittest.TestCase):

    def setUp(self):
        self.tensor = np.random.rand(2, 3, 4, 5, 6)  
        self.axes_lengths = {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}

    def test_identity_patterns(self):
        identity_patterns = [
            "...->...",
            "a b c d e-> a b c d e",
            "a b c d e ...-> ... a b c d e",
            "a b c d e ...-> a ... b c d e",
            "... a b c d e -> ... a b c d e",
            "a ... e-> a ... e",
            "a ... -> a ... ",
            "a ... c d e -> a (...) c d e",
        ]
        for pattern in identity_patterns:
            with self.subTest(pattern=pattern):
                out = rearrange(self.tensor, pattern, **self.axes_lengths)
                np.testing.assert_allclose(out, self.tensor)

    def test_equivalent_rearrange_patterns(self):
        equivalent_rearrange_patterns = [
            ("a b c d e -> (a b) c d e", "a b ... -> (a b) ... "),
            ("a b c d e -> a b (c d) e", "... c d e -> ... (c d) e"),
            ("a b c d e -> a b c d e", "... -> ... "),
            ("a b c d e -> (a b c d e)", "... ->  (...)"),
            ("a b c d e -> b (c d e) a", "a b ... -> b (...) a"),
            ("a b c d e -> b (a c d) e", "a b ... e -> b (a ...) e"),
        ]
        for pattern1, pattern2 in equivalent_rearrange_patterns:
            with self.subTest(p1=pattern1, p2=pattern2):
                out1 = rearrange(self.tensor, pattern1, **self.axes_lengths)
                out2 = rearrange(self.tensor, pattern2, **self.axes_lengths)
                np.testing.assert_allclose(out1, out2)

if __name__ == "__main__":
    unittest.main()
