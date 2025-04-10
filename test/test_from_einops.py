'''
Here, I have taken the cases that were given in the einops repo for testing
'''
from src.rearrange import rearrange
from einops import rearrange as einops_rearrange
import numpy as np

tensor = np.random.rand(2, 3, 4, 5, 6)
axes_lengths = {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}

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

equivalent_rearrange_patterns = [
    ("a b c d e -> (a b) c d e", "a b ... -> (a b) ... "),
    ("a b c d e -> a b (c d) e", "... c d e -> ... (c d) e"),
    ("a b c d e -> a b c d e", "... -> ... "),
    ("a b c d e -> (a b c d e)", "... ->  (...)"),
    ("a b c d e -> b (c d e) a", "a b ... -> b (...) a"),
    ("a b c d e -> b (a c d) e", "a b ... e -> b (a ...) e"),
]

def test_patterns():
    print("Testing identity patterns:")
    for pattern in identity_patterns:
        print(f"Pattern: {pattern}")
        try:
            out1 = einops_rearrange(tensor, pattern, **axes_lengths)
            out2 = rearrange(tensor, pattern, **axes_lengths)
            if out1.shape == out2.shape:
                print(f"Match: Shape {out1.shape}")
            else:
                print(f"Mismatch in output shapes: {out1.shape} vs {out2.shape}")
        except Exception as e1:
            try:
                rearrange(tensor, pattern, **axes_lengths)
                print(f"Custom function raised no error, but einops did: {e1}")
            except Exception:
                print(f"Both raised errors")

    print("\nTesting equivalent patterns:")
    for p1, p2 in equivalent_rearrange_patterns:
        print(f"Pattern 1: {p1} | Pattern 2: {p2}")
        try:
            out1 = einops_rearrange(tensor, p1, **axes_lengths)
            out2 = rearrange(tensor, p2, **axes_lengths)
            if out1.shape == out2.shape:
                print(f"Match: Shape {out1.shape}")
            else:
                print(f"Mismatch: {out1.shape} vs {out2.shape}")
        except Exception as e1:
            try:
                rearrange(tensor, p2, **axes_lengths)
                print(f"Custom function raised no error, but einops did: {e1}")
            except Exception:
                print(f"Both raised errors")

'''
Hint: The idea behind this was to test out if both eionops and my implementation fail at the same time, and to check if the output shapes have mismatch in dimensions, to check for each case,
kindly refer to the notebook.
'''

if __name__ == "__main__":
    test_patterns()
