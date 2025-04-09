import numpt as n
from src.rearrange import rearrange

rearrange_patterns = [
    ("a b -> b a"),

    ("a b c d -> (a b) (c d)"),
    ("(a b) (c d) -> a b c d"),

    ("... h w -> (...) (h w)"),
    ("b ... h w -> b (...) (h w)"),
    ("b ... h w -> (b ...) h w"),

    ("a b c d e -> e d c b a"),
    ("(a b) c (d e) -> a b d e c"),

    ("a b c d -> a (b c) d"),
    ("a b c d -> (a d) b c"),
    
    ("a 1 c -> a b c",),  
    ("a 1 c -> a (1 c)"),

    ("(a b c) -> a b c",),  
    ("(a b) (c d) -> a b c d"),

    ("... x y z -> ... (x y z)"),
    ("a ... b -> a (...) b"),

    ("a b c -> a b c"),
    ("... -> ..."),

    ("a b c d e -> (a b c) (d e)"),

    ("a b -> a b 1"),
    ("(a b) -> a b",),  

    ("a b c d -> (...) d"),
    
    ("a b c d -> (a b c) d"),
    ("a b c d -> a (b c d)"),

    ("a b c d e f -> (a b) (c d) (e f)"),
    
    ("a b c d -> b c d a"),
    ("a b c d -> d a b c"),

    ("a b c d -> a (b d) c"),
]

def test_rearrange_patterns():
    x = np.random.rand(2, 3, 4, 5, 6) 
    for i, pattern in enumerate(rearrange_patterns):
        try:
            print(f"Testing pattern {i+1}: '{pattern}'")
            y = rearrange(x, pattern, a=2, b=3, c=4, d=5, e=6) 
            print(f"Pattern {i+1} succeeded. Output shape: {y.shape}")
        except Exception as e:
            print(f"Pattern {i+1} failed with error: {e}")

test_rearrange_patterns()
