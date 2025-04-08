import numpy as np
from src.rearrange import rearrange 

invalid_patterns = [
    "a b -> (a b",               
    "a b -> a * b",             
    "(a b)) -> a b",             
    "a b) -> (a b)",             
    "a b -> a b)",               
    "a b -> a c",                
    "... a -> b ...",            
    "(a b) -> c d",              
    "(a b c) -> a b c",          
    "a a -> a",                  
    "a b -> a b b",              
    "(a b) c -> a b c",          
    "(a b) -> a b",              
    "... -> ... ...",            
    "a b ... -> ... ... b",      
    "a-b -> b",                  
    "a$ -> a",                   
    "123 -> 321",                
]

def test_invalid_patterns():
    x = np.random.rand(2, 3, 4)
    for i, pattern in enumerate(invalid_patterns):
        try:
            print(f"Testing invalid pattern {i+1}: '{pattern}'")
            rearrange(x, pattern)
            raise AssertionError(f"Pattern {i+1} did not raise an error: '{pattern}'")
        except Exception as e:
            print(f"Caught expected error: {e.__class__.__name__}: {e}")

    print("All invalid patterns correctly raised errors!")

test_invalid_patterns()
