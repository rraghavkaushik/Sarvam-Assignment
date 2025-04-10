# Sarvam-Assignment
This repository contains a custom implementation of the `rearrange` function, inspired by the `einops.rearrange` operation. This implementation supports transposition, splitting, merging, repeating axes and ellipsis handling in NumPy arrays.

---
## Implementing rearrange method of einops from scratch

### My approach and design decisions in brief:

I have tried to implement the rearrange method from einops in the most efficient way, with proper error handling and with proper support for complex patterns and edge cases.

I have re-iterated this process to handle all cases to the best of my knowledge.

- My initial approach was to keep everything modular, but that turned out not to be the best decision, as I tried creating a transpose() method just for handling vanilla transpose conditions, but that didn't produce much of a difference in terms of the time taken to execute.

- So, the idea was to modularize the code wherever possible, and do it only if neccessary.

- I have created a parser function that makes sure that the input and the output axes are captured correctly, further using regular expressions to get individual axes (turned out to be the most efficient way).

- Major issues I faced were with split axes and ellipses operations, as they were not very easy to develop in the first phase of coding. Hence, ended up changing them multiple times. With the split axes, the problem was with the axes lengths and the problem with ellipsis was the axes it covered. I have added error handling cases to handle mismatch in shapes, and also with missing inputs.

- So, then to count get the unique input axes, used simple if elif conditions, to get axes in case of ellipsis and to handle split axes operations. Added error handling to make sure the input pattern and dimensions match with each other.

- Now, comes the reshaping of the tensor according to the input pattern's dimensions.

- After this, the repeating axes condition has to be handled, for that, I went with an efficient method of figuring out if there were differences in the input and output axes using unique axes from the both of them.

- Now, in some cases, I faced an issue of operations mixed together, wherein, there was a mix of transpose, merge axes and ellipsis ops. So, for this case, I had to change the input axes in the right order accoring to the output, if they didn't match.

- Now, finally, the tensor has to be reshaped according to the shape of the output dimensions, so added conditions to check for ellipsis, merge operations seperately. In some cases, the intial merge condition failed when ellipses were included, so had to add another function to handle this case.

## Code

Steps to clone the repository and run the code.

#### 1. Clone the Repository

```bash
git clone https://github.com/rraghavkaushik/Sarvam-Assignment
```

#### 2. Change Directory

```bash
cd Sarvam-Assignment
```

#### 3. Run the Main File

To execute the custom `rearrange` implementation:

```bash
python -m src.rearrange
```

---

### Running Tests

Please refer to 'einops-rearrange_from_scratch.ipynb' from notebook folder for case by case implementation. 

The repo includes multiple test modules to validate functionality against known cases and ensure correctness.

#### Input Pattern Testing

Tests different rearrangement patterns with various axes and tensor shapes.

```bash
python -m test.input_pattern_test
```

#### Equivalent Pattern Testing

Validates whether different patterns that should produce the same result actually do.

```bash
python -m test.rearrange_pattern_test
```

#### Runs sample test cases directly adapted from the original `einops` repo to ensure parity.

```bash
python -m test.test_from_einops
```

---

### Structure of the repo

```
Sarvam-Assignment/
├── notebook/
│   └── einops-rearrange_from_scratch.ipynb
├── src/
│   └── rearrange.py             
├── test/
│   ├── input_pattern_test.py     
│   ├── rearrange_pattern_test.py 
│   └── test_from_einops.py       
└──README.md
```
