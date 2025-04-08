import numpy as np
import re
from typing import Tuple, List, Dict

def validate_pattern(pattern: str):
    if '->' not in pattern:
        raise ValueError("Pattern must contain '->'")

    lhs, rhs = pattern.split('->')
    
    for side, name in [(lhs, "input"), (rhs, "output")]:
        if side.count('(') != side.count(')'):
            raise ValueError(f"Unbalanced parentheses in {name} pattern: '{side}'")
        
        if side.count('...') > 1:
            raise ValueError(f"Too many ellipses in {name} pattern: '{side}'")
        
        invalid = re.findall(r'[^\w\s\.\(\)]', side)
        if invalid:
            raise ValueError(f"Invalid characters in {name} pattern: {set(invalid)}")
        
        axes = flatten_axes(find_axes(side))
        axes_wo_ellipsis = [ax for ax in axes if ax != '...']
        if len(axes_wo_ellipsis) != len(set(axes_wo_ellipsis)):
            raise ValueError(f"Duplicate axes found in {name} pattern: {axes_wo_ellipsis}")

    return True

def parse(pattern: str) -> Tuple[List[str], List[str]]:
  if '->' not in pattern:
    raise ValueError("Pattern does not ->, make sure that the pattern is right")
  input = pattern.split('->')[0]
  output = pattern.split('->')[1]
  input = find_axes(input)
  output = find_axes(output)
  return input, output

def find_axes(pattern:str ) -> List[str]:
    return re.findall(r'\.\.\.|[\w]+|\([^\)]+\)', pattern)

def flatten_axes(axes: List[str]) -> List[str]:
    flat = []
    for ax in axes:
        if ax.startswith("(") and ax.endswith(")"):
            # flat += re.findall(r'\w+', ax)
            flat += re.findall(r'\w+|\.\.\.', ax)
        else:
            flat.append(ax)
    return flat

def transpose(tensor, input, output):
  for i in output:
    if i not in input:
      raise ValueError(f"Output axis '{ax}' not found in input axes")
  perm = [input.index(ax) for ax in output]
  return tensor.transpose(perm)

def is_transpose(pattern) -> bool:
  return ('...' not in pattern) and ('(' not in pattern) and (')' not in pattern)

# def is_split_axes(pattern: str) -> bool:
#   return ( '(' in pattern.split('->')[0] and '(' not in pattern.split('->')[1] )

def split_axes(ip_ax: str, tensor_shape: int, axes_lengths: Dict[str, int], shape: Dict[str, int]) -> List[str]:
  # print(ip_ax)
  axes = re.findall(r'\w+', ip_ax)
  print(axes)

  known_ax = [i for i in axes if i in axes_lengths]
  known_dim = [axes_lengths[i] for i in axes if i in axes_lengths]
  print(known_ax, known_dim)

  unknown_ax = [i for i in axes if i not in axes_lengths]
  print(unknown_ax)

  if len(unknown_ax) == 0:
    return axes

  prod = int(np.prod(known_dim))

  if len(unknown_ax) == 1:
    axes_lengths[unknown_ax[0]] = tensor_shape // prod

  sizes = [axes_lengths.get(i) for i in axes]

  if None in sizes:
    raise ValueError(f"Missing axis lengths for: {[i for i, s in zip(axes, sizes) if s is None]}")

  if tensor_shape != np.prod(sizes):
    raise ValueError(f'Mismatch in shapes: {tensor_shape} cannot be split into {sizes}')
    # return None
  for i, s in zip(axes, sizes):
    shape[i] = s
  return axes


# def merge_axis(op_ax, axes_shape):  
  # ax_split = re.findall(r'\w+', op_ax)
  # return int(np.prod([axes_shape[i] for i in ax_split]))
def merge_axis(op_ax: str, axes_shape: Dict[str, int], ellipsis_ax: List[str] = []):
    # Handle ellipsis inside parentheses
    tokens = re.findall(r'\w+|\.{3}', op_ax)
    resolved_axes = []
    for token in tokens:
        if token == '...':
            resolved_axes.extend(ellipsis_ax)
        else:
            resolved_axes.append(token)
    return int(np.prod([axes_shape[i] for i in resolved_axes]))

def repeat_new_axes(tensor: np.ndarray,
    input_axes: List[str],
    output: List[str],
    axes_shape: Dict[str, int],
    axes_lengths: Dict[str, int]) -> Tuple[np.ndarray, List[str]]:
  
  new_ax = [i for i in output if i not in input_axes]
  # print(new_ax)

  for i in new_ax:
    if i.startswith('(') or i == '...':
      continue
    if i not in axes_lengths:
      raise ValueError(f"The axis {i} needs a specified length")

    tensor = np.expand_dims(tensor, axis=-1)
    tensor = np.repeat(tensor, axes_lengths[i], axis=-1)

    axes_shape[i] = axes_lengths[i]
    input_axes.append(i)

    target_index = output.index(i)
    tensor = np.moveaxis(tensor, -1, target_index)
    input_axes = input_axes[:-1]
    input_axes.insert(target_index, i)

  return tensor, input_axes

def ellipsis(tensor_shape: List[int], axes: List[str]) -> Tuple[List[int], List[str], int]:
  # n = len(axes)
  n_ellipsis = len(tensor_shape) - len(axes)

  if n_ellipsis < 0:
    raise ValueError("Not enough dimensions in input tensor")
  
  ellipsis_dim = tensor_shape[:n_ellipsis]
  ellipsis_ax = [f'_ellipsis_{i}' for i in range(n_ellipsis)]
  return ellipsis_dim, ellipsis_ax, n_ellipsis

def extract_axes(expr: str, ellipsis_ax: List[str]) -> List[str]:
    ''' Hint: added this to handle cases where ellipsis occur in the merge axis operation, 
    and also when the case becomes too complex with the transpose part coming in as well '''
    expr = expr.strip("()")
    return [ax if ax != '...' else ell for ax in expr.split() for ell in (ellipsis_ax if ax == '...' else [ax])]


def rearrange(tensor: np.array, pattern: str, **axes_lengths) -> np.ndarray:
    
    validate_pattern(pattern)
    input, output = parse(pattern)
    # print(input, output)
    # if is_transpose(pattern):
        # print(transpose(tensor, input, output))
        # return transpose(tensor, input, output)

    # print("...")
    pattern_axes = set(input + output)
    extra_keys = set(axes_lengths.keys()) - pattern_axes
    if extra_keys:
      raise ValueError(f"Extra keys in axes_lengths not used in pattern: {extra_keys}")

    axes_shape = {}
    input_axes = []
    # output_axes = []
    ellipsis_dim = []
    axes = []

    for ip_ax in input:
        if ip_ax == '...':
          continue
        elif ip_ax.startswith("(") and ip_ax.endswith(")"):
          axes += re.findall(r'\w+', ip_ax)
        else:
          axes.append(ip_ax)
    # print(axes)

    tensor_shape = list(tensor.shape)
    # print(tensor_shape)

    '''ellipsis_dim = []
    ellipsis_ax = []
    n_ellipsis = 0
    if '...' in pattern:
      ellipsis_dim, ellipsis_ax, n_ellipsis = ellipsis(input, tensor_shape, axes)'''
    # else:
    #   ellipsis_dim = []
    #   ellipsis_ax = []
    #   n_ellipsis = 0

    n_ellipsis = len(tensor_shape) - len(axes)
    if '...' in pattern:
      if n_ellipsis < 0:
        raise ValueError("Too many axoes in pattern for input tensor")

    ellipsis_ax = [f'_ellipsis_{i}' for i in range(n_ellipsis)]
    ellipsis_dim = tensor_shape.copy()
    if '...' in pattern:
      for i in axes:
        ind = tensor_shape.index(axes_lengths.get(i, None)) if i in axes_lengths else None
        if ind is not None:
          ellipsis_dim.pop(i)
    # idx = n_ellipsis
    idx = 0
    for ip_ax in input:
        if ip_ax == '...':
          # continue
          # if len( ellipsis_ax) > 0:
          # for ax, dim in zip(ellipsis_ax, ellipsis_dim): # changed ellipses_ax to ellipsis_ax
          #   axes_shape[ax] = dim
          # input_axes.extend(ellipsis_ax) # changed ellipses_ax to ellipsis_ax
          # else:
            # continue
          for i in range(n_ellipsis):
            axes_shape[ellipsis_ax[i]] = tensor_shape[idx]
            input_axes.append(ellipsis_ax[i])
            idx += 1

        elif ip_ax.startswith('(') and ip_ax.endswith(')'):
          ax_split = split_axes(ip_ax, tensor_shape[idx], axes_lengths, axes_shape)
          input_axes.extend(ax_split)
          idx += 1

        # else:
        #   axes_shape[ip_ax] = tensor_shape[idx]
        #   input_axes.append(ip_ax)
        #   idx += 1
        else:
          if idx < len(tensor_shape):  
            axes_shape[ip_ax] = tensor_shape[idx]
            input_axes.append(ip_ax)
            idx += 1
          else:
            raise ValueError(f"Input pattern '{pattern}' expects more dimensions than the input tensor has.") 

    # print(axes_shape)
    # print("Inferred Input Axes:", input_axes)
    # tensor = tensor.reshape()
    tensor_reshaped = tensor.reshape([axes_shape[i] for i in input_axes])
    # print(tensor_reshaped)

    new_axes_to_repeat = set(output) - set(input_axes)

    if new_axes_to_repeat:
      tensor_reshaped, input_axes = repeat_new_axes(tensor_reshaped, input_axes, output, axes_shape, axes_lengths)

    # print('...', input_axes, output)
    # print(tensor_reshaped, input_axes)

    final_axes_order = []
    for op_ax in output:
        if op_ax == '...':
            final_axes_order.extend(ellipsis_ax)
        elif op_ax.startswith('(') and op_ax.endswith(')'):
            final_axes_order.extend(extract_axes(op_ax, ellipsis_ax))
            # continue
        else:
            final_axes_order.append(op_ax)

    if input_axes != final_axes_order:
        perm = [input_axes.index(ax) for ax in final_axes_order]
        tensor_reshaped = tensor_reshaped.transpose(perm)
        input_axes = final_axes_order

    output_ax_dim = []
    for op_ax in output:
        if op_ax == '...':
          # continue
          output_ax_dim.extend([axes_shape[i] for i in ellipsis_ax])
        elif op_ax.startswith('(') and op_ax.endswith(')'):
          # output_ax_dim.append(merge_axis(op_ax, axes_shape))
          output_ax_dim.append(merge_axis(op_ax, axes_shape, ellipsis_ax))
        else:
          output_ax_dim.append(axes_shape[op_ax])

    # print("final output shape:", output_ax_dim)

    return tensor_reshaped.reshape(output_ax_dim)

                     
