import numpy as np
"""
from cs231n class materials

"""



def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N,C,H,W= x_shape

    out_height = (H+2*padding-field_height)/stride +1
    out_width  = (W+2*padding-field_width)/stride +1
    print(out_height,out_width)
    i0 = np.repeat(np.arange(field_height),field_width)
    i0 = np.tile(i0,C)
    i1 = stride*np.repeat(np.arange(out_height),out_width)
    j0 = np.tile(np.arange(field_width),field_height*C)
    j1 = stride*np.tile(np.arange(out_width),out_height)

    i = i0.reshape(-1,1)+i1.reshape(1,-1)
    j = j0.reshape(-1,1)+j1.reshape(1,-1)

    k = np.repeat(np.arange(C),field_width*field_height).reshape(-1,1)
    return (k,i,j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """
    converts input into an array with receptive fields as columns
    in vector form
    """
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k,i,j = get_im2col_indices(x.shape,field_height,field_width,0,stride)
    #array of 2D arrays containg all receptor fields as columns and channels group into rows
    #each is per batch sample
    cols = x[:,k,i,j]
    #channel axis size
    C = x.shape[1]
    #flip 3rd axis and 1st axis so that recepive fields corresponding to the same field in
    #a different sample are in the same matrix
    cols = cols.transpose(2,1,0)
    #make it into one 2D array with corresponding receptive fields form different
    #sample next to eachother
    cols =cols.reshape(C*field_height*field_width,-1)
    return cols


a = np.arange(36).reshape(2,2,3,3)
print(a)
col = im2col_indices(a,2,2)
print(col)
print(col.shape)


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  padding,stride = conv_param["pad"],conv_param["stride"]
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  H_p = 1+(H+2*pad-HH)/stride
  W_P = 1+(W+2*pad-WW)/stride
  out = None
  x_col = im2col_indices(x,HH,WW,padding,stride)
  #linearize weight matrices, each row is a different filter
  w_col = w.reshape(F,-1)
  #each row represents the result of the application of
  #a different filter.
  res = np.dot(w_col,x_col)+b.reshape(F,-1)
  #group same pixels in different samples together
  #pixels gouped based on out height and width
  out  = res.reshape(F,H_p,W_p,x.shape[0]).transpose(3,0,1,2)

  cache=(x,w,b,conv_param,x_cols)
  return out,H_P,W_P,cache



















  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  #each weight matrix for each feature is a row (weight matrix in vector form)
  col = im2col_indices(x, w.shape[2], w.shape[3], padding=conv_param['pad'], stride=conv_param['stride'])
  out_col = w.reshape(w.shape[0],-1).dot(col) + b.reshape(-1,1)

  out = out_col.reshape(w.shape[0],out_height,out_width)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db
co
