import os

import numpy as np
import scipy.io
import scipy.misc
from skimage.transform import resize as imresize


# Forward path implementation of AlexNet


def conv(bottom, weight, bias, K, S, pad, group):
    """
    Convolution Layer.
    bottom is a 3d matrix: Win x Hin x N.
    top is a 3d matrix: Wout x Hout x M.
    weight is a 4d matrix: K x K x N x M (or K x K x N/2 x M in case of group==2).
    bias is a 4d matrix: 1 x 1 x 1 x M.
    Kernel size K and stride S are integers.
    Padding 'pad' specifies the number of pixels to (implicitly) add to each
    side of the input.
    If group==1, the OFMs depend on all IFMs. If group==2, the IFMs and OFMs are
    split into two groups, therefore the OFMs only depend on half the IFMs.
    """
    [Win, Hin, N] = bottom.shape
    M = weight.shape[3]
    bottomPadded = np.zeros(Win + 2 * pad, Hin + 2 * pad, N)
    bottomPadded[pad + 1:bottom.shape[0] - pad, pad + 1:bottom.shape[1] - pad, :] = bottom
    Wout = int((Win + 2 * pad - K) / S + 1)
    Hout = int((Hin + 2 * pad - K) / S + 1)
    top = np.zeros(Wout, Hout, M)
    # Convolve kernels with input feature maps.
    # Code is vectorized over N: For one specific output feature map,
    # convolve all input feature maps and the corresponding kernels at the
    # same time.
    if group == 1:
        # This is the normal convolution where every output feature map
        # depends on every input feature map.
        for w in range(1, Wout):
            for h in range(1, Hout):
                for m in range(1, M):
                    wStart = (w - 1) * S + 1
                    wEnd = wStart + K - 1
                    hStart = (h - 1) * S + 1
                    hEnd = hStart + K - 1
                    top[w, h, m] = top[w, h, m] + sum(sum(sum(
                        bottomPadded[range(wStart, wEnd), range(hStart, hEnd), range(1, N)] *
                        weight[:, :, range(1, N), m])))
                    # In this special case, the output feature maps depend on only half
                    # the input feature maps.
    elif group == 2:
        # Group 1
        for w in range(1, Wout):
            for h in range(1, Hout):
                for m in range(1, M / 2):
                    wStart = (w - 1) * S + 1
                    wEnd = wStart + K - 1
                    hStart = (h - 1) * S + 1
                    hEnd = hStart + K - 1
                    top[w, h, m] = top[w, h, m] + sum(sum(sum(
                        bottomPadded[range(wStart, wEnd), range(hStart, hEnd), range(1, N / 2)]) * \
                                                          weight[:, :, range(1, N / 2), m]))
        # Group 2
        for w in range(1, Wout):
            for h in range(1, Hout):
                for m in range(M / 2 + 1, M):
                    wStart = (w - 1) * S + 1
                    wEnd = wStart + K - 1
                    hStart = (h - 1) * S + 1
                    hEnd = hStart + K - 1
                    top[w, h, m] = top[w, h, m] + sum(sum(sum(
                        bottomPadded[range(wStart, wEnd), range(hStart, hEnd), range(N / 2 + 1, N)] *
                        weight[:, :, range(1, N / 2), m])))
    else:
        raise ValueError('Convolution group must be 1 or 2')
    # Add bias.
    for m in range(1, M):
        top[:, :, m] = top[:, :, m] + bias[1, 1, 1, m]
    return top


def relu(bottom):
    """
    ReLU Nonlinearity: Rectified Linear Units.
    Formula: top=max(0,bottom).
    """
    top = max(0, bottom)
    return top


def maxpool(bottom, K, S):
    """
    Maxpool over a window of K*K.
    bottom is a 3d matrix: Win x Hin x N.
    top is a 3d matrix: Wout x Hout x N.
    The kernel size K and stride S are integers.
    Pool the input (bottom) with windows of size K and with the specified stride.
    No padding needed.
    """
    [Win, Hin, N] = bottom.shape
    Wout = (Win - K) / S + 1
    Hout = (Hin - K) / S + 1
    top = np.zeros(Wout, Hout, N)
    for n in range(1, N):
        for h in range(1, Hout):
            for w in range(1, Wout):
                hstart = (h - 1) * S + 1
                wstart = (w - 1) * S + 1
                hend = hstart + K - 1
                wend = wstart + K - 1
                top[w, h, n] = max(max(bottom[range(wstart, wend), range(hstart, hend), n]))
    return top


def lrn(bottom, localSize, alpha, beta, k):
    """
    Local Response Normalization accross nearby channels.
    bottom is a 3d matrix: W x H x N.
    top is a 3d matrix: W x H x N.
    localSize, alpha, beta and k are integers.
    The output pixels depend only on pixels of nearby feature maps at the same position
    (same w/h coordinates).
    Formula:
    top_xy_i=bottom_xy_i/(k+alpha/localSize*sum_i(bottom_xy_i^2))^beta.
    The padding pixels consist of zeros.
    """
    W, H, N = bottom.shape
    top = np.zeros(W, H, N)
    for w in range(1, W):
        for h in range(1, H):
            for n in range(1, N):
                nStart = max(n - np.floor(localSize / 2), 1)
                nEnd = min(n + np.floor(localSize / 2), N)
                top[w, h, n] = bottom[w, h, n] / \
                               ((k + alpha / localSize * sum(np.array([w, h, range(nStart, nEnd)]) ** 2)) ** beta)
    return top


def fc(bottom, weight, bias):
    """
    Fully Connected Layer.
    bottom is a 2d matrix: N x 1.
    top is a 2d matrix: M x 1.
    weight is a 4d matrix: 1 x 1 x N x M.
    bias is a 4d matrix: 1 x 1 x 1 x M.
    Formula: top=weights'*bottom+bias.
    """
    N, M = weight.shape[2:]
    weightFlattened = np.reshape(weight, [N, M])
    biasFlattened = np.reshape(bias, [M, 1])
    top = np.transpose(weightFlattened) * bottom + biasFlattened
    return top


def dropout(bottom):
    """
    Dropout does not do anything in test phase.
    """
    return bottom


# Return an image that serves as input to AlexNet.
# imageData is the KLAB grayscale image data
# preprocessed_image is a 3d matrix with dimensions 227x227x3.
# preprocessed_image is WxHxC major in BGR.
def prepareGrayscaleImage(imageData, imagesMean):
    IMAGE_DIM = 256
    CROPPED_DIM = 227
    # Convert an image returned by Matlab's imread to im_data in caffe's data
    # format: W x H x C with BGR channelsM = 227
    rgbImage = np.tile(imageData, (1, 1, 3))
    imageData = rgbImage[:, :, [3, 2, 1]]  # permute channels from RGB to BGR
    imageData = np.transpose(imageData, (2, 1, 3))  # flip width and height
    # imageData = single(imageData)  # convert from uint8 to single
    imageData = imresize(imageData, (IMAGE_DIM, IMAGE_DIM), 'bilinear')  # resize im_data
    imageData = imageData - imagesMean  # subtract mean_data (already in W x H x C, BGR)
    imageData = imresize(imageData, (CROPPED_DIM, CROPPED_DIM), 'bilinear')  # resize im_data
    preprocessedImage = np.zeros((CROPPED_DIM, CROPPED_DIM, 3), 'double')
    preprocessedImage[:, :, :] = imageData
    return preprocessedImage


## Preparation
# Load network parameters.
netParams = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'ressources/alexnetParams.mat'))
# obtained from https://drive.google.com/file/d/0B-VdpVMYRh-pQWV1RWt5NHNQNnc/view
weights = netParams['weights']
conv1Kernels = weights[0]['weights'][0]
conv1Bias = weights[0]['weights'][1]
conv2Kernels = weights[1]['weights'][0]
conv2Bias = weights[1]['weights'][1]
conv3Kernels = weights[2]['weights'][0]
conv3Bias = weights[2]['weights'][1]
conv4Kernels = weights[3]['weights'][0]
conv4Bias = weights[3]['weights'][1]
conv5Kernels = weights[4]['weights'][0]
conv5Bias = weights[4]['weights'][1]

fc6Weights = weights[5]['weights'][0]
fc6Bias = weights[5]['weights'][1]
fc7Weights = weights[6]['weights'][0]
fc7Bias = weights[6]['weights'][1]
fc8Weights = weights[7]['weights'][0]
fc8Bias = weights[7]['weights'][1]
# Prepare input image (data is WxHxC major in BGR)
img = scipy.misc.imread('ressources/cat.jpg')
data = prepareGrayscaleImage(img, None)

# pass image through network
print('Running AlexNet in forward path. This will take about half a minute')
# extract features
conv1 = conv(data, conv1Kernels, conv1Bias, 11, 4, 0, 1)
relu1 = relu(conv1)
pool1 = maxpool(relu1, 3, 2)
lrn1 = lrn(pool1, 5, .0001, 0.75, 1)
conv2 = conv(lrn1, conv2Kernels, conv2Bias, 5, 1, 2, 2)
relu2 = relu(conv2)
pool2 = maxpool(relu2, 3, 2)
norm2 = lrn(pool2, 5, .0001, 0.75, 1)
conv3 = conv(norm2, conv3Kernels, conv3Bias, 3, 1, 1, 1)
relu3 = relu(conv3)
conv4 = conv(relu3, conv4Kernels, conv4Bias, 3, 1, 1, 2)
relu4 = relu(conv4)
conv5 = conv(relu4, conv5Kernels, conv5Bias, 3, 1, 1, 2)
relu5 = relu(conv5)
pool5 = maxpool(relu5, 3, 2)
pool5_2d = np.reshape(pool5, [9216, 1])  # flatten data
# classify
fc6 = fc(pool5_2d, fc6Weights, fc6Bias)
relu6 = relu(fc6)
dropout6 = dropout(relu6)
fc7 = fc(dropout6, fc7Weights, fc7Bias)
relu7 = relu(fc7)
dropout7 = dropout(relu7)
fc8 = fc(dropout7, fc8Weights, fc8Bias)
