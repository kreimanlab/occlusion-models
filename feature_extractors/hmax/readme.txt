==============================================================================
Copyright 2005
Center for Biological & Computational Learning at MIT and MIT
All rights reserved.
Permission to copy and modify this data, software, and its documentation only
for internal research use in your organization is hereby granted, provided 
that this notice is retained thereon and on all copies. This data and software
should not be distributed to anyone outside of your organization without 
explicit written authorization by the author(s) and MIT. It should not be 
used for commercial purposes without specific permission from the authors 
and MIT. MIT also requires written authorization by the author(s) to publish
results obtained with the data or software and possibly citation of relevant 
CBCL reference papers.
We make no representation as to the suitability and operability of this data 
or software for any purpose. It is provided "as is" without express or implied
warranty.
==============================================================================


This directory contains a new implementation of the methods described in:

T. Serre, L. Wolf and T. Poggio. Object Recognition with Features 
Inspired by Visual Cortex. To Appear In: Proceedings of 2005 IEEE 
Computer Society Conference on Computer Vision and Pattern 
Recognition (CVPR 2005), IEEE Computer Society Press, San Diego, 
June 2005. 

A more complete AI Memo version is available at 
(ftp://publications.ai.mit.edu/ai-publications/2004/AIM-2004-026.pdf).

This new implementation is faster and easier to use than the original
one, and was tested to be almost fully compatible with it. Lior Wolf,
Stan Bileschi and Thomas Serre contributed to this implementation
which is based on the C implementation written by Thomas. Parts of
that C implementation were based on an earlier C code by Max
Riesenhuber,

BEFORE YOU RUN
==============

1. Set the directories for the positive and negative images. There are
four directories: train_set.pos, train_set.net, test_set.pos and
test_set.net. These are set at lines 18-21 of demoRelease.m

2. The input images have to be grayscale. Although the algorithm has
some invariance to scale, it might be a good idea to scale all images
to the same height. We use a height of 140 pixels in many of our
experiments.

3. If you would like to use an SVM classifier, please install OSU SVM
(http://www.ece.osu.edu/~maj/osu_svm/). Add the path of OSU SVM to the
second line of demoRelease. If you'd prefer a NN classifier set useSVM
(line 5) to zero.

4. If you'd like the algorithm to learn its own object-specific features
set READPATCHESFROMFILE (line 9) to zero (should give somewhat better
results on many datasets). If not, you can use the universal features
that are stored in PatchesFromNaturalImages250per4sizes.mat. This is done 
by setting READPATCHESFROMFILE to one. The stored prototypes are taken
from a set of "natural images" we collected.

Contents of the directory:

MAIN DEMO
=========
demoRelease.m - a demo showing a possible use of the code in this
directory. The demo reads a set of images, extracts the C2 features
for this set, and builds a classifier to learn the class of images.

MAIN STANDARD MODEL FUNCTIONS
=============================
C1.m - extracts s1 and c1 layers of the standard model representation
C2.m - extracts s2 and c2 layers of the standard model representation

CLASSIFICATION FUNCTIONS
========================
CLSnn.m   - Nearest Neighbor classifier train
CLSnnC.m  - Nearest Neighbor classifier test

CLSosusvm.m   - SVM train (a wrapper function for osusvm)
CLSosusvmC.m  - SVM test  (a wrapper function for osusvm)

UTILITY FUNCTIONS
=================
extractC2forcell.m      - extracts C2 for all images in a cell, using
all prototypes in another cell
extractRandC1Patches.m  - extracts random C1 patches to serve as prototypes
init_gabor.m            - creates Gabor filters
maxfilter.m             - local maximum in an image
padimage.m              - adds zeros around the boundary of the image
readAllImages.m         - reads all images in training and testing
directories into one cell
sumfilter.m             - local sums in an image
unpadimage.m            - reverses the effect of padimage
WindowedPatchDistance.m - scans an image looking for the best match
for a prototype (image fragment)

PRECOMPUTED UNIVERSAL FEATURES
==============================
PatchesFromNaturalImages250per4sizes.mat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BUG FIXES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Jul 01 2005 - changed bugs in extractC2forcell, extractRandC1Patches.m
