"""
Convolutional Neural Network
----------------------------
"""
# Author: Brigitta Sipocz
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2019)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general

import numpy as np
from matplotlib import pyplot as plt

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)

fig = plt.figure(figsize=(6, 4), facecolor='w')

# ------------------------------------------------------------
# Get the galaxy image
#
# TODO: use astroquery once it supports SDSS ImageCutout
# http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx?ra=202.469575&dec=47.1952583&scale=2&width=500&height=500&
#

m51 = plt.imread('m51.jpeg')
ax1 = fig.add_axes((0, 0.4, 0.25, 0.25))
ax1.set_axis_off()
ax1.imshow(m51)

# ------------------------------------------------------------
# CNN cartoon

ax = fig.add_axes([0, 0, 1, 1],
                  xticks=[], yticks=[])
plt.box(False)
circ = plt.Circle((1, 1), 2)

radius = 0.15


# function to draw arrows
def draw_connecting_arrow(ax, circ1, rad1, circ2, rad2, fc='grey', **kwargs):
    theta = np.arctan2(circ2[1] - circ1[1],
                       circ2[0] - circ1[0])

    starting_point = (circ1[0] + rad1 * np.cos(theta),
                      circ1[1] + rad1 * np.sin(theta))

    length = (circ2[0] - circ1[0] - (rad1 + rad2) * np.cos(theta),
              circ2[1] - circ1[1] - (rad1 + rad2) * np.sin(theta))

    ax.arrow(starting_point[0], starting_point[1],
             length[0], length[1], fc=fc, linestyle=':', **kwargs)


# function to draw circles
def draw_circle(ax, center, radius):
    circ = plt.Circle(center, radius, fc='none', lw=2)
    ax.add_patch(circ)


# function to squares circles
def draw_squares(ax, center, size, num, shift=(0.25, -0.25),
                 line=None, **kwargs):
    ec = kwargs.pop('ec', 'black')
    lw = kwargs.pop('lw', 2)
    fc = kwargs.pop('fc', 'white')

    shift = np.array(shift)
    back_left_bottom = np.array(center) - num // 2 * shift - size / 2

    back_right_top = np.array(center) - num // 2 * shift + size / 2
    front_right_bottom = np.array(center) + num // 2 * shift + np.array([size, - size]) / 2

    for i in range(num):
        rec = plt.Rectangle(back_left_bottom + i * shift, size, size,
                            fc=fc, ec=ec, lw=lw, **kwargs)
        ax.add_patch(rec)

    if line:
        node1, node2, radius = line
        draw_connecting_arrow(ax, back_right_top, 0, node1, radius)
        draw_connecting_arrow(ax, front_right_bottom, 0, node2, radius)

        # To draw more lines to the fully connected layer
        if num > 1:
            draw_connecting_arrow(ax, back_right_top, 0, node2, radius)
            draw_connecting_arrow(ax, front_right_bottom, 0, node1, radius)

    return (back_right_top, front_right_bottom)


x1 = -3.1
x2 = -1
x3 = 1
x4 = 3
x5 = 3.8
seq1 = np.linspace(1.8, -1, 4)
seq2 = np.linspace(1, 0, 2)

# ------------------------------------------------------------
# convolution layer

_, first_layer = draw_squares(ax, (x2, 0), 0.9, 9)
_, second_layer = draw_squares(ax, (x3, 0), 0.8, 5, line=((x4, seq1[0]), (x4, seq1[-1]), radius))

# ------------------------------------------------------------
# convolution layer connections

elem1 = draw_squares(ax, (x1, 0), 0.2, 1, ec='white', fc='none', lw=1)

draw_squares(ax, first_layer + [-0.5, 0.6], 0.3, 1, line=(*elem1, 0.2),
             ec='black', fc='none', lw=1)

elem2 = draw_squares(ax, first_layer + [-0.3, 0.2], 0.2, 1,
                     ec='grey', fc='grey', lw=1)

draw_squares(ax, second_layer + [-0.15, 0.5], 0.15, 1, line=(*elem2, 0),
             ec='grey', fc='grey', lw=1)

# ------------------------------------------------------------
# fully connected layer
#
# draw circles
for i, y4 in enumerate(seq1):
    draw_circle(ax, (x4, y4), radius)

for i, y5 in enumerate(seq2):
    draw_circle(ax, (x5, y5), radius)

# draw connecting arrows
for i, y4 in enumerate(seq1):
    for j, y5 in enumerate(seq2):
        draw_connecting_arrow(ax, (x4, y4), radius, (x5, y5), radius)

# ------------------------------------------------------------
# Add text labels

plt.text(x1, -2.2, 'Input Image', ha='center')
plt.text(x2, -2.2, 'Convolution Layer', ha='center')
plt.text(x3, -2.2, 'Max-pooling', ha='center')
plt.text((x4 + x5)/2, -2.2, 'Fully Connected Layer', ha='center')

ax.set_aspect('equal')
plt.xlim(-4, 4)
plt.ylim(-3, 3)
plt.show()
