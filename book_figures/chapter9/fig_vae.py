"""
Variational AutoEncoder
-----------------------
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
if "setup_text_plots" not in globals():
    from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=12, usetex=True)

fig = plt.figure(figsize=(6, 4), facecolor='w')
ax = fig.add_axes([0, 0, 1, 1],
                  xticks=[], yticks=[])
plt.box(False)
circ = plt.Circle((1, 1), 2)

radius = 0.3


# function to draw arrows
def draw_connecting_arrow(ax, circ1, rad1, circ2, rad2,
                          arrow_kwargs={'head_width': 0.05, 'fc': 'black',
                                        'alpha': 0.5}):
    theta = np.arctan2(circ2[1] - circ1[1],
                       circ2[0] - circ1[0])

    starting_point = (circ1[0] + rad1 * np.cos(theta),
                      circ1[1] + rad1 * np.sin(theta))

    length = (circ2[0] - circ1[0] - (rad1 + 1.4 * rad2) * np.cos(theta),
              circ2[1] - circ1[1] - (rad1 + 1.4 * rad2) * np.sin(theta))

    ax.arrow(starting_point[0], starting_point[1],
             length[0], length[1], **arrow_kwargs)


# function to draw circles
def draw_circle(ax, center, radius, **kwargs):
    circ = plt.Circle(center, radius, fc='none', lw=2, **kwargs)
    ax.add_patch(circ)


x1 = -3.4
x2 = -2
x3 = -0.5
x4 = 0.5
x5 = 2
x6 = 3.5

seq1 = np.linspace(2.5, -2, 4)
seq2 = np.linspace(1.75, -1.5, 3)
seq3 = np.hstack([np.linspace(2.5, 1, 3), np.linspace(-0.5, -2, 3)])
seq4 = np.linspace(1, -0.5, 3)
seq5 = np.linspace(1.75, -1.5, 3)
seq6 = np.linspace(2.5, -2, 4)


# ------------------------------------------------------------
# draw circles
for i, y1 in enumerate(seq1):
    draw_circle(ax, (x1, y1), radius)

for i, y2 in enumerate(seq2):
    draw_circle(ax, (x2, y2), radius)

for i, y3 in enumerate(seq3):
    draw_circle(ax, (x3, y3), radius * 0.75)

ax.add_patch(plt.Rectangle((x3-radius, seq3[2]-radius), radius * 2,
                           seq3[0]-seq3[2] + 2 * radius, fc='b', alpha=0.5))
ax.text(x3, seq3[1], r'$\sigma$', fontsize=12, ha='center', va='center')

ax.add_patch(plt.Rectangle((x3-radius, seq3[5]-radius), radius * 2,
                           seq3[3]-seq3[5] + 2 * radius, fc='y', alpha=0.5))
ax.text(x3, seq3[4], r'$\mu$', fontsize=12, ha='center', va='center')

draw_connecting_arrow(ax, (x3 + radius, seq3[1]), radius * 0.15,
                      (x4 - radius, seq4[1]), radius * 0.2)
draw_connecting_arrow(ax, (x3 + radius, seq3[4]), radius * 0.15,
                      (x4 - radius, seq4[1]), radius * 0.2)

for i, y4 in enumerate(seq4):
    draw_circle(ax, (x4, y4), radius * 0.75, alpha=0.5)

ax.text(x4, seq4[1], 'Sample', fontsize=12, ha='center', va='center', rotation=90)

ax.add_patch(plt.Rectangle((x4-radius, seq4[2]-radius), radius * 2,
                           seq4[0]-seq4[2] + 2 * radius, fc='g', alpha=0.5))

for i, y5 in enumerate(seq5):
    draw_circle(ax, (x5, y5), radius)

for i, y6 in enumerate(seq6):
    draw_circle(ax, (x6, y6), radius)

# ------------------------------------------------------------
# draw connecting arrows
for i, y1 in enumerate(seq1):
    for j, y2 in enumerate(seq2):
        draw_connecting_arrow(ax, (x1, y1), radius, (x2, y2), radius)

for i, y2 in enumerate(seq2):
    for j, y3 in enumerate(seq3):
        draw_connecting_arrow(ax, (x2, y2), radius, (x3, y3), radius * 0.8)


for i, y4 in enumerate(seq4):
    for j, y5 in enumerate(seq5):
        draw_connecting_arrow(ax, (x4, y4), radius * 0.75, (x5, y5), radius)

for i, y5 in enumerate(seq5):
    for j, y6 in enumerate(seq6):
        draw_connecting_arrow(ax, (x5, y5), radius, (x6, y6), radius)


# ------------------------------------------------------------
# Add axis

ax.arrow(-3.8, -2.6, 0, 5, head_width=0.05)
ax.arrow(-3.8, -2.6, 7.5, 0, head_width=0.05)
plt.text(0, -2.8, "Latent Space 1", ha='center', va='center')
plt.text(-3.9, 0, "Latent Space 2", ha='center', va='center', rotation=90)


ax.set_aspect('equal')
plt.xlim(-4, 4)
plt.ylim(-3, 3)
plt.show()
