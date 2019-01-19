"""
Compare Image Tests
-------------------

This script compares all the mis-matching images found when running

$ nosetests astroML_fig_tests

The result of running this script is an html page comparing each output file
to the baseline result, showing only the ones with a mismatch above the
threshold specified in astroML_fig_tests.
"""


import os

TABLE = """
<html>
<table>
  {rows}
</table>
</html>
"""

ROW = """
<tr>
  <td align="center">Difference {0}</td>
  <td align="center">actual</td>
  <td align="center">baseline</td>
</tr>
<tr>
  <td><img src="{1}" width="100%"></td>
  <td><img src="{2}" width="100%"></td>
  <td><img src="{3}" width="100%"></td>
</tr>
"""

baseline = "astroML_fig_tests/baseline/book_figures"
results = "astroML_fig_tests/results/book_figures"

figlist = []

for chapter in os.listdir(baseline):
    if not os.path.isdir(os.path.join(results, chapter)):
        print("Results for {} doesn't exist.".format(chapter))
        continue
    for pyfile in os.listdir(os.path.join(baseline, chapter)):
        if os.path.join(results, chapter, os.path.basename(pyfile)[:-4]+"-failed-diff.png"):
            root = pyfile.split('.png')[0]
            figlist.append((os.path.join("book_figures", chapter, root),
                            os.path.join(results, chapter, os.path.basename(root)+"-failed-diff.png"),
                            os.path.join(results, chapter, root + '.png'),
                            os.path.join(baseline, chapter, root + '.png')))
        elif not os.path.exists(os.path.join(results, chapter, os.path.basename(pyfile))):
            figlist.append((os.path.join(pyfile.split('.png')),
                            '',
                            '',
                            pyfile))


outfile = "_compare_images.html"

with open(outfile, 'w') as f:
    f.write(TABLE.format(rows = '\n'.join([ROW.format(*figs, width="90%")
                                           for figs in figlist])))

import webbrowser
webbrowser.open_new("file://localhost" + os.path.abspath(outfile))
