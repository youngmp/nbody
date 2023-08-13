Last run on Ubuntu 20.04.6 LTS
pdoc3 0.10.0
pandoc 2.5
pdoc3 and pandoc installed using pip3

To create docs, cd to the root directory and run the command

$ pdoc3 --pdf --template-dir ./docs/ nBodyCoupling Thalamic CGL > ./docs/docstrings.md

Note the $ just indicates a terminal command so only run the command(s) to its right. This command reads all the docstrings from the files nBodyCoupling.py, Thalamic.py, and CGL.py and puts them into markdown format in /docs/docstrings.md

To combine the README.md and /docs/docstrings.md files into a single .tex file, run the command from the root directory (you can run this command anywhere, just modify it as needed)

$ pandoc --from=markdown+abbreviations+tex_math_single_backslash --toc --toc-depth=4 --output=./docs/docs.tex -t latex -s ./README.md ./docs/docstrings.md 

-s flag enables standalone, so it generates the .tex file that can be compiled directly using PDFLaTeX.

README.md contains custom information generated outside of docstrings (introduction, recommended versions).

It may be convenient to combine the above commands with a double ampersand from the root directory:

$ pdoc3 --pdf --template-dir ./docs/ nBodyCoupling Thalamic CGL > ./docs/docstrings.md && pandoc --from=markdown+abbreviations+tex_math_single_backslash --toc --toc-depth=4 --output=./docs/docs.tex -t latex -s ./README.md ./docs/docstrings.md
