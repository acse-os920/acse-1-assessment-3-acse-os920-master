import sys
import os

## We're working in the ./docs directory, but need the package root in the path
## This command appends the directory one level up, in a cross-platform way. 
sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, '..'))))

latex_elements = {"extraclassoptions":"openany"}

project = 'ACSE_la'
author = "Orkan Sezer"
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build']
autoclass_content = "both"
