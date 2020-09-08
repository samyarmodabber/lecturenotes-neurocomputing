#!/bin/bash
jupyter-book build neurocomputing
msg="Rebuilding site `date`"
git commit -a -m "$msg"
git push origin master
ghp-import -n -p -f neurocomputing/_build/html