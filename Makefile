all: build

build:
	jupyter-book build neurocomputing

export:
	git commit -a -m "Rebuilding site `date`"
	git push origin master
	ghp-import -n -p -f neurocomputing/_build/html
	rsync -avze ssh --progress --delete ./neurocomputing/_build/html/ vitay@login.tu-chemnitz.de:/afs/tu-chemnitz.de/www/root/informatik/KI/edu/neurocomputing/notes/

clean:
	rm -rf ./neurocomputing/_build/html
