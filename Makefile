all: build

build:
	touch neurocomputing/zreferences.md
	jupyter-book build neurocomputing

export: build
	git add neurocomputing/_build/html
	git commit -a -m "Rebuilding site `date`"
	git push origin master
	ghp-import -n -p -f neurocomputing/_build/html
	scp .htaccess neurocomputing/_build/html/
	rsync -avze ssh --progress --delete ./neurocomputing/_build/html/ vitay@login.tu-chemnitz.de:/afs/tu-chemnitz.de/www/root/informatik/KI/edu/neurocomputing/notes/

clean:
	rm -rf ./neurocomputing/_build/html