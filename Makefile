lint:
	isort .
	black .
	ruff check .

check-scripts:
	shellcheck *.sh

export:
	# remove prefix and mac specific packages for cross platform compatibility
	conda env export --from-history | grep -v -e '^prefix: ' -e 'pyobjc' > environment.yml

export-with-version:
	conda env export --no-build | grep -v '^prefix' > environment.yml

environment:
	conda env create -f environment.yml
