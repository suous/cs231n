lint:
	isort .
	black .
	ruff check .

check-scripts:
	shellcheck *.sh

export:
	# remove prefix and mac specific packages for cross platform compatibility
	micromamba env export --from-history | grep -v -e '^prefix: ' -e 'pyobjc' > environment.yml

export-with-version:
	micromamba env export --no-build | grep -v '^prefix' > environment.yml

export-detail:
	micromamba env export > environment-detail.yml

environment:
	micromamba env create -f environment-detail.yml
