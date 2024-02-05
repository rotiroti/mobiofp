.PHONY: help dev-install dev-uninstall macos-dev-install macos-dev-uninstall clean format

help:
	@echo "dev-install - install the package in editable mode"
	@echo "dev-uninstall - uninstall the package"
	@echo "macos-dev-install - install the package in editable mode with tensorflow-metal"
	@echo "macos-dev-uninstall - uninstall the package and tensorflow-metal"
	@echo "clean - remove all build and Python artifacts"

dev-install:
	pip install --editable .

macos-dev-install:
	pip install --editable .
	pip install tensorflow-metal

dev-uninstall:
	pip uninstall -y mobiofp
	pip uninstall -y -r requirements.txt

macos-dev-uninstall:
	pip uninstall -y mobiofp
	pip uninstall -y tensorflow-metal
	pip uninstall -y -r requirements.txt

format:
	python -m nbstripout **/*.ipynb
	python -m black --target-version py39 **/*.py **/*.ipynb
	python -m isort **/*.py **/*.ipynb

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf *.egg-info
