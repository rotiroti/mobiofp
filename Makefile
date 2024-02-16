.PHONY: help dev-install dev-uninstall macos-dev-install macos-dev-uninstall clean install macos-install uninstall macos-uninstall

help:
	@echo "--------------------------------------"
	@echo "MoBioFP - Mobile Biometric Fingerphoto"
	@echo "--------------------------------------"
	@echo ""
	@printf "%-25s %-50s\n" "Task" "Description"
	@echo "--------------------------------------"
	@printf "%-25s %-50s\n" "dev-install" "install the package in editable mode"
	@printf "%-25s %-50s\n" "macos-dev-install" "install the package in editable mode with tensorflow-metal"
	@printf "%-25s %-50s\n" "dev-uninstall" "uninstall the package"
	@printf "%-25s %-50s\n" "macos-dev-uninstall" "uninstall the package and tensorflow-metal"
	@echo ""
	@printf "%-25s %-50s\n" "install" "install the package"
	@printf "%-25s %-50s\n" "macos-install" "install the package with tensorflow-metal"
	@printf "%-25s %-50s\n" "uninstall" "uninstall the package"
	@printf "%-25s %-50s\n" "macos-uninstall" "uninstall the package and tensorflow-metal"
	@echo ""
	@printf "%-25s %-50s\n" "clean" "remove all build and Python artifacts"
	@echo "--------------------------------------"

dev-install:
	pip install --editable .

dev-uninstall:
	pip uninstall -y mobiofp
	pip uninstall -y -r requirements.txt
	@echo "\nPleae run 'pip uninstall -y -r <(pip freeze)' to remove ALL PACKAGES installed by pip in the current environment."

macos-dev-install:
	pip install --editable .
	pip install tensorflow-metal

macos-dev-uninstall:
	pip uninstall -y mobiofp
	pip uninstall -y tensorflow-metal
	pip uninstall -y -r requirements.txt
	@echo "\nPleae run 'pip uninstall -y -r <(pip freeze)' to remove ALL PACKAGES installed by pip in the current environment."

install:
	pip install .

macos-install:
	pip install .
	pip install tensorflow-metal

uninstall:
	pip uninstall -y mobiofp
	pip uninstall -y -r requirements.txt
	@echo "\nPleae run 'pip uninstall -y -r <(pip freeze)' to remove ALL PACKAGES installed by pip in the current environment."

macos-uninstall:
	pip uninstall -y mobiofp
	pip uninstall -y tensorflow-metal
	pip uninstall -y -r requirements.txt
	@echo "\nPleae run 'pip uninstall -y -r <(pip freeze)' to remove ALL PACKAGES installed by pip in the current environment."

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf *.egg-info
	rm -fr cache
	rm -fr notebooks/.ipynb_checkpoints
	rm -fr notebooks/cache
