default:
	sudo python setup.py build -b /tmp/shanepy install --record /tmp/files.txt
	sudo python3 setup.py build -b /tmp/shanepy install --record /tmp/files.txt
	sudo python3.6 setup.py build -b /tmp/shanepy install --record /tmp/files.txt
	. ~/sh-source/conda-init
	sudo chown shane:shane /tmp/files.txt
	python setup.py build -b /tmp/shanepy install --record /tmp/files.txt