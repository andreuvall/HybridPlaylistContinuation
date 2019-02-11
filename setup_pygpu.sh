#!/usr/bin/env bash

# install pygpu requirements
source venv/bin/activate
pip install Cython==0.29.4 Mako==1.0.7 MarkupSafe==1.1.0
deactivate

# manually install pygpu into the virtual environment
cd venv
git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray
git reset --hard 23f30c27f590994e1d796bbe95e69d633c989bf3
git pull
mkdir Build
cd Build
cmake .. -DCMAKE_INSTALL_PREFIX=~/.local -DCMAKE_BUILD_TYPE=Release
make
make install
cd ..
source ../bin/activate
python setup.py build
python setup.py install
deactivate
cd ../..
