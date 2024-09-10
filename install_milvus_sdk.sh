git clone git@github.com:milvus-io/milvus-sdk-cpp.git
cd milvus-sdk-cpp

rm -rf build
mkdir build

cd build
cmake .. -DMILVUS_WITH_GTEST=ON -DCMAKE_INSTALL_PREFIX=~/milvus_install_dir -DBUILD_SHARED_LIBS=YES
make -j12 
make install  