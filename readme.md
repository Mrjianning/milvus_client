## 向量数据库Milvus c++ sdk  安装使用说明

### 1、安装milvus sdk


### 2、安装grpc

- 安装版本与 milvus sdk 编译版本对应：1.49.x

```shell
git clone -b v1.49.x https://github.com/grpc/grpc
cd grpc
git submodule update --init
mkdir -p cmake/build
cd cmake/build
cmake -DCMAKE_INSTALL_PREFIX=/home/ljn/apps/project/test_project/milvus_client/libs/grpc_install  -DBUILD_SHARED_LIBS=ON ../..
make -j12
make install

# 安装到系统
rm -rf *
cmake -DBUILD_SHARED_LIBS=ON ../..
make -j12
make install
```




