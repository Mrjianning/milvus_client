cmake_minimum_required(VERSION 3.12)
project(MilvusExample)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${MILVUS_SDK_PATH}/lib;${PROJECT_SOURCE_DIR}/libs/grpc_install/lib")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN")

# 指定Milvus SDK库路径
set(MILVUS_SDK_PATH ${PROJECT_SOURCE_DIR}/libs/milvus_install_dir)

# 包含Milvus SDK头文件
include_directories(${MILVUS_SDK_PATH}/include
                    ${PROJECT_SOURCE_DIR}/libs/grpc_install/include    
                )

# 链接Milvus SDK库
link_directories(${MILVUS_SDK_PATH}/lib 
                ${PROJECT_SOURCE_DIR}/libs/grpc_install/lib)

# 设置可执行文件
add_executable(milvus_example src/main.cpp src/MilvusClient.cpp)

# 链接库文件
target_link_libraries(milvus_example 
        milvus_sdk 
        grpc++ 
        protobuf)
