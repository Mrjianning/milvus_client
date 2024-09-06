## 向量数据库Milvus c++ sdk  安装使用说明

### 一、milvus 安装

- docker-compose 安装

  ```shell
  wget https://github.com/milvus-io/milvus/releases/download/v2.4.10/milvus-standalone-docker-compose.yml -O docker-compose.yml
  
  sudo docker compose up -d
  ```

​	

#### milvus基础知识

##### （1）Milvus 支持的索引类型

1. **FLAT** (Brute-force search)：

   - **描述**：使用全局遍历进行精确搜索，适用于小规模数据集。

   - **优点**：可以返回最精确的结果，没有近似误差。

   - **缺点**：随着数据量增加，查询速度会变慢。

   - ##### **应用场景**：**适用于小数据集或者需要精确匹配的场景**。

2. **IVF_FLAT** (Inverted File with FLAT)：

   - **描述**：通过将向量聚类，创建反向索引，从而加速查询。
   - **优点**：比 FLAT 索引快，适用于中等规模数据集。
   - **缺点**：查询结果是近似的，可能会有轻微误差。
   - **应用场景**：**适用于中等规模数据集的近似搜索。**

3. **IVF_SQ8** (Inverted File with Scalar Quantization)：

   - **描述**：在 IVF 的基础上进一步压缩数据，加速查询并减少内存使用。
   - **优点**：相比 IVF_FLAT 更快，适用于大规模数据集。
   - **缺点**：压缩导致误差更大，查询结果会更不精确。
   - **应用场景**：**适用于大规模数据集的近似搜索，需要在速度和精度之间权衡。**

4. **IVF_PQ** (Inverted File with Product Quantization)：

   - **描述**：使用乘积量化方法进一步压缩数据，适合大规模数据集的快速搜索。
   - **优点**：可以显著减少内存使用和计算复杂度，加速搜索。
   - **缺点**：近似误差较大，结果可能不够精确。
   - **应用场景**：**适用于大规模数据集的近似搜索，适合在搜索速度和结果精度之间找到平衡。**

5. **HNSW** (Hierarchical Navigable Small World)：

   - **描述**：一种图搜索算法，适合高维向量检索，速度快。
   - **优点**：在高维向量检索中表现优异，速度和精度都较好。
   - **缺点**：构建索引的时间较长，内存使用较高。
   - **应用场景**：**适用于高维数据集，特别是对检索速度有较高要求的应用。**

6. **ANNOY** (Approximate Nearest Neighbors Oh Yeah)：

   - **描述**：用于快速的近似最近邻搜索。
   - **优点**：内存占用较低，查询速度快。
   - **缺点**：精度相对较低。
   - **应用场景**：**适用于需要在低资源条件下进行快速近似搜索的应用。**

7. **RHNSW_FLAT** (Reciprocal HNSW and Flat)：

   - **描述**：通过对 HNSW 的扩展提供更高效的搜索性能。
   - **优点**：在非常高维和大规模数据集上表现良好。
   - **应用场景**：**适合对高维数据进行大规模搜索的场景。**

8. **RHNSW_PQ** (Reciprocal HNSW and Product Quantization)：

   - **描述**：HNSW 与乘积量化的结合，减少内存使用并加速大规模高维向量的搜索。
   - **优点**：在大规模数据集上提供高效的搜索性能，内存占用较低。
   - **缺点**：精度稍有下降，适合对搜索速度要求较高的场景。
   - **应用场景**：**适用于大规模高维数据集的快速近似搜索。**

9. **RHNSW_SQ** (Reciprocal HNSW and Scalar Quantization)：

   - **描述**：HNSW 与标量量化结合，在加速搜索的同时进一步减少内存使用。
   - **优点**：在维持快速搜索的同时减少了存储空间需求。
   - **缺点**：近似误差更大，适合对内存占用有严格要求的场景。
   - **应用场景**：**适用于需要在内存和性能之间平衡的大规模高维数据集。**

选择合适的索引：

- **FLAT**：适用于小规模精确搜索。
- **IVF_FLAT**：适合中等规模近似搜索。
- **IVF_SQ8**：适合大规模数据集的快速近似搜索。
- **IVF_PQ**：大规模数据集的快速近似搜索，适合内存敏感的场景。
- **HNSW**：适用于高维数据的快速、高精度搜索。
- **ANNOY**：适合在低资源环境下进行快速近似搜索。
- **RHNSW_FLAT**：适合超大规模和高维度数据的高效搜索。
- **RHNSW_PQ**：在大规模高维数据中平衡速度与内存使用的高效搜索。
- **RHNSW_SQ**：适合对内存占用有较高要求的大规模数据搜索。



##### （2）Milvus 支持的度量方式：

Milvus 支持多种度量方式（Metric Types），用于计算向量之间的距离或相似性。每种度量方式适用于不同类型的数据和应用场景。以下是 Milvus 支持的主要度量方式

1. **L2 (Euclidean Distance)**：
   - **描述**：L2 距离，也叫欧几里得距离，是向量空间中两点之间的直线距离。
   - **应用场景**：适合大多数需要进行相似性搜索的场景，尤其是低维和中维度的数据。
2. **IP (Inner Product / Cosine Similarity)**：
   - **描述**：内积度量是通过计算两个向量之间的点积来判断相似度，值越大表示越相似。通过单位向量化，这种度量方式还可以表示余弦相似度。
   - **应用场景**：适合需要通过角度来衡量相似性的场景，比如推荐系统。
     （负的内积表示最大相似性）
3. **HAMMING (Hamming Distance)**：
   - **描述**：汉明距离用于计算两个向量在同一位置上不同元素的个数。
   - **应用场景**：通常用于二进制向量或者字符匹配的场景，比如 DNA 序列分析。
4. **JACCARD (Jaccard Similarity)**：
   - **描述**：用于计算两个集合的交集与并集的比例，主要适用于离散值或集合数据的比较。
   - **应用场景**：用于集合相似度的计算，比如推荐系统、文本相似度。
5. **TANIMOTO (Tanimoto Similarity)**：
   - **描述**：是 Jaccard 相似性的推广，适用于二值或者连续向量的相似度比较。
   - **应用场景**：通常用于化学分子结构相似度或者二进制数据的比较。
6. **SUPERSTRUCTURE**：
   - **描述**：主要用于化学分子结构的比对，用于计算分子超结构的相似性。
   - **应用场景**：化学分子结构比较的专业领域。
7. **SUBSTRUCTURE**：
   - **描述**：与 SUPERSTRUCTURE 类似，但侧重于计算分子的子结构相似性。
   - **应用场景**：化学分子子结构比较。

选择合适的度量方式：

- **L2（欧几里得距离）** 是常用的标准度量，适合大多数向量相似性搜索任务。
- **IP（内积/余弦相似度）** 适用于通过角度衡量向量相似度的场景。
- **HAMMING** 适合离散数据、二进制向量或者字符串比较。
- **JACCARD 和 TANIMOTO** 用于集合相似度计算，通常用于离散型或化学数据分析。
- **SUPERSTRUCTURE 和 SUBSTRUCTURE** 主要用于化学领域。

这些度量方式提供了灵活性，可以根据数据类型和应用场景选择合适的度量方法，从而优化向量搜索的性能和准确性。



##### （3）Milvus 中分区的作用

在 Milvus 中，**分区**（Partition）是一种用于将集合中的数据进行逻辑划分的机制。分区的主要目的是通过对数据进行分类管理，提高数据的查询效率和数据管理的灵活性。

1. **数据的逻辑分隔**：
   - 分区允许你按照某个逻辑规则（例如日期、地理位置、类型等）将集合中的数据进行分隔。例如，你可以为每一年、每个月或某个类别的数据创建单独的分区。
   - 通过这种方式，数据的存储更加有序且易于管理。

2. **提高查询效率**：
   - 如果数据量非常大，Milvus 中的搜索操作可以通过限制在某些分区内进行，减少需要搜索的数据范围。这样可以大大提高查询效率，避免在整个集合中进行全局搜索。
   - 例如，你可以在查询时指定只在某几个分区中进行搜索，而不是在整个集合中搜索。

3. **数据生命周期管理**：
   - 分区支持单独加载、释放或删除。这对于大规模数据集的生命周期管理非常有用，比如只需要加载最近的数据分区，而历史数据的分区可以释放以节省内存。
   - 例如，分区可以按照时间段进行划分，可以定期删除或归档某些时间段的分区，以方便数据维护。

4. **灵活的数据导入与存储**：
   - 分区支持数据的分批导入。当你往集合中插入数据时，可以指定将数据插入到某个特定分区，帮助你更好地管理和组织数据。
   - 通过分区，你可以轻松地将不同类别的数据存储在不同的分区中，从而保持数据的逻辑一致性。



### 二、安装milvus sdk

- 下载sdk源码：

```shell
git clone git@github.com:milvus-io/milvus-sdk-cpp.git
```

- 编译
  - DCMAKE_INSTALL_PREFIX ： 指定安装目录
  -  -DBUILD_SHARED_LIBS ： 编译共享库

```sh
rm -rf build
mkdir build

cd build
cmake .. -DMILVUS_WITH_GTEST=ON -DCMAKE_INSTALL_PREFIX=~/milvus_install_dir -DBUILD_SHARED_LIBS=YES
make -j12 
make install  
```


### 三、安装grpc

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

### 四、测试

- 目录结构

  ```sh
  milvus_client/
  │
  ├── CMakeLists.txt                     # CMake 配置文件
  ├── main.cpp                           # 主程序源文件
  ├── libs/                              # 库文件夹
  │   └── grpc_install/                  # gRPC 安装文件夹
  │       ├── include/                   # gRPC 和 Protobuf 头文件
  │       └── lib/                       # gRPC 和 Protobuf 库文件
  │   └── milvus_install_dir/            # Milvus SDK 安装目录
  │       ├── include/                  
  │       └── lib/                      
  └── build/                             # 构建文件夹（由 CMake 创建）
  ```

- 编译运行

  ```sh
  mkdir build 
  cd build
  cmake ..
  make -j12 
  ./milvus_example
  ```

  ```sh
  示例开始...
  已连接到 Milvus 服务器。
  成功创建集合。
  成功创建索引。
  成功创建分区。
  成功插入 1000 行数据。
  分区 Year_2022 的行数: 1000
  正在搜索第 393 个实体...
  搜索成功。
  ID: 885 距离: 0.0212743 年龄: 44
  ID: 712 距离: 0.0354251 年龄: 96
  ID: 106 距离: 0.0457776 年龄: 78
  ID: 716 距离: 0.0496291 年龄: 49
  ID: 304 距离: 0.0518493 年龄: 58
  ID: 674 距离: 0.0534134 年龄: 50
  ID: 353 距离: 0.0549357 年龄: 58
  ID: 571 距离: 0.0559261 年龄: 90
  ID: 595 距离: 0.0640901 年龄: 69
  ID: 725 距离: 0.0650012 年龄: 69
  删除分区 Year_2022
  集合 TEST 的行数: 1000
  删除集合 TEST
  ```



