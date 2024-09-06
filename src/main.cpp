#include "MilvusClient.h"
#include <random>

int main() {
    // 连接数据库
    MilvusClient milvus_client("localhost", 19530);

    // 定义参数 
    const std::string collection_name = "TEST";         // 集合名称
    const std::string partition_name = "Year_2022";     // 分区名称
    const uint32_t dimension = 4;                       // 向量维度
    const int64_t row_count = 1000;                     // 向量条数

    // 1、创建集合
    // 1.1 构建字段
    const std::string field_id_name = "identity";
    const std::string field_age_name = "age";
    const std::string field_face_name = "face";
    // 1.2 构建集合架构
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id_name, milvus::DataType::INT64, "用户 ID", true, false});
    collection_schema.AddField({field_age_name, milvus::DataType::INT8, "用户年龄"});
    collection_schema.AddField(milvus::FieldSchema(field_face_name, milvus::DataType::FLOAT_VECTOR, "脸部特征").WithDimension(dimension));

    // 1.2 创建集合
    milvus_client.CreateCollection(collection_schema);

    // 2、创建索引
    milvus_client.CreateIndex(collection_name, "face");

    // 3、创建分区
    milvus_client.CreatePartition(collection_name, partition_name);

    // 4、插入数据
    // 4.1 构建插入数据
    std::vector<int64_t> insert_ids;
    std::vector<int8_t> insert_ages;
    std::vector<std::vector<float>> insert_vectors;

    // 随机生成数据
    std::default_random_engine ran_;
    std::uniform_int_distribution<int> int_gen(1, 100);
    std::uniform_real_distribution<float> float_gen(0.0, 1.0);
    for (auto i = 0; i < row_count; ++i) {
        insert_ids.push_back(i);
        insert_ages.push_back(int_gen(ran_));
        std::vector<float> vector(dimension);
        for (auto i = 0; i < dimension; ++i) {
            vector[i] = float_gen(ran_);
        }
        insert_vectors.emplace_back(vector);
    }
    std::vector<milvus::FieldDataPtr> fields_data{
        std::make_shared<milvus::Int64FieldData>("identity", insert_ids),
        std::make_shared<milvus::Int8FieldData>("age", insert_ages),
        std::make_shared<milvus::FloatVecFieldData>("face", insert_vectors)
    };
    // 4.2 插入数据
    milvus_client.InsertData(collection_name, partition_name, fields_data);

    // 5、查询数据
    // 5.1 随机选择一个向量进行查询
    std::uniform_int_distribution<int64_t> int64_gen(0, 999);
    int64_t q_number = int64_gen(ran_);
    std::vector<float> q_vector(dimension);
    for (auto& val : q_vector) {
        val = static_cast<float>(rand()) / RAND_MAX;
    }
    // 5.2 查询数据
    std::cout << std::endl;
    std::cout << "正在搜索第 " << q_number << " 个实体..." << std::endl;
    milvus_client.SearchData(collection_name, partition_name, "face", "age", q_vector);

    // 6、删除分区
    milvus_client.GetPartitionStatistics(collection_name, partition_name);
    milvus_client.DeletePartition(collection_name, partition_name);
    milvus_client.GetPartitionStatistics(collection_name, partition_name);

    // 7、删除集合
    milvus_client.GetCollectionStatistics(collection_name);
    milvus_client.DropCollection(collection_name);
    milvus_client.GetCollectionStatistics(collection_name);

    return 0;
}
