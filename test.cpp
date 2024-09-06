#include <iostream>
#include <random>
#include <string>

#include "milvus/MilvusClient.h"
#include "milvus/types/CollectionSchema.h"

// 检查 Milvus 操作的状态，如果有错误则打印并退出
void CheckStatus(std::string&& prefix, const milvus::Status& status) {
    if (!status.IsOk()) {
        std::cout << prefix << " " << status.Message() << std::endl;
        exit(1);
    }
}

// 计算时间差
template <typename Func>
void MeasureTime(const std::string& task_name, Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << task_name << " 耗时: " << duration.count() *1000 << " 毫秒" << std::endl;
}

int main(int argc, char* argv[]) {
    printf("示例开始...\n");

    // 创建 Milvus 客户端
    auto client = milvus::MilvusClient::Create();
    
    // 设置连接参数（本地服务器地址和端口）
    milvus::ConnectParam connect_param{"localhost", 19530};
    auto status = client->Connect(connect_param);
    CheckStatus("连接 Milvus 服务器失败:", status);
    std::cout << "已连接到 Milvus 服务器。" << std::endl;

    // 如果集合已存在，则删除集合
    const std::string collection_name = "TEST";
    status = client->DropCollection(collection_name);

    // 创建一个集合
    const std::string field_id_name = "identity";   // 用户 ID 字段
    const std::string field_age_name = "age";       // 用户年龄字段
    const std::string field_face_name = "face";     // 用户脸部特征字段
    const uint32_t dimension = 4;                   // 特征向量的维度
    milvus::CollectionSchema collection_schema(collection_name);
    // 添加字段到集合中
    collection_schema.AddField({field_id_name, milvus::DataType::INT64, "用户 ID", true, false});
    collection_schema.AddField({field_age_name, milvus::DataType::INT8, "用户年龄"});
    collection_schema.AddField(milvus::FieldSchema(field_face_name, milvus::DataType::FLOAT_VECTOR, "脸部特征").WithDimension(dimension));

    // 创建集合
    status = client->CreateCollection(collection_schema);
    CheckStatus("创建集合失败:", status);
    std::cout << "成功创建集合。" << std::endl;

    // 创建索引（2.2.0 版本后需要）
    milvus::IndexDesc index_desc(field_face_name, "", milvus::IndexType::FLAT, milvus::MetricType::L2, 0);
    status = client->CreateIndex(collection_name, index_desc);
    CheckStatus("创建索引失败:", status);
    std::cout << "成功创建索引。" << std::endl;

    // 创建分区
    std::string partition_name = "Year_2022";
    status = client->CreatePartition(collection_name, partition_name);
    CheckStatus("创建分区失败:", status);
    std::cout << "成功创建分区。" << std::endl;

    // 通知服务器准备加载集合
    status = client->LoadCollection(collection_name);
    CheckStatus("加载集合失败:", status);

    // 插入一些数据
    const int64_t row_count = 1000;
    std::vector<int64_t> insert_ids;              // 存储用户 ID
    std::vector<int8_t> insert_ages;              // 存储用户年龄
    std::vector<std::vector<float>> insert_vectors;  // 存储脸部特征向量
    std::default_random_engine ran(time(nullptr));   // 用于生成随机数
    std::uniform_int_distribution<int> int_gen(1, 100);
    std::uniform_real_distribution<float> float_gen(0.0, 1.0);
    for (auto i = 0; i < row_count; ++i) {
        insert_ids.push_back(i);                 // 随机生成用户 ID
        insert_ages.push_back(int_gen(ran));     // 随机生成用户年龄
        std::vector<float> vector(dimension);

        // 随机生成特征向量
        for (auto i = 0; i < dimension; ++i) {
            vector[i] = float_gen(ran);
        }
        insert_vectors.emplace_back(vector);
    }

    // 构建字段数据
    std::vector<milvus::FieldDataPtr> fields_data{
        std::make_shared<milvus::Int64FieldData>(field_id_name, insert_ids),
        std::make_shared<milvus::Int8FieldData>(field_age_name, insert_ages),
        std::make_shared<milvus::FloatVecFieldData>(field_face_name, insert_vectors)};
    milvus::DmlResults dml_results;

    // 插入数据到集合的分区
    status = client->Insert(collection_name, partition_name, fields_data, dml_results);
    CheckStatus("插入数据失败:", status);
    std::cout << "成功插入 " << dml_results.IdArray().IntIDArray().size() << " 行数据。" << std::endl;

    // 获取分区统计信息
    milvus::PartitionStat part_stat;
    status = client->GetPartitionStatistics(collection_name, partition_name, part_stat);
    CheckStatus("获取分区统计信息失败:", status);
    std::cout << "分区 " << partition_name << " 的行数: " << part_stat.RowCount() << std::endl;

    // 执行搜索
    milvus::SearchArguments arguments{};
    arguments.SetCollectionName(collection_name);
    arguments.AddPartitionName(partition_name);
    arguments.SetTopK(10);  // 返回前10个匹配结果
    arguments.AddOutputField(field_age_name);
    arguments.SetExpression(field_age_name + " > 40");  // 查询年龄大于40的用户
    // 设置强一致性保证，确保搜索在数据持久化后执行
    arguments.SetGuaranteeTimestamp(milvus::GuaranteeStrongTs());

    // 随机选择一个向量进行查询
    std::uniform_int_distribution<int64_t> int64_gen(0, row_count - 1);
    int64_t q_number = int64_gen(ran);
    std::vector<float> q_vector = insert_vectors[q_number];
    arguments.AddTargetVector(field_face_name, std::move(q_vector));
    std::cout << "正在搜索第 " << q_number << " 个实体..." << std::endl;

    milvus::SearchResults search_results{};
    status = client->Search(arguments, search_results);
    CheckStatus("搜索失败:", status);
    std::cout << "搜索成功。" << std::endl;

    // 输出搜索结果
    for (auto& result : search_results.Results()) {
        auto& ids = result.Ids().IntIDArray();
        auto& distances = result.Scores();
        if (ids.size() != distances.size()) {
            std::cout << "结果不合法！" << std::endl;
            continue;
        }

        // 输出搜索到的年龄字段
        auto age_field = result.OutputField(field_age_name);
        milvus::Int8FieldDataPtr age_field_ptr = std::static_pointer_cast<milvus::Int8FieldData>(age_field);
        auto& age_data = age_field_ptr->Data();

        // 遍历搜索结果
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << "ID: " << ids[i] << "\t距离: " << distances[i]
                      << "\t年龄: " << static_cast<int32_t>(age_data[i]) << std::endl;
            // 验证返回的年龄值是否与插入时的一致
            if (insert_ages[ids[i]] != age_data[i]) {
                std::cout << "错误！返回的值与插入的值不匹配" << std::endl;
            }
        }
    }

    // 删除分区
    status = client->DropPartition(collection_name, partition_name);
    CheckStatus("删除分区失败:", status);
    std::cout << "删除分区 " << partition_name << std::endl;

    // 验证集合中的行数应为0
    milvus::CollectionStat col_stat;
    status = client->GetCollectionStatistics(collection_name, col_stat);
    CheckStatus("获取集合统计信息失败:", status);
    std::cout << "集合 " << collection_name << " 的行数: " << col_stat.RowCount() << std::endl;

    // 删除集合
    status = client->DropCollection(collection_name);
    CheckStatus("删除集合失败:", status);
    std::cout << "删除集合 " << collection_name << std::endl;

    return 0;
}
