#include "MilvusClient.h"
#include <chrono>

// 构造函数：初始化 Milvus 客户端并连接到服务器
MilvusClient::MilvusClient(const std::string& host, int port) {
    client_ = milvus::MilvusClient::Create();
    milvus::ConnectParam connect_param{host, port};
    auto status = client_->Connect(connect_param);
    CheckStatus("连接 Milvus 服务器失败:", status);
    std::cout << "已连接到 Milvus 服务器。" << std::endl;
}

// 析构函数：确保客户端正确关闭
MilvusClient::~MilvusClient() {
    client_->Disconnect();
}

// 检查状态并在失败时退出
void MilvusClient::CheckStatus(std::string&& prefix, const milvus::Status& status) {
    if (!status.IsOk()) {
        std::cout << prefix << " " << status.Message() << std::endl;
        exit(1);
    }
}

// 创建集合
void MilvusClient::CreateCollection(const milvus::CollectionSchema collection_schema) {
    auto status = client_->CreateCollection(collection_schema);
    CheckStatus("创建集合失败:", status);
    std::cout << "成功创建集合。" << std::endl;
}

/**
 * @brief 在指定的集合上创建索引，用于加速向量搜索。
 *
 * Milvus 支持多种索引类型，每种索引适用于不同的应用场景。创建索引的类型可以根据数据集的规模和查询需求进行选择。
 * 本函数默认使用 FLAT 索引类型，适合小规模数据集的精确搜索。
 *
 * @param collection_name 指定集合的名称。
 * @param field_face_name 指定需要创建索引的字段名称，通常是用于向量搜索的字段。
 */
void MilvusClient::CreateIndex(const std::string& collection_name, const std::string& field_face_name, 
                               milvus::IndexType index_type, milvus::MetricType metric_type) {
    // 使用传入的索引类型和度量方式来创建索引
    milvus::IndexDesc index_desc(field_face_name, "", index_type, metric_type, 0);
    
    // 创建索引
    auto status = client_->CreateIndex(collection_name, index_desc);
    
    // 检查创建索引的状态
    CheckStatus("创建索引失败:", status);
    std::cout << "成功创建索引。" << std::endl;
}


/**
 * @brief 在指定的集合中创建一个分区。
 *
 * Milvus 的集合可以通过分区来组织数据。分区的作用是将数据集分隔成多个逻辑段，便于管理和提高查询效率。
 * 
 * 主要用途：
 * 1. **逻辑分隔**：分区可以帮助将数据按一定的逻辑标准（例如时间、类别等）进行分类和存储。
 * 2. **提高查询效率**：当数据量非常大时，分区可以用来限制搜索范围，避免在整个集合中搜索，提高查询性能。
 * 3. **数据管理**：分区支持分批管理和操作，比如按时间段将数据存储到不同的分区中。
 *
 * 注意：一个集合可以包含多个分区，每个分区都独立存储数据，查询时可以指定查询哪些分区，从而提升性能。
 *
 * @param collection_name 集合的名称，指定在哪个集合中创建分区。
 * @param partition_name 需要创建的分区名称，分区名称需要唯一。
 *
 * @throws 如果创建分区失败，会输出错误消息并终止程序。
 */
void MilvusClient::CreatePartition(const std::string& collection_name, const std::string& partition_name) {
    // 尝试在指定集合中创建分区
    auto status = client_->CreatePartition(collection_name, partition_name);
    
    // 检查分区创建状态
    CheckStatus("创建分区失败:", status);
    std::cout << "成功创建分区。" << std::endl;
}


// 插入数据
void MilvusClient::InsertData(const std::string& collection_name, const std::string& partition_name, std::vector<milvus::FieldDataPtr> fields_data) {
    milvus::DmlResults dml_results;
    auto status = client_->Insert(collection_name, partition_name, fields_data, dml_results);
    CheckStatus("插入数据失败:", status);
    std::cout << "成功插入 " << dml_results.IdArray().IntIDArray().size() << " 行数据。" << std::endl;
    
}

// 搜索数据
void MilvusClient::SearchData(const std::string& collection_name, const std::string& partition_name
                            , const std::string& field_face_name, const std::string& field_age_name, std::vector<float> q_vector) {
    
    // 搜索参数设置
    milvus::SearchArguments arguments{};
    arguments.SetCollectionName(collection_name);       // 设置需要搜索的集合
    arguments.AddPartitionName(partition_name);         // 设置搜索的分区
    arguments.SetTopK(10);                              // 返回前10个最相似的结果
    arguments.AddOutputField(field_age_name);           // 指定要返回的字段
    arguments.SetExpression(field_age_name + " > 0");  // 查询年龄大于40的用户
    arguments.SetGuaranteeTimestamp(milvus::GuaranteeStrongTs()); // 设置强一致性保证，确保搜索在数据持久化后执行

    // 将查询向量添加到搜索参数中
    arguments.AddTargetVector(field_face_name, std::move(q_vector));

    // 执行搜索
    milvus::SearchResults search_results{};
    client_->LoadCollection(collection_name);                       // 加载集合
    auto status = client_->Search(arguments, search_results);       // 查询
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
        }
    }
}

// 删除分区
void MilvusClient::DeletePartition(const std::string& collection_name, const std::string& partition_name) {
    auto status = client_->DropPartition(collection_name, partition_name);
    CheckStatus("删除分区失败:", status);
    std::cout << "删除分区 " << partition_name << std::endl;
}

// 删除集合
void MilvusClient::DropCollection(const std::string& collection_name) {
    auto status = client_->DropCollection(collection_name);
    CheckStatus("删除集合失败:", status);
    std::cout << "删除集合 " << collection_name << std::endl;
}

/**
 * @brief 获取指定集合中某个分区的统计信息，并打印分区的行数。
 *
 * @param collection_name 集合的名称。
 * @param partition_name 分区的名称。
 */
void MilvusClient::GetPartitionStatistics(const std::string& collection_name, const std::string& partition_name) {
    // 声明 PartitionStat 对象用于存储统计信息
    milvus::PartitionStat part_stat;

    // 调用 Milvus 客户端的 GetPartitionStatistics 方法获取分区的统计信息
    auto status = client_->GetPartitionStatistics(collection_name, partition_name, part_stat);

    // 检查状态，如果失败则输出错误信息并退出
    CheckStatus("获取分区统计信息失败:", status);

    // 打印分区的行数
    std::cout << "分区 " << partition_name << " 的行数: " << part_stat.RowCount() << std::endl;
}


// 辅助函数：计时器
template <typename Func>
void MilvusClient::MeasureTime(const std::string& task_name, Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << task_name << " 耗时: " << duration.count() * 1000 << " 毫秒" << std::endl;
}
