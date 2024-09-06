#include <iostream>
#include <random>
#include <string>
#include "milvus/MilvusClient.h"
#include "milvus/types/CollectionSchema.h"

class MilvusClient {
public:
    MilvusClient(const std::string& host, int port);
    ~MilvusClient();

    void CreateCollection(const milvus::CollectionSchema collection_schema);
    void CreateIndex(const std::string& collection_name, const std::string& field_face_name, 
                 milvus::IndexType index_type = milvus::IndexType::FLAT, 
                 milvus::MetricType metric_type = milvus::MetricType::L2);
    void CreatePartition(const std::string& collection_name, const std::string& partition_name);
    void InsertData(const std::string& collection_name, const std::string& partition_name, std::vector<milvus::FieldDataPtr> fields_data);
    void SearchData(const std::string& collection_name, const std::string& partition_name, 
                    const std::string& field_face_name, const std::string& field_age_name, std::vector<float> q_vector);
    void DeletePartition(const std::string& collection_name, const std::string& partition_name);
    void DropCollection(const std::string& collection_name);
    void GetPartitionStatistics(const std::string& collection_name, const std::string& partition_name);
    void GetCollectionStatistics(const std::string& collection_name) ;
    
private:
    std::shared_ptr<milvus::MilvusClient> client_;
    void CheckStatus(std::string&& prefix, const milvus::Status& status);

    // Helper function for timing operations
    template <typename Func>
    void MeasureTime(const std::string& task_name, Func&& func);
};
