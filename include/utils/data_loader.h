// include/utils/data_loader.h
#pragma once
#include <string>
#include <vector>

namespace semantic_search {
namespace utils {

class DataLoader {
public:
    // Function to load vector datasets
    static bool loadGloveVectors(const std::string& filename, 
        std::vector<std::vector<float>>& vectors, 
        std::vector<std::string>& labels);
    
    static bool loadSIFTVectors(const std::string& filename,
        std::vector<std::vector<float>>& vectors);
        
    static bool loadCustomEmbeddings(const std::string& filename,
        std::vector<std::vector<float>>& vectors,
        std::vector<std::string>& documents);
};

} // namespace utils
} // namespace semantic_search