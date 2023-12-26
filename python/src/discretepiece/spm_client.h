#include "discretepiece_processor.h"

class SpmClient {
public:
  SpmClient(const char* model): model_(model) {
    sp_.Load(model_);
  }

  std::vector<int> Encode(const std::vector<int>& input) {
    std::vector<char32> char32_input(input.size());
    std::transform(
      input.begin(), 
      input.end(), 
      char32_input.begin(), 
      [](int v) { return static_cast<char32>(v); }
    );
    std::vector<int> ret;
    auto status = sp_.Encode(char32_input, &ret);
    // check status
    return ret;
  }

  std::vector<int> Decode(const std::vector<int>& input) {
    std::vector<char32> ret;
    auto status = sp_.Decode(input, &ret);
    std::vector<int> int_ret(ret.size());
    std::transform(
      ret.begin(), 
      ret.end(), 
      int_ret.begin(), 
      [](int v) { return static_cast<int>(v); }
    );
    return int_ret;
  }
  
private:
  std::string model_;
  discretepiece::DiscretePieceProcessor sp_;
};
