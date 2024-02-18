#ifndef IO_UTILS_H_
#define IO_UTILS_H_

#include <string>
#include <vector>
#include "util.h"
#include "filesystem.h"
#include "kaldiio/kaldiio/kaldi-table.h"

namespace io_utils {

bool is_valid_kaldi_rspec(const std::string &rspec);

bool is_valid_kaldi_wspec(const std::string &wspec);

using FloatMatrix = kaldiio::Matrix<float>;
using SequentialFloatMatrixReader = kaldiio::SequentialTableReader<kaldiio::KaldiObjectHolder<FloatMatrix>>;
using FloatMatrixWriter = kaldiio::TableWriter<kaldiio::KaldiObjectHolder<FloatMatrix>>;

enum class IO_TYPES {
    TEXT_FILE,
    KALDI_INPUT,
    KALDI_OUTPUT,
};

class GeneralIndexReader {
public:
  GeneralIndexReader(const std::string &filename);

  ~GeneralIndexReader();

  bool Done();

  void Next();

  std::string Key();

  std::vector<char32> Value();

private:
IO_TYPES input_type_;

std::unique_ptr<discretepiece::filesystem::ReadableFile> file_reader_;

std::unique_ptr<SequentialFloatMatrixReader> kaldi_reader_;

std::vector<char32> value_;

std::string key_;

bool done_;

};


class GeneralIndexWriter {
public:
  GeneralIndexWriter(const std::string &filename);

  ~GeneralIndexWriter();

  void Write(const std::string &key, const std::vector<char32> &value);

  void WritePieces(const std::string &key, const std::vector<std::string> &pieces);

private:
  IO_TYPES output_type_;

  std::unique_ptr<discretepiece::filesystem::WritableFile> file_writer_;

  std::unique_ptr<FloatMatrixWriter> kaldi_writer_;
};

}   // namespace io_utils

#endif      // IO_UTILS_H_