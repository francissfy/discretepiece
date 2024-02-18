#include <memory>
#include "util.h"
#include "io_utils.h"
#include "absl/strings/str_split.h"
#include "absl/strings/str_join.h"


bool io_utils::is_valid_kaldi_rspec(const std::string &rspec) {
  auto input_rspec_type = kaldiio::ClassifyRspecifier(rspec, NULL, NULL);
  bool is_kaldiio = (input_rspec_type == kaldiio::RspecifierType::kArchiveRspecifier || 
                      input_rspec_type == kaldiio::RspecifierType::kScriptRspecifier);
  return is_kaldiio;  
}

bool io_utils::is_valid_kaldi_wspec(const std::string &wspec) {
  auto output_wspec_type = kaldiio::ClassifyWspecifier(wspec, NULL, NULL, NULL);
  bool is_kaldiio = (output_wspec_type == kaldiio::WspecifierType::kArchiveWspecifier || 
                      output_wspec_type == kaldiio::WspecifierType::kScriptWspecifier || 
                      output_wspec_type == kaldiio::WspecifierType::kBothWspecifier);
  return is_kaldiio;
}

io_utils::GeneralIndexReader::GeneralIndexReader(const std::string &filename): done_(false) {
    if (is_valid_kaldi_rspec(filename)) {
        input_type_ = IO_TYPES::KALDI_INPUT;
        kaldi_reader_ = std::make_unique<io_utils::SequentialFloatMatrixReader>(filename);
        CHECK(kaldi_reader_->IsOpen()) << "error open rspec: " << filename;
    } else {
        input_type_ = IO_TYPES::TEXT_FILE;
        auto input = discretepiece::filesystem::NewReadableFile(filename);
        CHECK_OK(input->status());
        file_reader_.swap(input);
    }
    // init reading
    Next();
}

io_utils::GeneralIndexReader::~GeneralIndexReader() {
    if (input_type_ == IO_TYPES::TEXT_FILE) {
        // pass
    } else if (input_type_ == IO_TYPES::KALDI_INPUT) {
        kaldi_reader_->Close();
    } else {
        // pass
    }
}

bool io_utils::GeneralIndexReader::Done() {
    return done_;
}

void io_utils::GeneralIndexReader::Next() {
    if (input_type_ == IO_TYPES::TEXT_FILE) {
        std::string line;
        done_ = !file_reader_->ReadLine(&line);
        if (!done_) {
            std::vector<std::string> str_parts = absl::StrSplit(line, ' ', false);
            key_ = str_parts[0];
            value_.resize(str_parts.size()-1);
            std::transform(
                str_parts.begin()+1, str_parts.end(),
                value_.begin(),
                [] (const std::string &s) { return static_cast<char32>(std::stoi(s)); }
            );
        }
    } else if (input_type_ == IO_TYPES::KALDI_INPUT) {
        kaldi_reader_->Next();
        done_ = kaldi_reader_->Done();
        if (!done_) {
            key_ = kaldi_reader_->Key();
            auto v = kaldi_reader_->Value();
            CHECK(v.NumCols() == 1) << "input kaldi shape should be (t, 1)";
            value_.resize(v.NumRows());
            for (int i=0; i<v.NumRows(); i++)
                value_[i] = static_cast<char32>(v(i, 0));
            // free current
            kaldi_reader_->FreeCurrent();
        }
    } else {
        // invalid
    }
}

std::string io_utils::GeneralIndexReader::Key() {
    return key_;
}

std::vector<char32> io_utils::GeneralIndexReader::Value() {
    return value_;
}

io_utils::GeneralIndexWriter::GeneralIndexWriter(const std::string &filename) {
    if (is_valid_kaldi_wspec(filename)) {
        output_type_ = IO_TYPES::KALDI_OUTPUT;
        kaldi_writer_ = std::make_unique<io_utils::FloatMatrixWriter>(filename);
        CHECK(kaldi_writer_->IsOpen()) << "error open wspec: " << filename;
    } else {
        output_type_ = IO_TYPES::TEXT_FILE;
        auto output = discretepiece::filesystem::NewWritableFile(filename);
        CHECK_OK(output->status());
        file_writer_.swap(output);
    }
}

io_utils::GeneralIndexWriter::~GeneralIndexWriter() {
    if (output_type_ == IO_TYPES::TEXT_FILE) {
        // pass
    } else if (output_type_ == IO_TYPES::KALDI_OUTPUT) {
        kaldi_writer_->Close();
    } else {
        // pass
    }
}

void io_utils::GeneralIndexWriter::Write(const std::string &key, const std::vector<char32> &value) {
    if (output_type_ == IO_TYPES::TEXT_FILE) {
        auto value_string = discretepiece::string_util::VectorChar32ToString(value, " ");
        value_string.insert(0, key + " ");
        file_writer_->WriteLine(value_string);
    } else if (output_type_ == IO_TYPES::KALDI_OUTPUT) {
        FloatMatrix matrix(value.size(), 1);
        for (int i=0; i<value.size(); i++)
            matrix(i, 0) = static_cast<float>(value[i]);
        kaldi_writer_->Write(key, matrix);
    } else {
        // pass
    }
}

void io_utils::GeneralIndexWriter::WritePieces(const std::string &key, const std::vector<std::string> &pieces) {
    if (output_type_ == IO_TYPES::TEXT_FILE) {
        std::string value_string = absl::StrJoin(pieces, " ");
        value_string.insert(0, key + " ");
        file_writer_->WriteLine(value_string);
    } else {
        // pass
        // TODO error check
    }
}