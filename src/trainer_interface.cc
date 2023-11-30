// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#include "trainer_interface.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <set>
#include <unordered_set>
#include <string>
#include <utility>
#include <vector>

#include "filesystem.h"
#include "discretepiece_trainer.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/random/distributions.h"
#include "third_party/absl/random/random.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_format.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/str_split.h"
#include "util.h"

namespace discretepiece {

namespace {

util::Status VerifySpec(const TrainerSpec &trainer_spec) {
  CHECK_GT_OR_RETURN(trainer_spec.vocab_size(), 0);

#define CHECK_RANGE(variable, minval, maxval) \
  CHECK_OR_RETURN(variable >= minval && variable <= maxval)

  CHECK_RANGE(trainer_spec.num_sub_iterations(), 1, 10);
  CHECK_RANGE(trainer_spec.num_threads(), 1, 1024);
#undef CHECK_RANGE

  CHECK_OR_RETURN(trainer_spec.input_sentence_size() <= 0 ||
                  trainer_spec.input_sentence_size() > 100);

  return util::OkStatus();
}

class SentenceSelector {
 public:
  using Sampler = random::ReservoirSampler<TrainerInterface::Sentence>;

  static constexpr int64 kTooBigSentencesSize = 1000000;

  SentenceSelector(TrainerInterface::Sentences *sentences, const TrainerSpec &spec)
      : sentences_(sentences), spec_(&spec) {
    if (spec_->input_sentence_size() > 0) {
      if (spec_->shuffle_input_sentence()) {
        constexpr size_t kSeed = 12345678;
        sampler_ = absl::make_unique<Sampler>(
            sentences, spec_->input_sentence_size(), kSeed);
      } else {
        LOG(INFO)
            << "First " << spec_->input_sentence_size()
            << " sentences are selected. Remaining sentences are discarded.";
      }
    }
  }

  void Finish() const {
    if (sentences_->size() > kTooBigSentencesSize) {
      LOG(WARNING) << "Too many sentences are loaded! (" << sentences_->size()
                   << "), which may slow down training.";
      LOG(WARNING) << "Consider using "
                      "--input_sentence_size=<size> and "
                      "--shuffle_input_sentence=true.";
      LOG(WARNING) << "They allow to randomly sample <size> sentences from "
                      "the entire corpus.";
    }
  }

  bool Add(const std::pair<std::vector<char32>, int64> &sentence) {
    if (spec_->input_sentence_size() == 0) {
      sentences_->emplace_back(sentence);
    } else {
      if (spec_->shuffle_input_sentence()) {
        sampler_->Add(sentence);
      } else {
        sentences_->emplace_back(sentence);
        if (sentences_->size() >= spec_->input_sentence_size()) return false;
      }
    }

    if (total_size() > 0 && total_size() % kTooBigSentencesSize == 0) {
      LOG(INFO) << "Loaded " << total_size() << " lines";
    }

    return true;
  }

  size_t total_size() const {
    return sampler_.get() ? sampler_->total_size() : sentences_->size();
  }

 private:
  TrainerInterface::Sentences *sentences_ = nullptr;
  const TrainerSpec *spec_ = nullptr;
  std::unique_ptr<Sampler> sampler_;
};

}  // namespace

MultiFileSentenceIterator::MultiFileSentenceIterator(
    const std::vector<std::string> &files)
    : files_(files) {
  Next();
}

bool MultiFileSentenceIterator::done() const {
  return (!read_done_ && file_index_ == files_.size());
}

util::Status MultiFileSentenceIterator::status() const {
  CHECK_OR_RETURN(fp_);
  return fp_->status();
}

void MultiFileSentenceIterator::Next() {
  TryRead();

  if (!read_done_ && file_index_ < files_.size()) {
    const auto &filename = files_[file_index_++];
    fp_ = filesystem::NewReadableFile(filename);
    LOG(INFO) << "Loading corpus: " << filename;
    if (fp_->status() != util::OkStatus()) {
      file_index_ = files_.size();
      read_done_ = false;
      return;
    }

    TryRead();
  }
}

void MultiFileSentenceIterator::TryRead() {
  read_done_ = fp_ && fp_->ReadLine(&value_);
}


TrainerInterface::TrainerInterface(const TrainerSpec &trainer_spec)
  : trainer_spec_(trainer_spec) {
  status_ = VerifySpec(trainer_spec_);
  if (status_.ok()) 
    status_ = InitDeliminatorPieces();
}

TrainerInterface::~TrainerInterface() {}

bool TrainerInterface::IsValidDiscretePiece(const std::vector<char32> &piece) const {
  // Returns false if the length of piece is invalid.
  if (piece.empty() || piece.size() > static_cast<size_t>(trainer_spec_.max_discretepiece_length()))
    return false;
  
  for (char32 c: piece) {
    // we do not allow deliminator in piece
    if (c == deliminator_char32_value_)
      return false;
    
    // NOTE for any futher checks
  }
  return true;
}


util::Status TrainerInterface::InitDeliminatorPieces() {
  const std::string &deliminator_str = trainer_spec_.deliminator();
  for (char c: deliminator_str) {
    deliminator_map_.emplace(c, deliminator_char32_value_);
  }
  return util::OkStatus();
}


util::Status TrainerInterface::LoadSentences() {
  RETURN_IF_ERROR(status());
  CHECK_OR_RETURN(sentences_.empty());
  CHECK_OR_RETURN(trainer_spec_.input_format().empty() ||
                  trainer_spec_.input_format() == "text")
      << "Supported formats are 'text'.";

  CHECK_OR_RETURN(
      (sentence_iterator_ != nullptr && trainer_spec_.input().empty()) ||
      (sentence_iterator_ == nullptr && !trainer_spec_.input().empty()))
      << "SentenceIterator and trainer_spec.input() must be exclusive.";

  CHECK_OR_RETURN(
      (output_model_proto_ != nullptr && trainer_spec_.model_prefix().empty()) ||
      (output_model_proto_ == nullptr && !trainer_spec_.model_prefix().empty()))
      << "ModelProto and trainer_spec.model_prefix() must be exclusive.";

  SentenceSelector selector(&sentences_, trainer_spec_);

  std::unique_ptr<SentenceIterator> sentence_iterator_impl;
  if (sentence_iterator_ == nullptr) {
    LOG(INFO) << "SentenceIterator is not specified. Using "
                 "MultiFileSentenceIterator.";
    sentence_iterator_impl =
        absl::make_unique<MultiFileSentenceIterator>(std::vector<std::string>(
            trainer_spec_.input().begin(), trainer_spec_.input().end()));
    sentence_iterator_ = sentence_iterator_impl.get();
  }

  for (; !sentence_iterator_->done(); sentence_iterator_->Next()) {
    int64 freq = 1;
    std::string sentence_str = sentence_iterator_->value();

    if (sentence_str.empty()) continue;

    // split sentence into list of ints & mapping deliminators
    std::vector<char32> sentence = string_util::StringToVectorChar32(sentence_str, deliminator_map_);

    if (!selector.Add(std::make_pair(sentence, freq))) {
      goto END;
    }
  }

  RETURN_IF_ERROR(sentence_iterator_->status());

END:
  // Emits error message if any.
  selector.Finish();

  if (sentences_.size() == selector.total_size()) {
    LOG(INFO) << "Loaded all " << sentences_.size() << " sentences";
  } else {
    LOG(INFO) << "Sampled " << sentences_.size() << " sentences from "
              << selector.total_size() << " sentences.";
  }

  // report vocabulary size
  for (const auto &w : sentences_) {
    for (char32 c: w.first) {
      if (c == deliminator_char32_value_) 
        continue;
      required_chars_[c] += w.second;
      // required_chars_.find(c)->second += w.second;
    }
  }

  LOG(INFO) << "Alphabet size=" << required_chars_.size();
  LOG(INFO) << "Done! preprocessed " << sentences_.size() << " sentences.";
  return util::OkStatus();
}


void TrainerInterface::SplitSentencesByWhitespace() {
  LOG(INFO) << "Tokenizing input sentences with whitespace: " << sentences_.size();
  
  absl::flat_hash_map<std::vector<char32>, int64, port::VectorChar32Hash> tokens;

  for (const auto &s : sentences_) {
    for (const auto &w: port::VectorSplit(s.first, deliminator_char32_value_)) {
      tokens[w] += s.second;
    }
  }

  sentences_ = Sorted(tokens);
  LOG(INFO) << "Done! " << sentences_.size();
}


util::Status TrainerInterface::Serialize(ModelProto *model_proto) const {
  RETURN_IF_ERROR(status());

  // Duplicated piece is not allowed.
  std::unordered_set<std::vector<char32>, port::VectorChar32Hash> dup;

  model_proto->Clear();

#define CHECK_PIECE(piece)                                  \
  CHECK_OR_RETURN(!piece.empty());                          \
  CHECK_OR_RETURN(dup.insert(piece).second) << piece << " is already defined";

  CHECK_EQ_OR_RETURN(static_cast<int32>(final_pieces_.size()), trainer_spec_.vocab_size()) << "final piece size not equal";
  
  for (auto fid=0; fid<final_pieces_.size(); fid++) {
    const auto &w = final_pieces_[fid];
    auto *sp = model_proto->add_pieces();
    CHECK_PIECE(w.first);
    sp->set_type(ModelProto::DiscretePiece::NORMAL);
    sp->set_piece(string_util::VectorChar32ToString(w.first, "_"));
    sp->set_score(w.second);
  }

  *(model_proto->mutable_trainer_spec()) = trainer_spec_;

  return util::OkStatus();
}

util::Status TrainerInterface::SaveModel(absl::string_view filename) const {
  LOG(INFO) << "Saving model: " << filename;
  ModelProto model_proto;

  RETURN_IF_ERROR(Serialize(&model_proto));

  auto output = filesystem::NewWritableFile(filename.data(), true);
  RETURN_IF_ERROR(output->status());
  output->Write(model_proto.SerializeAsString());
  return util::OkStatus();
}

util::Status TrainerInterface::SaveVocab(absl::string_view filename) const {
  LOG(INFO) << "Saving vocabs: " << filename;
  ModelProto model_proto;
  RETURN_IF_ERROR(Serialize(&model_proto));
  auto output = filesystem::NewWritableFile(filename);
  RETURN_IF_ERROR(output->status());

  for (const auto &piece : model_proto.pieces()) {
    if (piece.piece().find_first_of(" \t\r\n") != std::string::npos) {
      LOG(WARNING) << "The piece [" << piece.piece()
                   << "] contains escaped characters that break the format of "
                   << filename;
    }
  }

  if (trainer_spec_.vocabulary_output_piece_score()) {
    for (const auto &piece : model_proto.pieces()) {
      std::ostringstream os;
      os << piece.piece() << "\t" << piece.score();
      CHECK_OR_RETURN(output->WriteLine(os.str()));
    }
  } else {
    for (const auto &piece : model_proto.pieces()) {
      CHECK_OR_RETURN(output->WriteLine(piece.piece()));
    }
  }

  return util::OkStatus();
}

util::Status TrainerInterface::Save() const {
  if (output_model_proto_) {
    RETURN_IF_ERROR(Serialize(output_model_proto_));
  } else {
    RETURN_IF_ERROR(SaveModel(trainer_spec_.model_prefix() + ".model"));
    RETURN_IF_ERROR(SaveVocab(trainer_spec_.model_prefix() + ".vocab"));
  }
  return util::OkStatus();
}


}  // namespace discretepiece
