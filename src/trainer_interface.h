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

#ifndef TRAINER_INTERFACE_H_
#define TRAINER_INTERFACE_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <limits>

#include "common.h"
#include "filesystem.h"
#include "discretepiece_model.pb.h"
#include "discretepiece_trainer.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "util.h"

namespace discretepiece {

template <typename K, typename V>
std::vector<std::pair<K, V>> Sorted(const std::vector<std::pair<K, V>> &m) {
  std::vector<std::pair<K, V>> v = m;
  std::sort(v.begin(), v.end(),
            [](const std::pair<K, V> &p1, const std::pair<K, V> &p2) {
              return (p1.second > p2.second ||
                      (p1.second == p2.second && p1.first < p2.first));
            });
  return v;
}

template <typename K, typename V, typename H>
std::vector<std::pair<K, V>> Sorted(const absl::flat_hash_map<K, V, H> &m) {
  std::vector<std::pair<K, V>> v(m.begin(), m.end());
  return Sorted(v);
}


class MultiFileSentenceIterator : public SentenceIterator {
 public:
  explicit MultiFileSentenceIterator(const std::vector<std::string> &files);

  ~MultiFileSentenceIterator() {}

  bool done() const override;
  void Next() override;
  const std::string &value() const override { return value_; }
  util::Status status() const override;

 private:
  void TryRead();

  bool read_done_ = false;
  size_t file_index_ = 0;
  std::vector<std::string> files_;
  std::string value_;
  std::unique_ptr<filesystem::ReadableFile> fp_;
};


// Base trainer class
class TrainerInterface {
 public:
  using Sentence = std::pair<std::vector<char32>, int64>;
  using Sentences = std::vector<Sentence>;

  TrainerInterface(const TrainerSpec &trainer_spec);

  virtual ~TrainerInterface();

  // Loads sentence from `sentence_iterator` and stores the model to `output_model_proto`.
  virtual util::Status Train(SentenceIterator *sentence_iterator, ModelProto *output_model_proto) {
    sentence_iterator_ = sentence_iterator;
    output_model_proto_ = output_model_proto;
    return Train();
  }

  virtual util::Status Train() { return status(); }

  virtual util::Status status() const { return status_; }

  // FRIEND_TEST(TrainerInterfaceTest, IsValidSentencePieceTest);
  // FRIEND_TEST(TrainerInterfaceTest, OverrideSpecialPiecesTest);
  // FRIEND_TEST(TrainerInterfaceTest, BytePiecesTest);
  // FRIEND_TEST(TrainerInterfaceTest, SerializeTest);
  // FRIEND_TEST(TrainerInterfaceTest, CharactersTest);

  // Loads all sentences from spec.input() or SentenceIterator.
  // It loads at most input_sentence_size sentences.
  util::Status LoadSentences();

 protected:
  // Returns true if |piece| is valid sentence piece.
  // The result is affected by
  // max_sentencepiece_length.
  bool IsValidDiscretePiece(const std::vector<char32> &piece) const;

  // Splits all sentencecs by deliminator and replace the |sentences_| with tokenized ones.
  // e.g.,
  // '#' or any other specified deliminator
  // "1 2 3 4 5 # 6 7 8" => [[1, 2, 3, 4, 5], [6, 7, 8]]
  void SplitSentencesByWhitespace();

  // Save model files into spec.model_prefix().
  util::Status Save() const;

  // Set of characters which must be included in the final vocab.
  // The value of this map stores the frequency.
  absl::flat_hash_map<char32, int64> required_chars_;

  // Final output pieces
  std::vector<std::pair<std::vector<char32>, float>> final_pieces_;

  // All sentences.
  Sentences sentences_;

  // Trainer spec.
  TrainerSpec trainer_spec_;

  // Mapping deliminator to a special char32 value for compatibility
  absl::flat_hash_map<char, char32> deliminator_map_;
  const char32 deliminator_char32_value_ = std::numeric_limits<char32>::max();

  // Detect errors on initialization.
  util::Status status_;

  // Loads sentences from SentenceIterator if not null.
  SentenceIterator *sentence_iterator_ = nullptr;

  // Emits model to this proto instead of file.
  ModelProto *output_model_proto_ = nullptr;

 private:
  // Serialize final_pieces_ to |model_proto|.
  util::Status Serialize(ModelProto *model_proto) const;

  // Saves model file.
  util::Status SaveModel(absl::string_view filename) const;

  // Saves vocabulary file for NMT.
  util::Status SaveVocab(absl::string_view filename) const;

  // Initializes deliminator from TrainerSpec.
  util::Status InitDeliminatorPieces();

};
}  // namespace discretepiece
#endif  // TRAINER_INTERFACE_H_
