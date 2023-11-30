// Copyright 2018 Google Inc.
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

#ifndef DISCRETEPIECE_TRAINER_H_
#define DISCRETEPIECE_TRAINER_H_

#include <string>
#include <unordered_map>
#include "util.h"

namespace discretepiece {

class TrainerSpec;

// Iterator over the training sentences.
// Training sentences are loaded sequentially as follows:
//
// for (; !it.done(); it.Next()) {
//    const std::string &s = it.value();
// }
// RETURN_IF_ERROR(it.status());
//
class SentenceIterator {
 public:
  virtual ~SentenceIterator() {}
  // Returns true if iteration finishes (including error case).
  // Uses SentenceIterator::status() method to know whether
  // all sentences are loaded successfully.
  virtual bool done() const = 0;
  virtual void Next() = 0;
  virtual const std::string &value() const = 0;
  virtual util::Status status() const = 0;
};

class DiscretePieceTrainer {
 public:

  static util::Status Train(const TrainerSpec &trainer_spec);

  static util::Status Train(const TrainerSpec &trainer_spec, 
                            SentenceIterator *sentence_iterator);

  // Populates model type from string representation, e.g., "bpe".
  // Supported model: "bpe".
  static util::Status PopulateModelTypeFromString(absl::string_view type,
                                                  TrainerSpec *trainer_spec);
 
 private:
  DiscretePieceTrainer() {}
  ~DiscretePieceTrainer() {}
};

}  // namespace discretepiece

#endif  // DISCRETEPIECE_TRAINER_H_
