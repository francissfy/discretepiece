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

#ifndef DISCRETEPIECE_PROCESSOR_H_
#define DISCRETEPIECE_PROCESSOR_H_

#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "model_interface.h"

namespace discretepiece {


class DiscretePieceProcessor {
 public:
  DiscretePieceProcessor();
  
  virtual ~DiscretePieceProcessor();

  // Loads model from `filename`.
  // Returns false if `filename` cannot be loaded.
  virtual util::Status Load(absl::string_view filename);

  // Loads model from `filename`.
  // Crash if `filename` cannot be loaded.
  virtual void LoadOrDie(absl::string_view filename);

  // Loads model from `model_proto`.
  // `model_proto` is moved.
  virtual util::Status Load(std::unique_ptr<ModelProto> model_proto);

  // Returns the status. Encode/Decode methods are valid when status is OK.
  virtual util::Status status() const;

  //////////////////////////////////////////////////////////////
  // Simple Encode and Decode API.
  //
  // Given a vector<char32> input, encodes it into a sequence of pieces
  virtual util::Status Encode(const std::vector<char32> &input, std::vector<std::vector<char32>> *tokenized) const;

  // Given a vector<char32> input, encodes it into a sequence of piece_ids
  virtual util::Status Encode(const std::vector<char32> &input, std::vector<int> *tokenized) const;

  // Given a sequence of pieces, decodes it into a detokenized output.
  virtual util::Status Decode(const std::vector<std::vector<char32>> &pieces, std::vector<char32> *detokenized) const;

  // Given a sequence of ids, decodes it into a detokenized output.
  virtual util::Status Decode(const std::vector<int> &ids, std::vector<char32> *detokenized) const;

 private:

  std::unique_ptr<ModelInterface> model_;

  // Underlying model protocol buffer. The same lifetime as model_.
  std::unique_ptr<ModelProto> model_proto_;
};


// IO related functions to absorb model formats.
namespace io {
// Loads `model_proto` from `filename`.
// We can instantiate SentencePieceProcessor as follows:
//
//  auto model_proto = absl::make_unique<ModelProto>();
//  io::LoadModelProto("//path/spm.model", model_proto.get());
//  SentencePieceProcessor sp;
//  CHECK_OK(sp.Load(std::move(model_proto)));
util::Status LoadModelProto(absl::string_view, ModelProto *model_proto);

// Saves `model_proto` as `filename`.
util::Status SaveModelProto(absl::string_view, const ModelProto &model_proto);
}  // namespace io
}  // namespace discretepiece
#endif  // DISCRETEPIECE_PROCESSOR_H_
