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

#ifndef MODEL_INTERFACE_H_
#define MODEL_INTERFACE_H_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <limits>

#include "common.h"
#include "discretepiece_model.pb.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/darts_clone/darts.h"
#include "util.h"

namespace discretepiece {

using EncodeResult = std::vector<std::pair<std::vector<char32>, int>>;

// declared in discretepiece_model.proto
class ModelProto;

// Underlying model interface.
// Given a index string, returns a sequence of pieces with ids.
class ModelInterface {
 public:
  
  using PieceToIdMap = absl::flat_hash_map<std::vector<char32>, int, port::VectorChar32Hash>;

  // `model_proto` should not be deleted until ModelInterface is destroyed.
  explicit ModelInterface(const ModelProto &model_proto);

  ModelInterface() {}

  virtual ~ModelInterface();

  // Returns Status.
  // Encode/Decode functions are valid only when status is OK.
  virtual util::Status status() const { return status_; }

  virtual const ModelProto &model_proto() const { return *model_proto_; }

  // Given a normalized string, returns a sequence of sentence pieces with ids.
  // The concatenation of pieces must be the same as `normalized`.
  virtual EncodeResult Encode(const std::vector<char32> &normalized) const = 0;

  // Returns the vocab id of `piece`.
  // piece are vector of char32(uint32_t)
  virtual int PieceToId(const std::vector<char32> &piece) const;

  // Returns the representation of vocab with `id`.
  // id must be 0 <= id < GetPieceSize().
  virtual std::vector<char32> IdToPiece(int id) const;

  // Returns the size of sentence pieces, which is the same
  // as the size of vocabulary for NMT.
  virtual int GetPieceSize() const;

  // Returns the score of `id`.
  // Score represents a log probability of the piece.
  // We can roughly estimate the unigram frequency of the piece.
  virtual float GetScore(int id) const;

protected:
  void InitializePieces();

  // Non-virtual (inlined) implementation for faster execution.
  inline float GetScoreInlined(int id) const {
    return model_proto_->pieces(id).score();
  }

  const ModelProto *model_proto_ = nullptr;

  // piece -> id map for normal pieces
  PieceToIdMap pieces_;

  // status.
  util::Status status_;
};
}  // namespace discretepiece
#endif  // MODEL_INTERFACE_H_
