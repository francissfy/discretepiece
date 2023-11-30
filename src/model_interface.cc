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

#include <algorithm>

#include "model_interface.h"
#include "discretepiece_model.pb.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/strings/str_format.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "util.h"

namespace discretepiece {

ModelInterface::ModelInterface(const ModelProto &model_proto): model_proto_(&model_proto), status_(util::OkStatus()) {}

ModelInterface::~ModelInterface() {}

int ModelInterface::PieceToId(const std::vector<char32> &piece) const {
  auto it = pieces_.find(piece);
  CHECK(it != pieces_.end()) << string_util::VectorChar32ToString(piece, "_") << " cannot found";
  return it->second;  
}

std::vector<char32> ModelInterface::IdToPiece(int id) const {
  return string_util::StringToVectorChar32(model_proto_->pieces(id).piece(), {}, '_');
}

int ModelInterface::GetPieceSize() const {
  if (!model_proto_) return 0;
  return model_proto_->pieces_size();
}

float ModelInterface::GetScore(int id) const {
  return model_proto_->pieces(id).score();
}

void ModelInterface::InitializePieces() {
  pieces_.clear();

  for (int i = 0; i < model_proto_->pieces_size(); ++i) {
    const auto &sp = model_proto_->pieces(i);
    if (sp.piece().empty()) {
      status_ = util::InternalError("piece must not be empty.");
      return;
    }

    if (!port::InsertIfNotPresent(&pieces_, IdToPiece(i), i)) {
        status_ = util::InternalError(sp.piece() + " is already defined.");
      return;
    }
  }
}


}  // namespace discretepiece
