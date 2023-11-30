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

#include "discretepiece_processor.h"

#include <map>
#include <set>
#include <utility>

#include "common.h"
#include "filesystem.h"
#include "model_factory.h"
#include "model_interface.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/str_replace.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/strings/strip.h"
#include "util.h"

namespace discretepiece {

DiscretePieceProcessor::DiscretePieceProcessor() {}

DiscretePieceProcessor::~DiscretePieceProcessor() {}

util::Status DiscretePieceProcessor::Load(absl::string_view filename) {
  auto model_proto = absl::make_unique<ModelProto>();
  RETURN_IF_ERROR(io::LoadModelProto(filename, model_proto.get()));
  return Load(std::move(model_proto));
}

void DiscretePieceProcessor::LoadOrDie(absl::string_view filename) {
  CHECK_OK(Load(filename));
}

util::Status DiscretePieceProcessor::Load(std::unique_ptr<ModelProto> model_proto) {
  model_proto_ = std::move(model_proto);
  model_ = ModelFactory::Create(*model_proto_);

  RETURN_IF_ERROR(status());

  return util::OkStatus();
}

util::Status DiscretePieceProcessor::status() const {
  CHECK_OR_RETURN(model_) << "Model is not initialized.";
  RETURN_IF_ERROR(model_->status());
  return util::OkStatus();
}

//////////////////////////////////////////////////////////////
// Simple API.
util::Status DiscretePieceProcessor::Encode(const std::vector<char32> &input, std::vector<std::vector<char32>> *tokenized) const {
  for (const auto &p: model_->Encode(input)) {
    tokenized->push_back(p.first);
  }
  return util::OkStatus();
}

util::Status DiscretePieceProcessor::Encode(const std::vector<char32> &input, std::vector<int> *tokenized) const {
  for (const auto &p: model_->Encode(input)) {
    tokenized->push_back(p.second);
  }
  return util::OkStatus();
}

util::Status DiscretePieceProcessor::Decode(const std::vector<std::vector<char32>> &pieces, std::vector<char32> *detokenized) const {
  for (const std::vector<char32> &p: pieces) {
    detokenized->insert(detokenized->end(), p.cbegin(), p.cend());
  }
  return util::OkStatus();
}


util::Status DiscretePieceProcessor::Decode(const std::vector<int> &ids, std::vector<char32> *detokenized) const {
  for (int id: ids) {
    auto piece = model_->IdToPiece(id);
    detokenized->insert(detokenized->end(), piece.begin(), piece.end());
  }
  return util::OkStatus();
}


namespace io {

util::Status LoadModelProto(absl::string_view filename,
                            ModelProto *model_proto) {
  if (filename.empty()) {
    return util::NotFoundError("model file path should not be empty.");
  }

  auto input = filesystem::NewReadableFile(filename, true);
  RETURN_IF_ERROR(input->status());
  std::string serialized;
  CHECK_OR_RETURN(input->ReadAll(&serialized));
  CHECK_OR_RETURN(
      model_proto->ParseFromArray(serialized.data(), serialized.size()));

  return util::OkStatus();
}

util::Status SaveModelProto(absl::string_view filename,
                            const ModelProto &model_proto) {
  if (filename.empty()) {
    return util::NotFoundError("model file path should not be empty.");
  }
  auto output = filesystem::NewWritableFile(filename, true);
  RETURN_IF_ERROR(output->status());
  CHECK_OR_RETURN(output->Write(model_proto.SerializeAsString()));

  return util::OkStatus();
}
}  // namespace io

}  // namespace discretepiece
