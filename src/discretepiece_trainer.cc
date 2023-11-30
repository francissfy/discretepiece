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

#include <string>
#include <vector>

#include "common.h"
#include "discretepiece_model.pb.h"
#include "discretepiece_trainer.h"
#include "spec_parser.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/strings/strip.h"
#include "trainer_factory.h"
#include "util.h"

namespace discretepiece {

//static 
util::Status DiscretePieceTrainer::Train(const TrainerSpec &trainer_spec) {
  return Train(trainer_spec, nullptr);
}

// static
util::Status DiscretePieceTrainer::Train(const TrainerSpec &trainer_spec, SentenceIterator *sentence_iterator) {

  auto trainer = TrainerFactory::Create(trainer_spec);
  std::string info = absl::StrCat(PrintProto(trainer_spec, "trainer_spec"));

  LOG(INFO) << "Starts training with : \n" << info;

  RETURN_IF_ERROR(trainer->Train(sentence_iterator, nullptr));

  return util::OkStatus();
}

util::Status DiscretePieceTrainer::PopulateModelTypeFromString(absl::string_view type, TrainerSpec *spec) {
  static const std::unordered_map<std::string, TrainerSpec::ModelType> kModelTypeMap = {
    {"bpe", TrainerSpec::BPE},
    // for future extension
  };
  const auto it = kModelTypeMap.find(absl::AsciiStrToLower(type));
  if (it != kModelTypeMap.end()) {
    spec->set_model_type(it->second);
    return util::OkStatus();
  }

  return util::StatusBuilder(util::StatusCode::kInternal, GTL_LOC)
         << "\"" << type << "\" is not found in TrainerSpec";
}

}  // namespace discretepiece
