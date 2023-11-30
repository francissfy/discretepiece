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

#include <map>

#include "filesystem.h"
#include "init.h"
#include "discretepiece_model.pb.h"
#include "discretepiece_trainer.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/strings/ascii.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/str_split.h"
#include "util.h"

using discretepiece::TrainerSpec;

namespace {
static discretepiece::TrainerSpec kDefaultTrainerSpec;
}  // namespace

ABSL_FLAG(std::string, input, "", "comma separated list of input sentences");
ABSL_FLAG(std::string, input_format, kDefaultTrainerSpec.input_format(),
          "Input format. Supported format is `text`.");
ABSL_FLAG(std::string, model_prefix, "", "output model prefix");
ABSL_FLAG(std::string, model_type, "bpe",
          "model algorithm: bpe");
ABSL_FLAG(int32, vocab_size, kDefaultTrainerSpec.vocab_size(),
          "vocabulary size");
ABSL_FLAG(std::uint64_t, input_sentence_size,
          kDefaultTrainerSpec.input_sentence_size(),
          "maximum size of sentences the trainer loads");
ABSL_FLAG(bool, shuffle_input_sentence,
          kDefaultTrainerSpec.shuffle_input_sentence(),
          "Randomly sample input sentences in advance. Valid when "
          "--input_sentence_size > 0");
ABSL_FLAG(int32, num_threads, kDefaultTrainerSpec.num_threads(),
          "number of threads for training");
ABSL_FLAG(int32, num_sub_iterations, kDefaultTrainerSpec.num_sub_iterations(),
          "number of EM sub-iterations");
ABSL_FLAG(int32, max_discretepiece_length,
          kDefaultTrainerSpec.max_discretepiece_length(),
          "maximum length of sentence piece");
ABSL_FLAG(bool, vocabulary_output_piece_score,
          kDefaultTrainerSpec.vocabulary_output_piece_score(),
          "Define score in vocab file");
ABSL_FLAG(uint32, random_seed, static_cast<uint32>(-1),
          "Seed value for random generator.");

int main(int argc, char *argv[]) {
  discretepiece::ScopedResourceDestructor cleaner;
  discretepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);

  discretepiece::TrainerSpec trainer_spec;

  CHECK(!absl::GetFlag(FLAGS_input).empty());
  CHECK(!absl::GetFlag(FLAGS_model_prefix).empty());

  if (absl::GetFlag(FLAGS_random_seed) != -1) {
    discretepiece::SetRandomGeneratorSeed(absl::GetFlag(FLAGS_random_seed));
  }

// Populates the value from flags to spec.
#define SetTrainerSpecFromFlag(name) \
  trainer_spec.set_##name(absl::GetFlag(FLAGS_##name));

#define SetTrainerSpecFromFile(name)                                   \
  if (!absl::GetFlag(FLAGS_##name##_file).empty()) {                   \
    const auto lines = load_lines(absl::GetFlag(FLAGS_##name##_file)); \
    trainer_spec.set_##name(absl::StrJoin(lines, ""));                 \
  }

#define SetRepeatedTrainerSpecFromFlag(name)                                \
  if (!absl::GetFlag(FLAGS_##name).empty()) {                               \
    for (const auto &v :                                                    \
         discretepiece::util::StrSplitAsCSV(absl::GetFlag(FLAGS_##name))) { \
      trainer_spec.add_##name(v);                                           \
    }                                                                       \
  }

#define SetRepeatedTrainerSpecFromFile(name)                               \
  if (!absl::GetFlag(FLAGS_##name##_file).empty()) {                       \
    for (const auto &v : load_lines(absl::GetFlag(FLAGS_##name##_file))) { \
      trainer_spec.add_##name(v);                                          \
    }                                                                      \
  }

  SetRepeatedTrainerSpecFromFlag(input);

  SetTrainerSpecFromFlag(input_format);
  SetTrainerSpecFromFlag(model_prefix);
  SetTrainerSpecFromFlag(vocab_size);
  SetTrainerSpecFromFlag(input_sentence_size);
  SetTrainerSpecFromFlag(shuffle_input_sentence);
  SetTrainerSpecFromFlag(num_threads);
  SetTrainerSpecFromFlag(num_sub_iterations);
  SetTrainerSpecFromFlag(max_discretepiece_length);
  SetTrainerSpecFromFlag(vocabulary_output_piece_score);

  CHECK_OK(discretepiece::DiscretePieceTrainer::PopulateModelTypeFromString(
      absl::GetFlag(FLAGS_model_type), &trainer_spec));

  CHECK_OK(discretepiece::DiscretePieceTrainer::Train(trainer_spec));

  return 0;
}
