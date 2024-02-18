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

#include <functional>
#include <string>
#include <vector>

#include "common.h"
#include "filesystem.h"
#include "init.h"
#include "util.h"
#include "io_utils.h"
#include "discretepiece_processor.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"

ABSL_FLAG(std::string, model, "", "model file name");
ABSL_FLAG(std::string, output_format, "piece", "choose from piece or id");
ABSL_FLAG(std::string, input, "", "input filename");
ABSL_FLAG(std::string, output, "", "output filename");


int main(int argc, char *argv[]) {
  discretepiece::ScopedResourceDestructor cleaner;
  discretepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);
  
  CHECK(!absl::GetFlag(FLAGS_model).empty()) << "empty --model";
  CHECK(
    absl::GetFlag(FLAGS_output_format) == "piece" || 
    absl::GetFlag(FLAGS_output_format) == "id"
  ) << "--output_format should be piece or id, piece is only allowed in text output format";

  // check whether kaldi output
  if (io_utils::is_valid_kaldi_wspec(absl::GetFlag(FLAGS_output)))
    CHECK(absl::GetFlag(FLAGS_output_format) != "piece") << "--output_format piece is not valid in kaldi output";

  discretepiece::DiscretePieceProcessor sp;
  CHECK_OK(sp.Load(absl::GetFlag(FLAGS_model)));

  auto index_reader = io_utils::GeneralIndexReader(absl::GetFlag(FLAGS_input));
  auto index_writer = io_utils::GeneralIndexWriter(absl::GetFlag(FLAGS_output));

  for (; !index_reader.Done(); index_reader.Next()) {
    std::string key = index_reader.Key();
    std::vector<char32> value = index_reader.Value();

    if (absl::GetFlag(FLAGS_output_format) == "piece") {
      std::vector<std::vector<char32>> pieces;
      CHECK_OK(sp.Encode(value, &pieces));
      std::vector<std::string> str_pieces(pieces.size());
      std::transform(
        pieces.begin(), pieces.end(),
        str_pieces.begin(),
        [] (const std::vector<char32> &s) {
          return discretepiece::string_util::VectorChar32ToString(s, "_");
        }
      );
      index_writer.WritePieces(key, str_pieces);

    } else {
      std::vector<int> encoded_value;
      CHECK_OK(sp.Encode(value, &encoded_value));
      std::vector<char32> encoded_value_char32(encoded_value.size());
      std::transform(
        encoded_value.begin(), encoded_value.end(),
        encoded_value_char32.begin(), 
        [] (int v) { return static_cast<char32>(v); }
      );
      index_writer.Write(key, encoded_value_char32);

    }
  }

  return 0;
}
