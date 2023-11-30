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
  
  CHECK(!absl::GetFlag(FLAGS_model).empty());

  discretepiece::DiscretePieceProcessor sp;
  CHECK_OK(sp.Load(absl::GetFlag(FLAGS_model)));

  auto output = discretepiece::filesystem::NewWritableFile(absl::GetFlag(FLAGS_output));
  CHECK_OK(output->status());

  std::string line;
  std::vector<std::vector<char32>> sps;
  std::vector<int> ids;
  std::function<void(absl::string_view line)> process;

  if (absl::GetFlag(FLAGS_output_format) == "piece") {
    process = [&](absl::string_view line) {
      std::vector<char32> normalized = discretepiece::string_util::StringToVectorChar32(line, {}, ' ');
      sps.clear();
      CHECK_OK(sp.Encode(normalized, &sps));
      std::vector<std::string> str_pieces(sps.size());
      std::transform(
        sps.begin(), sps.end(),
        str_pieces.begin(),
        [] (const std::vector<char32> &s) {
          return discretepiece::string_util::VectorChar32ToString(s, "_");
        }
      );
      output->WriteLine(absl::StrJoin(str_pieces, " "));
    };
  } else if (absl::GetFlag(FLAGS_output_format) == "id") {
    process = [&](absl::string_view line) {
      std::vector<char32> normalized = discretepiece::string_util::StringToVectorChar32(line, {}, ' ');
      ids.clear();
      CHECK_OK(sp.Encode(normalized, &ids));
      output->WriteLine(absl::StrJoin(ids, " "));
    };
  } else {
    LOG(FATAL) << "Unknown output format: " << absl::GetFlag(FLAGS_output_format);
  }

  // main 
  auto input = discretepiece::filesystem::NewReadableFile(absl::GetFlag(FLAGS_input));
  CHECK_OK(input->status());
  while (input->ReadLine(&line)) {
    process(line);
  }

  return 0;
}
