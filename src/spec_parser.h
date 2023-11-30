// Copyright 2016 Google LLC.
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

#ifndef SPEC_PARSER_H_
#define SPEC_PARSER_H_

#include <string>
#include <vector>

#include "third_party/absl/strings/ascii.h"
#include "third_party/absl/strings/str_split.h"
#include "util.h"

namespace discretepiece {

#define PARSE_STRING(param_name)                   \
  if (name == #param_name) {                       \
    message->set_##param_name(std::string(value)); \
    return util::OkStatus();                       \
  }

#define PARSE_REPEATED_STRING(param_name)                       \
  if (name == #param_name) {                                    \
    for (const std::string &val : util::StrSplitAsCSV(value)) { \
      message->add_##param_name(val);                           \
    }                                                           \
    return util::OkStatus();                                    \
  }

#define PARSE_BYTE(param_name)                             \
  if (name == #param_name) {                               \
    message->set_##param_name(value.data(), value.size()); \
    return util::OkStatus();                               \
  }

#define PARSE_INT32(param_name)                                               \
  if (name == #param_name) {                                                  \
    int32 v;                                                                  \
    if (!string_util::lexical_cast(value, &v))                                \
      return util::StatusBuilder(util::StatusCode::kInvalidArgument, GTL_LOC) \
             << "cannot parse \"" << value << "\" as int.";                   \
    message->set_##param_name(v);                                             \
    return util::OkStatus();                                                  \
  }

#define PARSE_UINT64(param_name)                                              \
  if (name == #param_name) {                                                  \
    uint64 v;                                                                 \
    if (!string_util::lexical_cast(value, &v))                                \
      return util::StatusBuilder(util::StatusCode::kInvalidArgument, GTL_LOC) \
             << "cannot parse \"" << value << "\" as int.";                   \
    message->set_##param_name(v);                                             \
    return util::OkStatus();                                                  \
  }

#define PARSE_DOUBLE(param_name)                                              \
  if (name == #param_name) {                                                  \
    double v;                                                                 \
    if (!string_util::lexical_cast(value, &v))                                \
      return util::StatusBuilder(util::StatusCode::kInvalidArgument, GTL_LOC) \
             << "cannot parse \"" << value << "\" as int.";                   \
    message->set_##param_name(v);                                             \
    return util::OkStatus();                                                  \
  }

#define PARSE_BOOL(param_name)                                                \
  if (name == #param_name) {                                                  \
    bool v;                                                                   \
    if (!string_util::lexical_cast(value.empty() ? "true" : value, &v))       \
      return util::StatusBuilder(util::StatusCode::kInvalidArgument, GTL_LOC) \
             << "cannot parse \"" << value << "\" as bool.";                  \
    message->set_##param_name(v);                                             \
    return util::OkStatus();                                                  \
  }

#define PARSE_ENUM(param_name, map_name)                                      \
  if (name == #param_name) {                                                  \
    const auto it = map_name.find(absl::AsciiStrToUpper(value));              \
    if (it == map_name.end())                                                 \
      return util::StatusBuilder(util::StatusCode::kInvalidArgument, GTL_LOC) \
             << "unknown enumeration value of \"" << value << "\" as "        \
             << #map_name;                                                    \
    message->set_##param_name(it->second);                                    \
    return util::OkStatus();                                                  \
  }

#define PRINT_PARAM(param_name) \
  os << "  " << #param_name << ": " << message.param_name() << "\n";

#define PRINT_REPEATED_STRING(param_name)    \
  for (const auto &v : message.param_name()) \
    os << "  " << #param_name << ": " << v << "\n";

#define PRINT_ENUM(param_name, map_name)               \
  const auto it = map_name.find(message.param_name()); \
  if (it == map_name.end())                            \
    os << "  " << #param_name << ": unknown\n";        \
  else                                                 \
    os << "  " << #param_name << ": " << it->second << "\n";


inline std::string PrintProto(const TrainerSpec &message,
                              absl::string_view name) {
  std::ostringstream os;

  os << name << " {\n";

  PRINT_REPEATED_STRING(input);
  PRINT_PARAM(input_format);
  PRINT_PARAM(model_prefix);

  static const std::map<TrainerSpec::ModelType, std::string> kModelType_Map = {
      {TrainerSpec::BPE, "BPE"},
      // for future extension
  };

  PRINT_ENUM(model_type, kModelType_Map);
  PRINT_PARAM(vocab_size);
  PRINT_PARAM(input_sentence_size);
  PRINT_PARAM(shuffle_input_sentence);
  PRINT_PARAM(num_threads);
  PRINT_PARAM(num_sub_iterations);
  PRINT_PARAM(max_discretepiece_length);
  PRINT_PARAM(vocabulary_output_piece_score);

  os << "}\n";

  return os.str();
}


#undef PARSE_STRING
#undef PARSE_REPEATED_STRING
#undef PARSE_BOOL
#undef PARSE_BYTE
#undef PARSE_INT32
#undef PARSE_DUOBLE
#undef PARSE_ENUM
#undef PRINT_MAP
#undef PRINT_REPEATED_STRING
#undef PRINT_ENUM
}  // namespace discretepiece

#endif  // SPEC_PARSER_H_
