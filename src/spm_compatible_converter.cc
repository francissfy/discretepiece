#include "init.h"
#include "util.h"
#include "common.h"
#include "filesystem.h"
#include "sentencepiece_model.pb.h"
#include "discretepiece_model.pb.h"

#include "third_party/absl/memory/memory.h"
#include "third_party/absl/flags/flag.h"


ABSL_FLAG(std::string, input_model, "", "model file to convert");
ABSL_FLAG(std::string, output_model_prefix, "", "model file after convert");

int main(int argc, char *argv[]) {
  discretepiece::ScopedResourceDestructor cleaner;
  discretepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);

  // check arguments
  CHECK(!absl::GetFlag(FLAGS_input_model).empty()) << "--input_model should not be empty";
  CHECK(!absl::GetFlag(FLAGS_output_model_prefix).empty()) << "--output_model_prefix should not be empty";

  // load old string type bpe model 
  auto old_model_proto = absl::make_unique<sentencepiece::ModelProto>();
  {
    auto input = discretepiece::filesystem::NewReadableFile(absl::GetFlag(FLAGS_input_model), true);
    CHECK(input->status().ok()) << "cannot open input model file: " << absl::GetFlag(FLAGS_input_model);

    std::string serialized;
    CHECK(input->ReadAll(&serialized)) << "error when reading input model file: " << absl::GetFlag(FLAGS_input_model);
    CHECK(old_model_proto->ParseFromArray(serialized.data(), serialized.size())) << "error parsing input model file: " << absl::GetFlag(FLAGS_input_model);
  }

  // compose new bpe model
  discretepiece::ModelProto new_model_proto;

  for (auto p_idx=0; p_idx<old_model_proto->pieces_size(); p_idx++) {
    const auto &old_piece = old_model_proto->pieces(p_idx);

    if (old_piece.type() != sentencepiece::ModelProto::SentencePiece::Type::ModelProto_SentencePiece_Type_NORMAL)
      continue;

    float old_piece_score = old_piece.score();
    const std::string &old_piece_str = old_piece.piece();

    const std::vector<char32> &old_piece_codepoints = discretepiece::string_util::UTF8ToUnicodeText(old_piece_str);
    std::vector<char32> old_piece_ids(old_piece_codepoints.size());
    std::transform(old_piece_codepoints.begin(), old_piece_codepoints.end(), old_piece_ids.begin(), [](char32 s) { return s-19968; });
    
    auto *sp = new_model_proto.add_pieces();
    sp->set_type(discretepiece::ModelProto::DiscretePiece::NORMAL);
    sp->set_piece(discretepiece::string_util::VectorChar32ToString(old_piece_ids, "_"));
    sp->set_score(old_piece_score);
  }

  // save new model and vocab
  {
    auto output = discretepiece::filesystem::NewWritableFile(absl::GetFlag(FLAGS_output_model_prefix) + ".model", true);
    CHECK(output->status().ok()) << "error opening output model file: " << absl::GetFlag(FLAGS_output_model_prefix) + ".model";
    output->Write(new_model_proto.SerializeAsString());    
  }
  {
    auto output = discretepiece::filesystem::NewWritableFile(absl::GetFlag(FLAGS_output_model_prefix) + ".vocab", true);
    CHECK(output->status().ok()) << "error opening output vocab file: " << absl::GetFlag(FLAGS_output_model_prefix) + ".vocab";

    for (const auto &piece: new_model_proto.pieces()) {
      std::ostringstream os;
      os << piece.piece() << "\t" << piece.score();
      CHECK(output->WriteLine(os.str())) << "error writing piece: " << os.str();
    }
  }

  return 0;
}
