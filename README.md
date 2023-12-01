# discretepiece

This is a modified BPE training tool based on Google's sentencepiece for discrete token sequence. Currently BPE training and encoding have been implemented. The decoding process is very simple, the detokenized sequence can be recovered by looking up indicies in trained BPE vocabulary and joining all piece indices. 

## compilation
requirements: cmake, g++/clang++, gcc/clang
```sh
git clone https://github.com/francissfy/discretepiece.git
cd discretepiece
mkdir -p build && cd build
cmake .. # for custom compiler, specify CC and CXX env variable
make -j 8
# binaries are under build/src/{spm_train,spm_encode}
```

## training
spm_train arguments
```sh
./build/src/spm_train --help
# 
# discretepiece
# 
# Usage: ./build/src/spm_train [options] files
# 
#    --input (comma separated list of input sentences)  type: std::string default: ""
#    --input_format (Input format. Supported format is `text`.)  type: std::string default: ""
#    --model_prefix (output model prefix)  type: std::string default: ""
#    --model_type (model algorithm: bpe)  type: std::string default: "bpe"
#    --vocab_size (vocabulary size)  type: int32 default: 8000
#    --input_sentence_size (maximum size of sentences the trainer loads)  type: std::uint64_t default: 0
#    --shuffle_input_sentence (Randomly sample input sentences in advance. Valid when --input_sentence_size > 0)  type: bool default: true
#    --num_threads (number of threads for training)  type: int32 default: 16
#    --num_sub_iterations (number of EM sub-iterations)  type: int32 default: 2
#    --max_discretepiece_length (maximum length of sentence piece)  type: int32 default: 16
#    --vocabulary_output_piece_score (Define score in vocab file)  type: bool default: true
#    --random_seed (Seed value for random generator.)  type: uint32 default: 4294967295
#    --help (show help)  type: bool default: false
#    --version (show version)  type: bool default: false
#    --minloglevel (Messages logged at a lower level than this don't actually get logged anywhere)  type: int default: 0

```
The input to spm_train is a text file, containing token sequence seperated by space
```text
1 2 3 4 5 ...
6 7 8 9 10 ...
```
Train BPE
```sh
../build/src/spm_train \
    --input "input file" \
    --model_prefix "output_model_prefix" \
    --shuffle_input_sentence false
```
This will produce two files: output_model_prefix.model and output_model_prefix.vocab. The output_model_prefix.vocab file is like:
```text
...
218_1353	-394
1782_293	-395
101_926	-396
155_1965	-397
655_1373	-398
670_201	-399
...
```

## encode
spm_encode arguments
```sh
./build/src/spm_encode --help
# discretepiece
# 
# Usage: ./build/src/spm_encode [options] files
# 
#    --model (model file name)  type: std::string default: ""
#    --output_format (choose from piece or id)  type: std::string default: "piece"
#    --input (input filename)  type: std::string default: ""
#    --output (output filename)  type: std::string default: ""
#    --help (show help)  type: bool default: false
#    --version (show version)  type: bool default: false
#    --minloglevel (Messages logged at a lower level than this don't actually get logged anywhere)  type: int default: 0
```
Input file to spm_encode has the same format as spm_train. spm_encode can encode token sequence into two formats: "id" and "piece", the former just encodes token sequence into BPE token sequence. The latter shows the exact token pattern encoded by BPE (piece is tokens joined with '_').
```sh
# token input
1222 1163 1525 1265 983 1532 1532 1145 1188 1333

# id output
./build/src/spm_encode \
    --model ".../trained.model" \
    --output_format "id" \
    --input "input file" \
    --output "output_file"
# id output
2776 6185 2215 197

# piece output
./build/src/spm_encode \
    --model ".../trained.model" \
    --output_format "piece" \
    --input "input file" \
    --output "output_file"
# piece output
1222_1163_1525_1265 983 1532_1532_1145 1188_1333
```

## verification
The correctness of spm_train and spm_encode are verified with original Google sentence piece on a sample test set containing 1000 token sequences. Testing vocabulary size: 8000.

## TODO
- implement word boundary (done, not tested)
