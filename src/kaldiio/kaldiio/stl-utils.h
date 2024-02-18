// kaldi_native_io/csrc/stl-utils.h
//
// This file is copied/modified from
// https://github.com/kaldi-asr/kaldi/blob/master/src/util/stl-utils.h

// Copyright 2009-2011  Microsoft Corporation;  Saarland University

#ifndef KALDI_NATIVE_IO_CSRC_STL_UTILS_H_
#define KALDI_NATIVE_IO_CSRC_STL_UTILS_H_
#include <string>
namespace kaldiio {

/// A hashing function object for strings.
struct StringHasher {  // hashing function for std::string
  size_t operator()(const std::string &str) const noexcept {
    size_t ans = 0, len = str.length();
    const char *c = str.c_str(), *end = c + len;
    for (; c != end; c++) {
      ans *= kPrime;
      ans += *c;
    }
    return ans;
  }

 private:
  static const int kPrime = 7853;
};

}  // namespace kaldiio

#endif  // KALDI_NATIVE_IO_CSRC_STL_UTILS_H_
