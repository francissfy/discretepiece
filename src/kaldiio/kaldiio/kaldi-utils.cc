// kaldi-utils.cc
//
// This file is copied/modified from
// https://github.com/kaldi-asr/kaldi/blob/master/src/base/kaldi-utils.cc

// Copyright 2009-2011   Karel Vesely;  Yanmin Qian;  Microsoft Corporation

#include "kaldi-utils.h"

namespace kaldiio {

std::string CharToString(const char &c) {
  char buf[20];
  if (std::isprint(c))
    std::snprintf(buf, sizeof(buf), "\'%c\'", c);
  else
    std::snprintf(buf, sizeof(buf), "[character %d]", static_cast<int>(c));
  return buf;
}

}  // namespace kaldiio
