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

#include "bpe_model.h"

#include <functional>
#include <memory>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "freelist.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "util.h"

namespace discretepiece {
namespace bpe {

Model::Model(const ModelProto &model_proto) {
  model_proto_ = &model_proto;
  InitializePieces();
}

Model::~Model() {}

EncodeResult Model::Encode(const std::vector<char32> &normalized) const {
  if (!status().ok() || normalized.empty()) {
    return {};
  }

  struct Symbol {
    int prev;     // prev index of this symbol. -1 for BOS.
    int next;     // next index of tihs symbol. -1 for EOS.
    bool freeze;  // this symbol is never be merged.
    std::vector<char32> piece;
  };

  struct SymbolPair {
    int left;     // left index of this pair
    int right;    // right index of this pair
    float score;  // score of this pair. large is better.
    size_t size;  // length of this piece
  };

  class SymbolPairComparator {
   public:
    const bool operator()(SymbolPair *h1, SymbolPair *h2) {
      return (h1->score < h2->score || (h1->score == h2->score && h1->left > h2->left));
    }
  };

  using Agenda = std::priority_queue<SymbolPair *, std::vector<SymbolPair *>, SymbolPairComparator>;
  Agenda agenda;
  std::vector<Symbol> symbols;
  symbols.reserve(normalized.size());

  // Pre-allocates SymbolPair for efficiency.
  constexpr size_t kPreallocateSymbolPairSize = 256;
  model::FreeList<SymbolPair> symbol_pair_allocator(kPreallocateSymbolPairSize);

  // Lookup new symbol pair at [left, right] and inserts it to agenda.
  auto MaybeAddNewSymbolPair = [this, &symbol_pair_allocator, &symbols, &agenda](int left, int right) {
    // in this, we can control sos & eos merging rules
    if (left == -1 || right == -1 || symbols[left].freeze || symbols[right].freeze)
      return;
    
    std::vector<char32> piece = symbols[left].piece;
    piece.insert(piece.end(), symbols[right].piece.begin(), symbols[right].piece.end());

    const auto it = pieces_.find(piece);
    if (it == pieces_.end()) {
      return; // current bigram not in piece
    }

    auto *h = symbol_pair_allocator.Allocate();
    h->left = left;
    h->right = right;
    h->score = GetScore(it->second);
    h->size = piece.size();
    agenda.push(h);
  };

  // Splits the input into character sequence
  // TODO: handling deliminator
  for (auto index=0; index<normalized.size(); index++) {
    Symbol s;
    s.piece.push_back(normalized[index]);
    s.prev = (index == 0? -1:index-1);
    s.next = (index == (normalized.size()-1)? -1:(index+1));
    s.freeze = false;
    symbols.emplace_back(s);
  }

  if (symbols.empty()) {
    return {};
  }

  // Lookup all bigrams.
  for (size_t i = 1; i < symbols.size(); ++i) {
    MaybeAddNewSymbolPair(i - 1, i);
  }

  // Main loop.
  while (!agenda.empty()) {
    SymbolPair *top = agenda.top();
    agenda.pop();

    // `top` is no longer available.
    if (symbols[top->left].piece.empty() || symbols[top->right].piece.empty() ||
        symbols[top->left].piece.size() + symbols[top->right].piece.size() != top->size) {
      continue;
    }

    // Replace `left` symbols with `top` rule.
    auto& right_piece = symbols[top->right].piece;
    auto& left_piece = symbols[top->left].piece;
    left_piece.insert(left_piece.end(), right_piece.begin(), right_piece.end());
    right_piece.clear();

    // Updates prev/next pointers.
    // eg.: [prev, left], [left, right], [right, next]
    // to: [prev, left], [left, next]
    symbols[top->left].next = symbols[top->right].next;
    if (symbols[top->right].next >= 0) {
      symbols[symbols[top->right].next].prev = top->left;
    }

    // Adds new symbol pairs which are newly added after symbol replacement.
    MaybeAddNewSymbolPair(symbols[top->left].prev, top->left);
    MaybeAddNewSymbolPair(top->left, symbols[top->left].next);
  }

  EncodeResult output;
  for (int index = 0; index != -1; index = symbols[index].next) {
    CHECK_GE(index, 0);
    CHECK_LT(index, static_cast<int>(symbols.size()));
    const auto& piece = symbols[index].piece;
    const int id = PieceToId(piece);
    output.emplace_back(piece, id);
  }

  return output;
}

}  // namespace bpe
}  // namespace discretepiece
