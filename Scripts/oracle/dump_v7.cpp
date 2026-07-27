// Oracle tool: dump KataGo V7 NN input features for a position as JSON.
// Usage: dump_v7 SGFFILE MOVENUM [RULES] [NNLEN]
// Prints {"hash": "...", "spatialNHWC": [...], "global": [...]} using the reference implementation.
#include <cstdio>
#include <string>
#include <vector>

// Include paths are relative to $KG/cpp (passed via -I by build-dumper.sh).
#include "dataio/sgf.h"
#include "game/board.h"
#include "game/boardhistory.h"
#include "neuralnet/nninputs.h"

int main(int argc, char* argv[]) {
  if(argc < 3) {
    fprintf(stderr, "usage: dump_v7 SGFFILE MOVENUM [RULES] [NNLEN]\n");
    return 1;
  }
  Board::initHash();
  std::string sgfFile = argv[1];
  int moveNum = atoi(argv[2]);
  Rules rules = Rules::getTrompTaylorish();
  if(argc >= 4)
    rules = Rules::parseRules(argv[3]);
  int nnLen = 19;
  if(argc >= 5)
    nnLen = atoi(argv[4]);

  std::unique_ptr<CompactSgf> sgf = CompactSgf::loadFile(sgfFile);
  Board board;
  Player nextPla;
  BoardHistory hist;
  sgf->setupInitialBoardAndHist(rules, board, nextPla, hist);
  sgf->playMovesTolerant(board, nextPla, hist, moveNum, false);

  MiscNNInputParams params;
  int numSpatial = NNInputs::NUM_FEATURES_SPATIAL_V7;
  int numGlobal = NNInputs::NUM_FEATURES_GLOBAL_V7;
  std::vector<float> rowBin((size_t)numSpatial * nnLen * nnLen, 0.0f);
  std::vector<float> rowGlobal(numGlobal, 0.0f);
  bool useNHWC = true;
  NNInputs::fillRowV7(board, hist, nextPla, params, nnLen, nnLen, useNHWC, rowBin.data(), rowGlobal.data());

  printf("{\"hash\":\"%s\",\"nextPla\":%d,\"nnLen\":%d,\"numSpatial\":%d,\"numGlobal\":%d,\"spatialNHWC\":[", board.pos_hash.toString().c_str(), (int)nextPla, nnLen, numSpatial, numGlobal);
  for(size_t i = 0; i < rowBin.size(); i++)
    printf(i == 0 ? "%.9g" : ",%.9g", rowBin[i]);
  printf("],\"global\":[");
  for(size_t i = 0; i < rowGlobal.size(); i++)
    printf(i == 0 ? "%.9g" : ",%.9g", rowGlobal[i]);
  printf("]}\n");
  return 0;
}
