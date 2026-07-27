// Oracle tool: dump raw KataGo NN outputs (logits, no softmax/tanh) plus the exact
// V7 input rows, at full float precision, using the reference Eigen backend.
// Usage: dump_nn MODELFILE SGFFILE MOVENUM [RULES] [NNLEN] [OPTIMISM]
// Output JSON:
// {hash, nextPla, modelVersion, nnLen, optimism,
//  spatialNHWC:[22*L*L], global:[G],
//  policy:[L*L+1]  (optimism-blended logits; pass logit last),
//  valueLogits:[3], scoreValues:[k], ownership:[L*L]}
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "core/config_parser.h"
#include "core/logger.h"
#include "dataio/sgf.h"
#include "game/board.h"
#include "game/boardhistory.h"
#include "neuralnet/modelversion.h"
#include "neuralnet/nneval.h"
#include "neuralnet/nninputs.h"
#include "neuralnet/nninterface.h"

static void printFloats(const char* key, const float* data, size_t n, bool trailingComma) {
  printf("\"%s\":[", key);
  for(size_t i = 0; i < n; i++)
    printf(i == 0 ? "%.9g" : ",%.9g", data[i]);
  printf("]%s", trailingComma ? "," : "");
}

int main(int argc, char* argv[]) {
  if(argc < 4) {
    fprintf(stderr, "usage: dump_nn MODELFILE SGFFILE MOVENUM [RULES] [NNLEN] [OPTIMISM]\n");
    return 1;
  }
  Board::initHash();
  std::string modelFile = argv[1];
  std::string sgfFile = argv[2];
  int moveNum = atoi(argv[3]);
  Rules rules = Rules::getTrompTaylorish();
  if(argc >= 5)
    rules = Rules::parseRules(argv[4]);
  int nnLen = 19;
  if(argc >= 6)
    nnLen = atoi(argv[5]);
  double optimism = 0.0;
  if(argc >= 7)
    optimism = atof(argv[6]);

  std::unique_ptr<CompactSgf> sgf = CompactSgf::loadFile(sgfFile);
  Board board;
  Player nextPla;
  BoardHistory hist;
  sgf->setupInitialBoardAndHist(rules, board, nextPla, hist);
  sgf->playMovesTolerant(board, nextPla, hist, moveNum, false);

  NeuralNet::globalInitialize();
  LoadedModel* loadedModel = NeuralNet::loadModelFile(modelFile, "");
  const ModelDesc& desc = NeuralNet::getModelDesc(loadedModel);
  int modelVersion = desc.modelVersion;
  int numSpatial = NNModelVersion::getNumSpatialFeatures(modelVersion);
  int numGlobal = NNModelVersion::getNumGlobalFeatures(modelVersion);

  Logger logger(NULL, false, false, false);
  ConfigParser cfg;
  ComputeContext* ctx = NeuralNet::createComputeContext(
    {-1}, &logger, nnLen, nnLen, "", enabled_t::False, loadedModel, cfg);
  ComputeHandle* handle =
    NeuralNet::createComputeHandle(ctx, loadedModel, &logger, 1, true, true, -1, 0);
  InputBuffers* inputBuffers = NeuralNet::createInputBuffers(loadedModel, 1, nnLen, nnLen);

  NNResultBuf buf;
  buf.rowSpatialBuf.resize((size_t)numSpatial * nnLen * nnLen);
  buf.rowGlobalBuf.resize(numGlobal);
  buf.hasRowMeta = false;
  buf.symmetry = 0;
  buf.policyOptimism = optimism;
  buf.includeOwnerMap = true;

  MiscNNInputParams params;
  params.policyOptimism = optimism;
  bool useNHWC = true;
  NNInputs::fillRowV7(
    board, hist, nextPla, params, nnLen, nnLen, useNHWC, buf.rowSpatialBuf.data(),
    buf.rowGlobalBuf.data());

  std::vector<NNOutput*> outputs;
  NNOutput* out = new NNOutput();
  out->nnXLen = nnLen;
  out->nnYLen = nnLen;
  out->whiteOwnerMap = new float[(size_t)nnLen * nnLen];
  outputs.push_back(out);

  NNResultBuf* bufs[1] = {&buf};
  NeuralNet::getOutput(handle, inputBuffers, 1, bufs, outputs);

  int policySize = nnLen * nnLen + 1;
  int numScoreValues = desc.numScoreValueChannels;
  float valueLogits[3] = {out->whiteWinProb, out->whiteLossProb, out->whiteNoResultProb};
  std::vector<float> scoreValues;
  scoreValues.push_back(out->whiteScoreMean);
  scoreValues.push_back(out->whiteScoreMeanSq);
  scoreValues.push_back(out->whiteLead);
  scoreValues.push_back(out->varTimeLeft);
  if(numScoreValues >= 6) {
    scoreValues.push_back(out->shorttermWinlossError);
    scoreValues.push_back(out->shorttermScoreError);
  }

  printf("{\"hash\":\"%s\",\"nextPla\":%d,\"modelVersion\":%d,\"nnLen\":%d,\"optimism\":%.3f,",
         board.pos_hash.toString().c_str(), (int)nextPla, modelVersion, nnLen, optimism);
  printFloats("spatialNHWC", buf.rowSpatialBuf.data(), buf.rowSpatialBuf.size(), true);
  printFloats("global", buf.rowGlobalBuf.data(), buf.rowGlobalBuf.size(), true);
  printFloats("policy", out->policyProbs, policySize, true);
  printFloats("valueLogits", valueLogits, 3, true);
  printFloats("scoreValues", scoreValues.data(), scoreValues.size(), true);
  printFloats("ownership", out->whiteOwnerMap, (size_t)nnLen * nnLen, false);
  printf("}\n");

  delete out;
  NeuralNet::freeInputBuffers(inputBuffers);
  NeuralNet::freeComputeHandle(handle);
  NeuralNet::freeComputeContext(ctx);
  NeuralNet::freeLoadedModel(loadedModel);
  NeuralNet::globalCleanup();
  return 0;
}
