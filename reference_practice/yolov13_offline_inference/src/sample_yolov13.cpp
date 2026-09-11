/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <map>
#include "common/sampleDevice.h"
#include "common/sampleModel.h"
#include "common/utils.h"

using namespace std;
bool g_isDevice = false;

// COCO dataset 80-class labels.
static const char *kCocoLabels[] = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

// Detection result of a single object.
struct Detection {
  float x1, y1, x2, y2;
  float conf;
  int class_id;
};

static const int kInputW = 640;
static const int kInputH = 640;
static const int kNumAnchors = 8400;
static const int kNumClasses = 80;
static const float kConfThresh = 0.25f;
static const float kIouThresh = 0.45f;

// Compute Intersection over Union (IoU) between two bounding boxes.
static float ComputeIoU(const Detection &a, const Detection &b) {
  float inter_x1 = std::max(a.x1, b.x1);
  float inter_y1 = std::max(a.y1, b.y1);
  float inter_x2 = std::min(a.x2, b.x2);
  float inter_y2 = std::min(a.y2, b.y2);
  float inter_w = std::max(0.0f, inter_x2 - inter_x1);
  float inter_h = std::max(0.0f, inter_y2 - inter_y1);
  float inter = inter_w * inter_h;
  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (area_a + area_b - inter + 1e-6f);
}

// Non-Maximum Suppression (NMS): remove duplicate detections of the same class
// based on IoU overlap threshold.
static std::vector<Detection> Nms(std::vector<Detection> &dets, float iou_thresh) {
  // Sort by confidence descending.
  std::sort(dets.begin(), dets.end(), [](const Detection &a, const Detection &b) { return a.conf > b.conf; });
  std::vector<Detection> keep;
  std::vector<bool> suppressed(dets.size(), false);
  for (size_t i = 0; i < dets.size(); i++) {
    if (suppressed[i]) continue;
    keep.push_back(dets[i]);
    for (size_t j = i + 1; j < dets.size(); j++) {
      if (suppressed[j]) continue;
      if (dets[i].class_id == dets[j].class_id) {
        if (ComputeIoU(dets[i], dets[j]) > iou_thresh) {
          suppressed[j] = true;
        }
      }
    }
  }
  return keep;
}

// Parse YOLO raw output tensor (1, 84, 8400) into detection list.
// Layout: [cx, cy, w, h, 80 class scores] per anchor, 8400 anchors total.
// Bounding boxes are in [cx, cy, w, h] format and converted to [x1, y1, x2, y2].
static std::vector<Detection> ParseYoloOutput(float *data, int num_anchors, int num_classes, float conf_thresh,
                                              float input_w, float input_h) {
  std::vector<Detection> dets;
  for (int i = 0; i < num_anchors; i++) {
    float *box = data + i;
    float cx = box[0];                      // offset 0
    float cy = box[num_anchors];            // offset 1 * 8400
    float bw = box[num_anchors * 2];        // offset 2 * 8400
    float bh = box[num_anchors * 3];        // offset 3 * 8400
    float *scores = box + num_anchors * 4;  // offset 4 * 8400

    float max_conf = 0.0f;
    int max_cls = 0;
    for (int c = 0; c < num_classes; c++) {
      float score = scores[c * num_anchors];
      if (score > max_conf) {
        max_conf = score;
        max_cls = c;
      }
    }
    if (max_conf < conf_thresh) continue;

    float x1 = std::max(0.0f, cx - bw * 0.5f);
    float y1 = std::max(0.0f, cy - bh * 0.5f);
    float x2 = std::min(cx + bw * 0.5f, input_w);
    float y2 = std::min(cy + bh * 0.5f, input_h);

    Detection det;
    det.x1 = x1;
    det.y1 = y1;
    det.x2 = x2;
    det.y2 = y2;
    det.conf = max_conf;
    det.class_id = max_cls;
    dets.push_back(det);
  }
  return dets;
}

class SampleYolov13 {
 public:
  SampleYolov13() = default;
  ~SampleYolov13() = default;

  Result InitResource(const char *aclConfigPath);
  Result PrepareModel(const char *modelPath);
  Result Process(const std::vector<std::string> &inputFiles, int warmupRuns, int runs);

 private:
  void OutputResult(const std::vector<Detection> &dets);
  bool PostProcessOutput(const std::string &filePath);

  // Device resource.
  std::shared_ptr<AclInstance> aclInstance_;
  std::shared_ptr<AclDevice> device_;
  std::shared_ptr<AclContext> context_;
  std::shared_ptr<AclStream> stream_;
  // Model.
  std::shared_ptr<AclModelWeight> modelWeight_;
  std::shared_ptr<AclModelWork> modelWork_;
  std::shared_ptr<AclModelDesc> modelDesc_;
  std::shared_ptr<AclModelInput> modelInput_;
  std::shared_ptr<AclModelOutput> modelOutput_;
  uint32_t modelId_;
};

Result SampleYolov13::InitResource(const char *aclConfigPath) {
  // ACL init.
  aclInstance_ = std::make_shared<AclInstance>(aclConfigPath);
  INFO_LOG("acl init success");

  // Set device.
  device_ = std::make_shared<AclDevice>(0);
  INFO_LOG("set device success");

  // Create context (set current).
  context_ = std::make_shared<AclContext>(0);
  INFO_LOG("create context success");

  // Create stream.
  stream_ = std::make_shared<AclStream>();
  INFO_LOG("create stream success");

  // Get run mode.
  // runMode is ACL_HOST which represents app is running in host.
  // runMode is ACL_DEVICE which represents app is running in device.
  aclrtRunMode runMode;
  auto ret = aclrtGetRunMode(&runMode);
  if (ret != ACL_SUCCESS) {
    return FAILED;
  }
  g_isDevice = (runMode == ACL_DEVICE);
  return SUCCESS;
}

Result SampleYolov13::PrepareModel(const char *modelPath) {
  // Load model.
  size_t modelWorkSize, modelWeightSize;
  aclError ret = aclmdlQuerySize(modelPath, &modelWorkSize, &modelWeightSize);
  if (ret != ACL_SUCCESS) {
    return FAILED;
  }
  modelWork_ = std::make_shared<AclModelWork>(modelWorkSize);
  modelWeight_ = std::make_shared<AclModelWeight>(modelWeightSize);

  ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelWork_->GetModelWorkPtr(), modelWork_->GetModelWorkSize(),
                                   modelWeight_->GetModelWeightPtr(), modelWeight_->GetModelWeightSize());
  if (ret != ACL_SUCCESS) {
    return FAILED;
  }
  INFO_LOG("load model %s success.", modelPath);

  // Create ModelDesc.
  modelDesc_ = std::make_shared<AclModelDesc>(modelId_);
  return SUCCESS;
}

void SampleYolov13::OutputResult(const std::vector<Detection> &dets) {
  // Print the detected objects with class label, confidence, and bbox.
  int person_count = 0;
  int vehicle_count = 0;
  INFO_LOG("detected %zu objects:", dets.size());
  for (size_t i = 0; i < dets.size(); i++) {
    if (dets[i].class_id == 0) {
      ++person_count;
    }
    if (dets[i].class_id == 2 || dets[i].class_id == 3 || dets[i].class_id == 5 || dets[i].class_id == 7) {
      ++vehicle_count;
    }
    const char *label = (dets[i].class_id >= 0 && dets[i].class_id < 80) ? kCocoLabels[dets[i].class_id] : "unknown";
    INFO_LOG("  [%zu] %s (%.2f)  bbox: [%d,%d,%d,%d]", i, label, dets[i].conf, (int)dets[i].x1, (int)dets[i].y1,
             (int)dets[i].x2, (int)dets[i].y2);
  }
  INFO_LOG("recognized persons: %d, vehicles: %d", person_count, vehicle_count);
}

static void SaveResultFile(const std::string &binPath, const std::vector<Detection> &dets) {
  // Write detection results to a text file for visualization.
  std::string resultPath = binPath;
  size_t pos = resultPath.rfind(".bin");
  if (pos != std::string::npos) {
    resultPath = resultPath.substr(0, pos) + "_result.txt";
  } else {
    resultPath += "_result.txt";
  }
  std::ofstream ofs(resultPath);
  if (!ofs.is_open()) return;
  for (size_t i = 0; i < dets.size(); i++) {
    const char *label = (dets[i].class_id >= 0 && dets[i].class_id < 80) ? kCocoLabels[dets[i].class_id] : "unknown";
    ofs << label << " " << dets[i].conf << " " << (int)dets[i].x1 << " " << (int)dets[i].y1 << " " << (int)dets[i].x2
        << " " << (int)dets[i].y2 << "\n";
  }
  ofs.close();
  INFO_LOG("result saved to %s", resultPath.c_str());
}

bool SampleYolov13::PostProcessOutput(const std::string &filePath) {
  // Get model output data buffer.
  aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(modelOutput_->GetDataSet(), 0);
  if (dataBuffer == nullptr) {
    ERROR_LOG("model output data buffer is null");
    return false;
  }
  void *data = aclGetDataBufferAddr(dataBuffer);
  uint32_t len = aclGetDataBufferSizeV2(dataBuffer);
  const uint64_t expected_len = static_cast<uint64_t>(kNumAnchors) * (kNumClasses + 4) * sizeof(float);
  if (data == nullptr || len < expected_len) {
    ERROR_LOG("model output buffer is invalid, actual size=%u, expected at least=%llu", len,
              static_cast<unsigned long long>(expected_len));
    return false;
  }

  void *outHostData = nullptr;
  if (!g_isDevice) {
    // If app is running in host, copy model output data from device to host.
    aclError aclRet = aclrtMallocHost(&outHostData, len);
    if (aclRet != ACL_SUCCESS) {
      ERROR_LOG("aclrtMallocHost failed, len=%u, errorCode=%d", len, static_cast<int32_t>(aclRet));
      return false;
    }
    aclRet = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
      ERROR_LOG("aclrtMemcpy failed, errorCode=%d", static_cast<int32_t>(aclRet));
      (void)aclrtFreeHost(outHostData);
      return false;
    }
  }

  // Parse YOLO output, filter by confidence, and apply NMS.
  float *output = reinterpret_cast<float *>(g_isDevice ? data : outHostData);
  auto detections = ParseYoloOutput(output, kNumAnchors, kNumClasses, kConfThresh, kInputW, kInputH);
  auto nmsResults = Nms(detections, kIouThresh);
  OutputResult(nmsResults);
  SaveResultFile(filePath, nmsResults);

  if (!g_isDevice) {
    (void)aclrtFreeHost(outHostData);
  }
  return true;
}

Result SampleYolov13::Process(const std::vector<std::string> &inputFiles, int warmupRuns, int runs) {
  void *picDevBuffer = nullptr;
  size_t devBufferSize = aclmdlGetInputSizeByIndex(modelDesc_->GetModelDesc(), 0);
  aclError aclRet = aclrtMalloc(&picDevBuffer, devBufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
  if (aclRet != ACL_SUCCESS) {
    ERROR_LOG("malloc device buffer failed. size=%zu, errorCode=%d", devBufferSize, static_cast<int32_t>(aclRet));
    return FAILED;
  }

  modelInput_ = std::make_shared<AclModelInput>(picDevBuffer, devBufferSize, modelDesc_->GetModelDesc());
  modelOutput_ = std::make_shared<AclModelOutput>(modelDesc_->GetModelDesc());

  bool anySuccess = false;
  for (size_t idx = 0; idx < inputFiles.size(); idx++) {
    INFO_LOG("start to process file: %s", inputFiles[idx].c_str());

    // Copy image data from host file to device buffer.
    auto ret = Utils::MemcpyFileToDeviceBuffer(inputFiles[idx], picDevBuffer, devBufferSize, g_isDevice);
    if (ret != SUCCESS) {
      ERROR_LOG("memcpy device buffer failed, file: %s", inputFiles[idx].c_str());
      continue;
    }

    bool executeSuccess = true;
    for (int run = 0; run < warmupRuns; ++run) {
      aclRet = aclmdlExecute(modelId_, modelInput_->GetDataSet(), modelOutput_->GetDataSet());
      if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("aclmdlExecute failed during warmup, run=%d", run);
        executeSuccess = false;
        break;
      }
    }
    if (!executeSuccess) {
      continue;
    }

    long long totalLatencyUs = 0;
    int successfulRuns = 0;
    for (int run = 0; run < runs; ++run) {
      const auto executeStart = std::chrono::steady_clock::now();
      aclRet = aclmdlExecute(modelId_, modelInput_->GetDataSet(), modelOutput_->GetDataSet());
      const auto executeEnd = std::chrono::steady_clock::now();
      if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("aclmdlExecute failed, run=%d", run);
        executeSuccess = false;
        break;
      }
      const auto latencyUs = std::chrono::duration_cast<std::chrono::microseconds>(executeEnd - executeStart).count();
      totalLatencyUs += latencyUs;
      ++successfulRuns;
      INFO_LOG("BENCHMARK aclmdlExecute run=%d latency_us=%lld", run, static_cast<long long>(latencyUs));
    }
    if (successfulRuns > 0) {
      const double avgLatencyMs = static_cast<double>(totalLatencyUs) / successfulRuns / 1000.0;
      INFO_LOG("Average aclmdlExecute latency: %.3f ms", avgLatencyMs);
    }

    if (executeSuccess && PostProcessOutput(inputFiles[idx])) {
      anySuccess = true;
    }
  }

  aclrtFree(picDevBuffer);
  return anySuccess ? SUCCESS : FAILED;
}

struct Args {
  std::string modelPath{"../model/yolov13.om"};
  std::string inputPath{"../data/bus.bin"};
  int warmupRuns{0};
  int runs{1};
  bool help{false};
};

static void PrintHelpInfo() {
  std::cout << "Usage: ./yolov13_main [options]" << std::endl;
  std::cout << "  --model=<path>         OM model path (default: ../model/yolov13.om)" << std::endl;
  std::cout << "  --input=<path>         Input file path (default: ../data/bus.bin)" << std::endl;
  std::cout << "  --warmup-runs=<number> Number of warmup runs (default: 0)" << std::endl;
  std::cout << "  --runs=<number>        Number of measured runs (default: 1)" << std::endl;
  std::cout << "  --help                 Show this help message" << std::endl;
}

static bool ParseRunCount(const std::string &value, bool allowZero, int &result) {
  try {
    size_t parsedSize = 0U;
    const int parsedValue = std::stoi(value, &parsedSize);
    if (parsedSize != value.size() || parsedValue < (allowZero ? 0 : 1)) {
      return false;
    }
    result = parsedValue;
    return true;
  } catch (...) {
    return false;
  }
}

enum class OptionParseResult { kNotMatched, kMatched, kMissingValue };

static OptionParseResult ParseOptionValue(const std::vector<std::string> &arguments, size_t &index,
                                          const std::string &longOption, const std::string &shortOption,
                                          std::string &value) {
  const std::string &argument = arguments[index];
  if (argument == longOption || argument == shortOption) {
    if (++index >= arguments.size()) {
      return OptionParseResult::kMissingValue;
    }
    value = arguments[index];
    return OptionParseResult::kMatched;
  }
  const std::string longOptionPrefix = longOption + "=";
  if (argument.compare(0, longOptionPrefix.size(), longOptionPrefix) == 0) {
    value = argument.substr(longOptionPrefix.size());
    return OptionParseResult::kMatched;
  }
  if (argument.compare(0, shortOption.size(), shortOption) == 0 && argument.size() > shortOption.size()) {
    value = argument.substr(shortOption.size());
    return OptionParseResult::kMatched;
  }
  return OptionParseResult::kNotMatched;
}

static bool ParseArgs(const std::vector<std::string> &arguments, Args &args) {
  for (size_t index = 1U; index < arguments.size(); ++index) {
    if (arguments[index] == "--help" || arguments[index] == "-h") {
      args.help = true;
      return true;
    }
    std::string value;
    OptionParseResult result = ParseOptionValue(arguments, index, "--model", "-m", value);
    if (result != OptionParseResult::kNotMatched) {
      if (result == OptionParseResult::kMissingValue) {
        return false;
      }
      args.modelPath = value;
      continue;
    }
    result = ParseOptionValue(arguments, index, "--input", "-i", value);
    if (result != OptionParseResult::kNotMatched) {
      if (result == OptionParseResult::kMissingValue) {
        return false;
      }
      args.inputPath = value;
      continue;
    }
    result = ParseOptionValue(arguments, index, "--warmup-runs", "-w", value);
    if (result != OptionParseResult::kNotMatched) {
      if (result == OptionParseResult::kMissingValue || !ParseRunCount(value, true, args.warmupRuns)) {
        ERROR_LOG("invalid --warmup-runs value: %s", value.c_str());
        return false;
      }
      continue;
    }
    result = ParseOptionValue(arguments, index, "--runs", "-r", value);
    if (result != OptionParseResult::kNotMatched) {
      if (result == OptionParseResult::kMissingValue || !ParseRunCount(value, false, args.runs)) {
        ERROR_LOG("invalid --runs value: %s", value.c_str());
        return false;
      }
      continue;
    }
    ERROR_LOG("invalid command-line argument: %s", arguments[index].c_str());
    return false;
  }
  if (args.modelPath.empty() || args.inputPath.empty()) {
    ERROR_LOG("invalid command-line arguments");
    return false;
  }
  return true;
}

int main(int argc, char *argv[]) {
  Args args;
  std::vector<std::string> arguments(argv, argv + argc);
  if (!ParseArgs(arguments, args)) {
    return FAILED;
  }
  if (args.help) {
    PrintHelpInfo();
    return SUCCESS;
  }
  INFO_LOG("YOLOv13 SAMPLE start to execute.");

  // To better demonstrate the core usage of the acl interface, the sample encapsulates
  // the resource management within SampleYolov13, and the resource
  // release relies on the destructor of SampleYolov13. Therefore,
  // using curly braces to define the scope ensures that it is destructed and resources
  // are released before the process exits.
  {
    SampleYolov13 sample;
    const char *aclConfigPath = "../src/acl.json";
    Result ret = sample.InitResource(aclConfigPath);
    if (ret != SUCCESS) {
      ERROR_LOG("SAMPLE NOT PASSED: sample init resource failed.");
      return FAILED;
    }

    ret = sample.PrepareModel(args.modelPath.c_str());
    if (ret != SUCCESS) {
      ERROR_LOG("SAMPLE NOT PASSED: sample prepare model failed.");
      return FAILED;
    }

    std::vector<std::string> inputFiles = {args.inputPath};
    ret = sample.Process(inputFiles, args.warmupRuns, args.runs);
    if (ret != SUCCESS) {
      ERROR_LOG("SAMPLE NOT PASSED: sample process failed.");
      return FAILED;
    }
  }

  INFO_LOG("YOLOv13 SAMPLE PASSED.");
  return SUCCESS;
}
