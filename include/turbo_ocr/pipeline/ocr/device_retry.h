#pragma once
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/recognition/ctc_decode.h"
#include <cmath>

namespace turbo_ocr::pipeline {
struct DeviceOcrInfo { float source_confidence = 0; int retries = 0; int reason = 0; };
struct DeviceOcrDiagnostics { std::vector<DeviceOcrInfo> readable, unreadable; };
struct DeviceOcrOptions { float minimum_confidence = 0.8f; unsigned retry_limit = 8; };

// Two independently cropped views must agree. Preserve a nonempty primary's
// text too; a retry may improve confidence/geometry, never erase a disagreement.
inline bool device_retry_agrees(const std::string &primary,
    const std::pair<std::string, float> &a, const std::pair<std::string, float> &b,
    const std::vector<recognition::CtcWord> &aw,
    const std::vector<recognition::CtcWord> &bw, float minimum) {
  if (a.first.empty() || a.first != b.first || (!primary.empty() && primary != a.first)) return false;
  if (aw.size()!=bw.size()) return false;
  for (std::size_t i=0;i<aw.size();++i) if (aw[i].text!=bw[i].text) return false;
  for (float c : {a.second, b.second})
    if (!std::isfinite(c) || c < minimum || c > 1) return false;
  for (const auto *words : {&aw, &bw}) {
    if (words->empty()) return false;
    for (const auto &w : *words)
      if (w.text.empty() || !std::isfinite(w.score) || w.score < minimum || w.score > 1 ||
          !std::isfinite(w.first) || !std::isfinite(w.last) || w.last < w.first) return false;
  }
  return true;
}

// UI retries are deliberately restricted to horizontal quads. Every output
// includes the original crop and stays in the same borrowed device image.
inline bool device_retry_boxes(const Box &box, int width, int height, Box &a, Box &b) {
  if (std::abs(box[0][1]-box[1][1]) > 2 || std::abs(box[2][1]-box[3][1]) > 2) return false;
  const auto r=aabb(box);
  if (r[0]<0 || r[1]<0 || r[2]>=width || r[3]>=height || r[2]<=r[0] || r[3]<=r[1]) return false;
  const int pad=std::clamp((r[3]-r[1])/4,2,8);
  auto expanded=[&](int p) {
    const int x0=std::max(0,r[0]-p), y0=std::max(0,r[1]-p);
    const int x1=std::min(width-1,r[2]+p), y1=std::min(height-1,r[3]+p);
    return Box{{{{{x0,y0}},{{x1,y0}},{{x1,y1}},{{x0,y1}}}}};
  };
  a=expanded(pad); b=expanded(pad*2);
  return a!=box && a!=b;
}
} // namespace turbo_ocr::pipeline
