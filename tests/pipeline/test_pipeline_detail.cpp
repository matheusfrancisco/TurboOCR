#include <catch_amalgamated.hpp>

#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "../../src/pipeline/ocr/ocr_pipeline_detail.h"

using turbo_ocr::Box;
using turbo_ocr::pipeline::OcrPipelineResult;
using turbo_ocr::pipeline::detail::combine_recognition;
using turbo_ocr::pipeline::detail::flag_dropped_crops;
using turbo_ocr::pipeline::detail::flag_text_degraded;

namespace {

Box box_at(int x) {
  Box b{};
  b[0] = {x, 0};
  b[1] = {x + 10, 0};
  b[2] = {x + 10, 10};
  b[3] = {x, 10};
  return b;
}

} // namespace

TEST_CASE("combine_recognition keeps confident non-empty results in box order",
          "[pipeline_detail]") {
  std::vector<Box> boxes{box_at(0), box_at(20), box_at(40)};
  std::vector<std::pair<std::string, float>> rec{
      {"first", 0.9f}, {"second", 0.8f}, {"third", 0.7f}};
  OcrPipelineResult out;
  combine_recognition(out, boxes, rec);
  REQUIRE(out.results.size() == 3);
  CHECK(out.results[0].text == "first");
  CHECK(out.results[1].text == "second");
  CHECK(out.results[2].text == "third");
  CHECK(out.results[1].box[0][0] == 20);
  CHECK_FALSE(out.text_degraded);
}

TEST_CASE("combine_recognition drops empty and low-confidence entries",
          "[pipeline_detail]") {
  std::vector<Box> boxes{box_at(0), box_at(20), box_at(40)};
  std::vector<std::pair<std::string, float>> rec{
      {"keep", 0.9f}, {"", 0.9f}, {"low", 0.1f}};
  OcrPipelineResult out;
  combine_recognition(out, boxes, rec);
  REQUIRE(out.results.size() == 1);
  CHECK(out.results[0].text == "keep");
}

TEST_CASE("combine_recognition places words only for the lines it keeps",
          "[pipeline_detail]") {
  // Two lines of ink on a white page; the middle entry is dropped.
  cv::Mat img(40, 200, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::rectangle(img, {3, 3}, {7, 7}, cv::Scalar(0, 0, 0), cv::FILLED);
  std::vector<Box> boxes{box_at(0), box_at(20), box_at(40)};
  std::vector<std::pair<std::string, float>> rec{
      {"a", 0.9f}, {"low", 0.1f}, {"c", 0.9f}};
  std::vector<std::vector<turbo_ocr::recognition::CtcWord>> words{
      {{"a", 0.9f, 0.5f, 0.5f}}, {{"low", 0.1f, 0.5f, 0.5f}}, {{"c", 0.9f, 0.5f, 0.5f}}};
  OcrPipelineResult out;
  combine_recognition(out, boxes, rec, &words, &img);
  REQUIRE(out.results.size() == 2);
  REQUIRE(out.results[0].words.size() == 1);
  CHECK(out.results[0].words[0].text == "a");
  REQUIRE(out.results[1].words.size() == 1);
  CHECK(out.results[1].words[0].text == "c");

  // Without the request the items carry no words at all.
  std::vector<std::pair<std::string, float>> rec2{{"a", 0.9f}};
  OcrPipelineResult plain;
  combine_recognition(plain, {box_at(0)}, rec2);
  REQUIRE(plain.results.size() == 1);
  CHECK(plain.results[0].words.empty());
}

TEST_CASE("combine_recognition places the words of a long page exactly as line by line",
          "[pipeline_detail]") {
  // 240 printed lines: enough to be split over threads. Every third line is
  // dropped, so result and box indices diverge.
  constexpr int kLines = 240;
  cv::Mat img(kLines * 24 + 20, 700, CV_8UC3, cv::Scalar(255, 255, 255));
  std::vector<Box> boxes;
  std::vector<std::pair<std::string, float>> rec;
  std::vector<std::vector<turbo_ocr::recognition::CtcWord>> words;
  for (int i = 0; i < kLines; ++i) {
    const int y = 10 + i * 24;
    cv::putText(img, "alpha beta gamma " + std::to_string(i), {10, y + 17},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    Box b{};
    b[0] = {6, y};
    b[1] = {330, y};
    b[2] = {330, y + 22};
    b[3] = {6, y + 22};
    boxes.push_back(b);
    rec.push_back({"alpha beta gamma " + std::to_string(i), i % 3 == 1 ? 0.1f : 0.9f});
    words.push_back({{"alpha", 0.9f, 0.03f, 0.2f}, {"beta", 0.9f, 0.3f, 0.43f},
                     {"gamma", 0.9f, 0.52f, 0.72f},
                     {std::to_string(i), 0.9f, 0.82f, 0.95f}});
  }
  OcrPipelineResult out;
  auto rec_copy = rec;
  combine_recognition(out, boxes, rec_copy, &words, &img);

  std::size_t k = 0;
  for (int i = 0; i < kLines; ++i) {
    if (rec[i].second < turbo_ocr::kDropScore) continue;
    REQUIRE(k < out.results.size());
    const auto expected = turbo_ocr::recognition::locate_words(img, boxes[i], words[i], boxes);
    const auto &got = out.results[k++].words;
    INFO("line " << i);
    REQUIRE(got.size() == expected.size());
    for (std::size_t w = 0; w < got.size(); ++w) {
      CHECK(got[w].text == expected[w].text);
      CHECK(got[w].box == expected[w].box);
    }
  }
  CHECK(k == out.results.size());
  CHECK(k == 160);
}

namespace {

// A page of `n` printed lines of the given type size, with the recognizer
// output that goes with it.
struct PrintedPage {
  cv::Mat img;
  std::vector<Box> boxes;
  std::vector<std::pair<std::string, float>> rec;
  std::vector<std::vector<turbo_ocr::recognition::CtcWord>> words;
};
PrintedPage printed_page(int n, double scale) {
  const int lh = static_cast<int>(40 * scale), lw = static_cast<int>(560 * scale);
  PrintedPage p;
  p.img = cv::Mat(n * lh + 2 * lh, lw + 60, CV_8UC3, cv::Scalar(255, 255, 255));
  for (int i = 0; i < n; ++i) {
    const int y = lh / 2 + i * lh;
    cv::putText(p.img, "alpha beta gamma " + std::to_string(i),
                {12, y + static_cast<int>(28 * scale)}, cv::FONT_HERSHEY_SIMPLEX,
                scale, cv::Scalar(0, 0, 0), std::max(1, static_cast<int>(2 * scale)),
                cv::LINE_AA);
    Box b{};
    b[0] = {6, y};
    b[1] = {lw, y};
    b[2] = {lw, y + static_cast<int>(36 * scale)};
    b[3] = {6, y + static_cast<int>(36 * scale)};
    p.boxes.push_back(b);
    p.rec.push_back({"alpha beta gamma " + std::to_string(i), 0.9f});
    p.words.push_back({{"alpha", 0.9f, 0.03f, 0.2f}, {"beta", 0.9f, 0.3f, 0.43f},
                       {"gamma", 0.9f, 0.52f, 0.72f},
                       {std::to_string(i), 0.9f, 0.82f, 0.95f}});
  }
  return p;
}

void check_same_words(const OcrPipelineResult &a, const OcrPipelineResult &b) {
  REQUIRE(a.results.size() == b.results.size());
  for (std::size_t i = 0; i < a.results.size(); ++i) {
    INFO("line " << i);
    REQUIRE(a.results[i].words.size() == b.results[i].words.size());
    for (std::size_t w = 0; w < a.results[i].words.size(); ++w) {
      CHECK(a.results[i].words[w].text == b.results[i].words[w].text);
      CHECK(a.results[i].words[w].box == b.results[i].words[w].box);
    }
  }
}

} // namespace

TEST_CASE("a few large lines are split by pixels and placed exactly as line by line",
          "[pipeline_detail]") {
  // Slide headlines: 12 lines, each far more pixels than a line of print.
  auto p = printed_page(12, 4.0);
  OcrPipelineResult out;
  auto rec = p.rec;
  combine_recognition(out, p.boxes, rec, &p.words, &p.img);
  REQUIRE(out.results.size() == 12);
  for (std::size_t i = 0; i < 12; ++i) {
    const auto expected = turbo_ocr::recognition::locate_words(p.img, p.boxes[i], p.words[i], p.boxes);
    INFO("line " << i);
    REQUIRE(out.results[i].words.size() == expected.size());
    for (std::size_t w = 0; w < expected.size(); ++w)
      CHECK(out.results[i].words[w].box == expected[w].box);
  }
}

TEST_CASE("deferred word boxes are the in-place ones, once the caller places them",
          "[pipeline_detail]") {
  auto p = printed_page(60, 1.0);
  auto rec_now = p.rec, rec_later = p.rec;
  OcrPipelineResult now, later;
  combine_recognition(now, p.boxes, rec_now, &p.words, &p.img);
  combine_recognition(later, p.boxes, rec_later, &p.words, &p.img, /*defer_words=*/true);
  REQUIRE(later.place_words);
  for (const auto &r : later.results) CHECK(r.words.empty());
  // The result owns what placing needs: the caller's page may be gone.
  p.img.release();
  const auto place = std::move(later.place_words);
  later.place_words = nullptr;
  place(later);
  check_same_words(now, later);

  // Pixels the result cannot own (a view of someone else's buffer) are
  // placed right away, never deferred.
  auto q = printed_page(10, 1.0);
  cv::Mat view(q.img.rows, q.img.cols, q.img.type(), q.img.data, q.img.step);
  auto rec_view = q.rec, rec_ref = q.rec;
  OcrPipelineResult from_view, ref;
  combine_recognition(from_view, q.boxes, rec_view, &q.words, &view, /*defer_words=*/true);
  CHECK_FALSE(from_view.place_words);
  combine_recognition(ref, q.boxes, rec_ref, &q.words, &q.img);
  check_same_words(ref, from_view);
}

TEST_CASE("combine_recognition tolerates a short rec vector", "[pipeline_detail]") {
  std::vector<Box> boxes{box_at(0), box_at(20)};
  std::vector<std::pair<std::string, float>> rec{{"only", 0.9f}};
  OcrPipelineResult out;
  combine_recognition(out, boxes, rec);
  REQUIRE(out.results.size() == 1);
  CHECK(out.results[0].text == "only");
}

TEST_CASE("all-empty recognition on detected boxes flags text_degraded",
          "[pipeline_detail]") {
  std::vector<Box> boxes{box_at(0), box_at(20)};
  std::vector<std::pair<std::string, float>> rec{{"", 0.0f}, {"", 0.0f}};
  OcrPipelineResult out;
  combine_recognition(out, boxes, rec);
  CHECK(out.results.empty());
  CHECK(out.text_degraded);
  CHECK_FALSE(out.text_warning.empty());
}

TEST_CASE("zero detections is a clean page, not a degraded one",
          "[pipeline_detail]") {
  std::vector<Box> boxes;
  std::vector<std::pair<std::string, float>> rec;
  OcrPipelineResult out;
  combine_recognition(out, boxes, rec);
  CHECK_FALSE(out.text_degraded);
}

TEST_CASE("flag_dropped_crops marks partial drops loud", "[pipeline_detail]") {
  OcrPipelineResult out;
  out.results.push_back({.text = "survivor", .confidence = 0.9f, .box = box_at(0)});
  flag_dropped_crops(out, 3);
  CHECK(out.text_degraded);
  CHECK(out.text_warning.find("3") != std::string::npos);
  // Zero drops must not touch the flags.
  OcrPipelineResult clean;
  flag_dropped_crops(clean, 0);
  CHECK_FALSE(clean.text_degraded);
}

TEST_CASE("flag_dropped_crops appends to an existing warning", "[pipeline_detail]") {
  OcrPipelineResult out;
  flag_text_degraded(out, 2); // no results + 2 boxes -> sets first warning
  flag_dropped_crops(out, 1);
  CHECK(out.text_degraded);
  CHECK(out.text_warning.find(';') != std::string::npos);
}

TEST_CASE("match-driven assembly retains low confidence nonempty text",
          "[pipeline_detail][device_matches]") {
  std::vector<Box> boxes{box_at(0), box_at(20), box_at(40)};
  std::vector<std::pair<std::string,float>> rec{{"possible",0.1f},{"",0.9f},{"known",0.99f}};
  OcrPipelineResult out;
  combine_recognition(out,boxes,rec,nullptr,nullptr,false,nullptr,0.0f);
  REQUIRE(out.results.size()==2);
  CHECK(out.results[0].text=="possible");
  CHECK(out.results[0].confidence==0.1f);
  CHECK(out.results[1].text=="known");
}
