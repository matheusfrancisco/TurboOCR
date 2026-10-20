#include <catch_amalgamated.hpp>
#include <array>
#include <cmath>
#include <cstdlib>
#include "turbo_ocr/common/cv_geometry.h"

#include "turbo_ocr/detection/det_config.h"
#include "turbo_ocr/detection/det_postprocess.h"

using turbo_ocr::Box;
using turbo_ocr::detection::box_score_fast;
using turbo_ocr::detection::compute_det_resize;
using turbo_ocr::detection::effective_det_max_side;
using turbo_ocr::detection::extract_boxes_from_bitmap;
using turbo_ocr::detection::kDetResizeDefault;
using turbo_ocr::detection::read_det_resize;
using turbo_ocr::detection::region_to_box;
using turbo_ocr::detection::unclip_rect;

TEST_CASE("unclip_rect grows the rectangle by area * ratio / perimeter a side",
          "[det_postprocess]") {
  // 40x20: d = 1.5 * 800 / 120 = 10.
  for (const float angle : {0.0f, 30.0f}) {
    const cv::RotatedRect r({50.0f, 25.0f}, {40.0f, 20.0f}, angle);
    const cv::RotatedRect g = unclip_rect(r, 1.5f);
    CHECK(g.size.width == Catch::Approx(60.0f));
    CHECK(g.size.height == Catch::Approx(40.0f));
    CHECK(g.center.x == Catch::Approx(50.0f));
    CHECK(g.center.y == Catch::Approx(25.0f));
    CHECK(g.angle == Catch::Approx(angle));
  }
  // A degenerate (zero-area) rectangle does not grow.
  const cv::RotatedRect line({10.0f, 10.0f}, {30.0f, 0.0f}, 0.0f);
  CHECK(unclip_rect(line, 1.5f).size.height == Catch::Approx(0.0f));
}

TEST_CASE("region_to_box orders the corners and rounds after scaling back",
          "[det_postprocess]") {
  // No growth, corners at .4 px: on a half-size map they land on .8 px in the
  // original, which rounds up -- rounding before scaling would lose that.
  const cv::RotatedRect r({30.4f, 20.4f}, {20.0f, 10.0f}, 0.0f);
  const auto box = region_to_box(r, 0.0f, 5.0f, 0.5f, 0.5f, 1000, 1000);
  REQUIRE(box.has_value());
  CHECK((*box)[0] == std::array<int, 2>{41, 31});  // tl
  CHECK((*box)[1] == std::array<int, 2>{81, 31});  // tr
  CHECK((*box)[2] == std::array<int, 2>{81, 51});  // br
  CHECK((*box)[3] == std::array<int, 2>{41, 51});  // bl
  // Clamped to the image.
  const auto edge = region_to_box(r, 0.0f, 5.0f, 0.5f, 0.5f, 60, 1000);
  REQUIRE(edge.has_value());
  CHECK((*edge)[1][0] == 59);
}

TEST_CASE("region_to_box rejects what is too thin to be text", "[det_postprocess]") {
  // Thinner than min_unclipped_side after the unclip.
  CHECK_FALSE(region_to_box({{50, 50}, {40, 4}, 0}, 0.0f, 5.0f, 1.0f, 1.0f, 100, 100));
  // Under 4 px a side once scaled to the original (a 2x-upsampled map).
  CHECK_FALSE(region_to_box({{50, 50}, {40, 6}, 0}, 0.0f, 5.0f, 2.0f, 2.0f, 100, 100));
  CHECK(region_to_box({{50, 50}, {40, 8}, 0}, 0.0f, 5.0f, 2.0f, 2.0f, 100, 100));
}

TEST_CASE("a region's box is its rectangle unclipped, not its outline", "[det_postprocess]") {
  // A text line whose top edge is jagged (every other column two rows
  // taller): its outline has much less area per perimeter than its
  // rectangle, and growing the outline gives a box that clips the line.
  cv::Mat bitmap = cv::Mat::zeros(60, 140, CV_8U);
  bitmap(cv::Rect(10, 20, 100, 10)).setTo(255);
  for (int x = 10; x < 110; x += 2) bitmap(cv::Rect(x, 18, 1, 2)).setTo(255);
  cv::Mat pred(bitmap.size(), CV_32F, cv::Scalar(0.0f));
  pred.setTo(0.9f, bitmap);

  std::vector<cv::Point> shifted;
  cv::Mat mask;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::Mat work = bitmap.clone();
  const auto boxes = extract_boxes_from_bitmap(pred, work, 60, 140, 60, 140, 0.4f, 1.4f, 3.0f,
                                               5.0f, shifted, mask, contours, hierarchy);
  REQUIRE(boxes.size() == 1);

  std::vector<cv::Point> px;
  cv::findNonZero(bitmap, px);
  const cv::RotatedRect rect = cv::minAreaRect(px);
  const float w = std::max(rect.size.width, rect.size.height);
  const float h = std::min(rect.size.width, rect.size.height);
  const float d = 1.4f * w * h / (2.0f * (w + h));
  const auto &b = boxes[0];
  CHECK(b[0][1] == static_cast<int>(std::round(rect.center.y - h / 2 - d)));
  CHECK(b[3][1] == static_cast<int>(std::round(rect.center.y + h / 2 + d)));
  CHECK(b[0][0] == static_cast<int>(std::round(rect.center.x - w / 2 - d)));
  CHECK(b[1][0] == static_cast<int>(std::round(rect.center.x + w / 2 + d)));
}

TEST_CASE("box_score_fast computes mean within polygon", "[det_postprocess]") {
  // Create a small prediction map filled with 0.8
  cv::Mat pred_map(100, 100, CV_32F, cv::Scalar(0.8f));

  // A rectangle covering part of the image
  std::vector<cv::Point> contour = {{20, 20}, {60, 20}, {60, 50}, {20, 50}};

  std::vector<cv::Point> shifted_buf;
  cv::Mat mask_buf;
  float score = box_score_fast(pred_map, contour, shifted_buf, mask_buf);

  // Should be approximately 0.8 (uniform fill)
  CHECK(score == Catch::Approx(0.8f).margin(0.01f));
}

TEST_CASE("box_score_fast returns zero for out-of-bounds contour", "[det_postprocess]") {
  cv::Mat pred_map(50, 50, CV_32F, cv::Scalar(0.9f));

  // Contour outside image bounds (negative coords clamped to 0)
  // All points at origin => xmax <= xmin => returns 0
  std::vector<cv::Point> contour = {{0, 0}, {0, 0}, {0, 0}};

  std::vector<cv::Point> shifted_buf;
  cv::Mat mask_buf;
  float score = box_score_fast(pred_map, contour, shifted_buf, mask_buf);

  CHECK(score == Catch::Approx(0.0f).margin(0.01f));
}

// Regression: DET_MAX_SIDE must clamp BOTH the engine-profile/buffer size
// (effective_det_max_side) AND the runtime resize cap (read_det_resize) so the
// resize output can never exceed the allocated buffer. A DET_MAX_SIDE below the
// model's max_side_limit used to size buffers at the smaller value while the
// resize still emitted up to max_side_limit px -> device overrun.
TEST_CASE("DET_MAX_SIDE shrinks both the buffer and the resize cap together", "[det_config]") {
  ::unsetenv("DET_MAX_SIDE");
  ::unsetenv("DET_MAX_SIDE_LIMIT");
  ::unsetenv("DET_LIMIT_TYPE");
  ::unsetenv("DET_LIMIT_SIDE_LEN");

  SECTION("shrink below max_side_limit") {
    ::setenv("DET_MAX_SIDE", "640", 1);
    auto p = read_det_resize();
    const int buf = effective_det_max_side(p);
    CHECK(p.max_side_limit == 640);  // resize cap shrank to 640
    CHECK(buf == 640);               // buffer/profile sized to 640
    // A large input must resize to <= the buffer side on every axis.
    auto [rh, rw] = compute_det_resize(4000, 3000, p);
    CHECK(std::max(rh, rw) <= buf);
    ::unsetenv("DET_MAX_SIDE");
  }

  SECTION("enlarge above max_side_limit leaves the resize cap untouched") {
    ::setenv("DET_MAX_SIDE", "2048", 1);
    auto p = read_det_resize();
    const int buf = effective_det_max_side(p);
    CHECK(p.max_side_limit == kDetResizeDefault.max_side_limit);  // unchanged (1280)
    CHECK(buf == 2048);  // buffer grows; resize stays <= 1280 < buf
    auto [rh, rw] = compute_det_resize(4000, 3000, p);
    CHECK(std::max(rh, rw) <= buf);
    ::unsetenv("DET_MAX_SIDE");
  }
}

// Regression (GitHub #23): DET_LIMIT_TYPE=max alone used to inherit the
// min-policy default limit_side_len=64, which under max semantics means
// "shrink the LONGEST side to 64px" — every image became a thumbnail and OCR
// silently returned zero results.
TEST_CASE("DET_LIMIT_TYPE=max without DET_LIMIT_SIDE_LEN targets the max-side cap",
          "[det_config]") {
  ::unsetenv("DET_MAX_SIDE");
  ::unsetenv("DET_MAX_SIDE_LIMIT");
  ::unsetenv("DET_LIMIT_TYPE");
  ::unsetenv("DET_LIMIT_SIDE_LEN");

  SECTION("bare max policy keeps native resolution up to the cap") {
    ::setenv("DET_LIMIT_TYPE", "max", 1);
    auto p = read_det_resize();
    CHECK(p.limit_side_len == p.max_side_limit);
    auto [rh, rw] = compute_det_resize(1000, 800, p);
    CHECK(std::max(rh, rw) >= 960);  // near-native, NOT a 64px thumbnail
    ::unsetenv("DET_LIMIT_TYPE");
  }

  SECTION("issue #23 env combo: max policy + DET_MAX_SIDE_LIMIT=2560") {
    ::setenv("DET_LIMIT_TYPE", "max", 1);
    ::setenv("DET_MAX_SIDE_LIMIT", "2560", 1);
    auto p = read_det_resize();
    CHECK(p.limit_side_len == 2560);
    auto [rh, rw] = compute_det_resize(4000, 3000, p);
    CHECK(std::max(rh, rw) == 2560);  // capped, not thumbnailed
    ::unsetenv("DET_LIMIT_TYPE");
    ::unsetenv("DET_MAX_SIDE_LIMIT");
  }

  SECTION("explicit DET_LIMIT_SIDE_LEN under max policy is honored") {
    ::setenv("DET_LIMIT_TYPE", "max", 1);
    ::setenv("DET_LIMIT_SIDE_LEN", "960", 1);
    auto p = read_det_resize();
    CHECK(p.limit_side_len == 960);
    auto [rh, rw] = compute_det_resize(4000, 3000, p);
    CHECK(std::max(rh, rw) == 960);
    ::unsetenv("DET_LIMIT_TYPE");
    ::unsetenv("DET_LIMIT_SIDE_LEN");
  }

  SECTION("garbage/zero numeric envs clamp instead of thumbnailing to 0") {
    ::setenv("DET_MAX_SIDE_LIMIT", "0", 1);
    ::setenv("DET_LIMIT_SIDE_LEN", "junk", 1);
    auto p = read_det_resize();
    CHECK(p.max_side_limit >= 32);
    CHECK(p.limit_side_len >= 32);
    ::unsetenv("DET_MAX_SIDE_LIMIT");
    ::unsetenv("DET_LIMIT_SIDE_LEN");
  }
}


TEST_CASE("filled contour scoring includes a hollow icon interior but retains text",
          "[det_postprocess][contour_score]") {
  cv::Mat pred(64, 96, CV_32F, cv::Scalar(.05f));
  // A high-score rim, with a low-score interior: foreground-only mean=.95.
  pred(cv::Rect(8, 8, 24, 24)).setTo(.95f);
  pred(cv::Rect(10, 10, 20, 20)).setTo(.05f);
  const std::vector<cv::Point> icon{{8,8},{31,8},{31,31},{8,31}};
  const std::vector<cv::Point> text{{48,12},{87,12},{87,21},{48,21}};
  pred(cv::Rect(48,12,40,10)).setTo(.95f);
  std::vector<cv::Point> shifted;
  cv::Mat mask;
  const float icon_score=box_score_fast(pred,icon,shifted,mask);
  CHECK(icon_score == Catch::Approx((176*.95+400*.05)/576).margin(1e-6));
  CHECK(icon_score < .75f);
  CHECK(box_score_fast(pred,text,shifted,mask) > .75f);

  cv::Mat bitmap;
  cv::compare(pred,.2,bitmap,cv::CMP_GT);
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  auto boxes=extract_boxes_from_bitmap(pred,bitmap,64,96,64,96,.75f,1.4f,3,5,
                                        shifted,mask,contours,hierarchy);
  REQUIRE(boxes.size()==1);
  CHECK(boxes[0][0][0] > 32); // only the solid text candidate survived
}

TEST_CASE("filled contour rasterization agrees with a full-image reference",
          "[det_postprocess][contour_score]") {
  std::vector<std::vector<cv::Point>> shapes{
    {{0,0},{12,0},{12,8},{0,8}}, // edge
    {{20,3},{29,13},{18,24},{9,14}}, // rotated
    {{2,25},{24,25},{24,30},{8,30},{8,45},{2,45}}, // concave
    {{35,3},{48,3},{48,16},{35,16}}, // translated
    {{55,45},{63,45},{63,49},{55,49}}, // tiny ID at image boundary
    {{-4,-3},{7,-3},{7,6},{-4,6}} // clipped
  };
  cv::Mat pred(50,64,CV_32F);
  cv::RNG rng(123);
  rng.fill(pred,cv::RNG::UNIFORM,0.0,1.0);
  std::vector<cv::Point> shifted;
  cv::Mat mask;
  for(const auto &contour:shapes) {
    cv::Mat reference=cv::Mat::zeros(pred.size(),CV_8UC1);
    cv::fillPoly(reference,std::vector<std::vector<cv::Point>>{contour},cv::Scalar(1));
    const auto roi=turbo_ocr::detection::filled_contour_mask(
        pred.cols,pred.rows,contour,shifted,mask);
    REQUIRE_FALSE(roi.empty());
    CHECK(cv::norm(reference(roi),mask,cv::NORM_INF)==0);
    CHECK(cv::countNonZero(reference)==cv::countNonZero(mask));
    CHECK(box_score_fast(pred,contour,shifted,mask)==
          Catch::Approx(cv::mean(pred,reference)[0]).margin(1e-6));
  }
  CHECK(turbo_ocr::detection::filled_contour_mask(64,50,{},shifted,mask).empty());
  CHECK(turbo_ocr::detection::filled_contour_mask(64,50,{{3,3},{3,3}},shifted,mask).empty());
  CHECK(turbo_ocr::detection::filled_contour_mask(0,50,shapes[0],shifted,mask).empty());
}
