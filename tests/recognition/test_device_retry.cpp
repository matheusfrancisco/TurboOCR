#include <catch_amalgamated.hpp>
#include "turbo_ocr/pipeline/ocr/device_retry.h"
#include <limits>
using namespace turbo_ocr;
using namespace turbo_ocr::pipeline;
TEST_CASE("device retry cannot erase disagreement or uncertain recognition", "[device_retry]") {
  const std::pair<std::string,float> good{"TEST",.99f};
  std::vector<recognition::CtcWord> words{{"TEST",.98f,.1f,.9f}};
  CHECK(device_retry_agrees("",good,good,words,words,.8f));
  CHECK(device_retry_agrees("TEST",good,good,words,words,.8f));
  CHECK_FALSE(device_retry_agrees("OTHER",good,good,words,words,.8f));
  CHECK_FALSE(device_retry_agrees("",good,{"OTHER",.99f},words,words,.8f));
  CHECK_FALSE(device_retry_agrees("",{"",.99f},{"",.99f},words,words,.8f));
  CHECK_FALSE(device_retry_agrees("",good,{"TEST",.7f},words,words,.8f));
  CHECK_FALSE(device_retry_agrees("",good,good,{},words,.8f));
  auto split=words;split.push_back({"EXTRA",.99f,.9f,1.f});
  CHECK_FALSE(device_retry_agrees("",good,good,split,words,.8f));
  auto weak=words;weak[0].score=.7f;
  CHECK_FALSE(device_retry_agrees("",good,good,weak,words,.8f));
  weak[0].score=std::numeric_limits<float>::quiet_NaN();
  CHECK_FALSE(device_retry_agrees("",good,good,weak,words,.8f));
}
TEST_CASE("device retry expands bounded horizontal source regions only", "[device_retry]") {
  Box original{{{{{20,20}},{{80,20}},{{80,40}},{{20,40}}}}},a{},b{};
  REQUIRE(device_retry_boxes(original,100,80,a,b));
  CHECK(a!=b);
  for (const auto &box:{a,b}) {
    auto r=aabb(box);CHECK(r[0]<=20);CHECK(r[1]<=20);CHECK(r[2]>=80);CHECK(r[3]>=40);
    CHECK(r[0]>=0);CHECK(r[1]>=0);CHECK(r[2]<100);CHECK(r[3]<80);
  }
  original[1][1]=30;
  CHECK_FALSE(device_retry_boxes(original,100,80,a,b));
  Box full{{{{{0,0}},{{99,0}},{{99,79}},{{0,79}}}}};
  CHECK_FALSE(device_retry_boxes(full,100,80,a,b));
}
