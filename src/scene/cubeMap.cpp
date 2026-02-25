#include "cubeMap.h"
#include "../scene/material.h"
#include "../ui/TraceUI.h"
#include "ray.h"

#include <algorithm> // std::max, std::min
#include <cmath>     // std::abs
extern TraceUI *traceUI;

glm::dvec3 CubeMap::getColor(ray r) const {
  // YOUR CODE HERE
  glm::dvec3 d = glm::normalize(r.getDirection());

  const double ax = std::abs(d.x);
  const double ay = std::abs(d.y);
  const double az = std::abs(d.z);

  int face = 0;
  double u = 0.0, v = 0.0;
  double ma = 1.0; // major axis abs value

  // Choose major axis to pick face
  if (ax >= ay && ax >= az) {
    ma = ax;
    if (d.x > 0) { // +X (index 0)
      face = 0;
      u = (d.z / ma + 1.0) * 0.5;
      v = (d.y / ma + 1.0) * 0.5;
    } else {       // -X (index 1)
      face = 1;
      u = (-d.z / ma + 1.0) * 0.5;
      v = (d.y / ma + 1.0) * 0.5;
    }
  } else if (ay >= ax && ay >= az) {
    ma = ay;
    if (d.y > 0) { // +Y (index 2)
      face = 2;
      u = ( d.x / ma + 1.0) * 0.5;
      v = ( d.z / ma + 1.0) * 0.5;
    } else {       // -Y (index 3)
      face = 3;
      u = ( d.x / ma + 1.0) * 0.5;
      v = (-d.z / ma + 1.0) * 0.5;
    }
  } else {
    ma = az;
    if (d.z > 0) { // +Z direction samples -Z image (index 5)
      face = 5;
      u = (-d.x / ma + 1.0) * 0.5;
      v = ( d.y / ma + 1.0) * 0.5;
    } else {       // -Z direction samples +Z image (index 4)
      face = 4;
      u = ( d.x / ma + 1.0) * 0.5;
      v = ( d.y / ma + 1.0) * 0.5;
    }
  }

  if (!tMap[face]) return glm::dvec3(0.0);

  const int fw = std::max(1, traceUI ? traceUI->getFilterWidth() : 1);
  if (fw == 1) {
    return tMap[face]->getMappedValue(glm::dvec2(u, v));
  }

  const int w = tMap[face]->getWidth();
  const int h = tMap[face]->getHeight();
  if (w <= 1 || h <= 1) {
    return tMap[face]->getMappedValue(glm::dvec2(u, v));
  }

  const int half = fw / 2;
  glm::dvec3 sum(0.0);
  int count = 0;

  for (int dy = -half; dy <= half; ++dy) {
    for (int dx = -half; dx <= half; ++dx) {
      const double uu = std::clamp(u + dx / double(w - 1), 0.0, 1.0);
      const double vv = std::clamp(v + dy / double(h - 1), 0.0, 1.0);
      sum += tMap[face]->getMappedValue(glm::dvec2(uu, vv));
      ++count;
    }
  }

  return sum / double(count);
}

CubeMap::CubeMap() {}

CubeMap::~CubeMap() {}

void CubeMap::setNthMap(int n, TextureMap *m) {
  if (m != tMap[n].get())
    tMap[n].reset(m);
}
