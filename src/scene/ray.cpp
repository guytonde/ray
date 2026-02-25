#include "ray.h"
#include "../ui/TraceUI.h"
#include "material.h"
#include "scene.h"
#include <algorithm>
#include <iterator>


const Material &isect::getMaterial() const {
  return material ? *material : obj->getMaterial();
}

ray::ray(const glm::dvec3 &pp, const glm::dvec3 &dd, const glm::dvec3 &w,
         RayType tt)
    : p(pp), d(dd), atten(w), t(tt) {
  TraceUI::addRay(ray_thread_id);
}

ray::ray(const ray &other)
    : p(other.p), d(other.d), atten(other.atten), t(other.t),
      media(other.media) {
  TraceUI::addRay(ray_thread_id);
}

ray::~ray() {}

ray &ray::operator=(const ray &other) {
  p = other.p;
  d = other.d;
  atten = other.atten;
  t = other.t;
  media = other.media;
  return *this;
}

glm::dvec3 ray::at(const isect &i) const { return at(i.getT()); }

double ray::currentMediumIor() const {
  if (media.empty()) {
    return 1.0;
  }
  return media.back().ior;
}

bool ray::containsMedium(const SceneObject *obj) const {
  if (obj == nullptr) {
    return false;
  }
  for (const auto &entry : media) {
    if (entry.object == obj) {
      return true;
    }
  }
  return false;
}

int ray::mediumDepth() const { return static_cast<int>(media.size()); }

void ray::setMediaFrom(const ray &other) { media = other.media; }

void ray::pushMedium(const SceneObject *obj, double ior) {
  if (obj == nullptr || containsMedium(obj)) {
    return;
  }
  media.push_back({obj, std::max(1.0, ior)});
}

bool ray::removeMedium(const SceneObject *obj) {
  if (obj == nullptr) {
    return false;
  }
  for (auto it = media.rbegin(); it != media.rend(); ++it) {
    if (it->object == obj) {
      media.erase(std::next(it).base());
      return true;
    }
  }
  return false;
}

thread_local unsigned int ray_thread_id = 0;
