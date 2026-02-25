#include <cmath>
#include <iostream>

#include "light.h"
#include <glm/glm.hpp>
#include <glm/gtx/io.hpp>

using namespace std;

double DirectionalLight::distanceAttenuation(const glm::dvec3 &) const {
  // distance to light is infinite, so f(di) goes to 0.  Return 1.
  return 1.0;
}

glm::dvec3 DirectionalLight::shadowAttenuation(const ray &r, 
                                               const glm::dvec3 &p,
                                               const glm::dvec3 &surface_normal) const {
  (void)r;
  const glm::dvec3 light_dir = glm::normalize(getDirection(p));
  const glm::dvec3 n =
      (glm::dot(surface_normal, light_dir) >= 0.0) ? surface_normal : -surface_normal;
  const double eps = RAY_EPSILON;

  ray shadow_ray(p + eps * n, light_dir, glm::dvec3(1.0), ray::SHADOW);
  isect hit;
  if (!getScene()->intersect(shadow_ray, hit)) {
    return glm::dvec3(1.0);
  }
  const glm::dvec3 kt = glm::clamp(hit.getMaterial().kt(hit), 0.0, 1.0);
  if (glm::length(kt) <= 1e-8) {
    return glm::dvec3(0.0);
  }
  return kt;
}

glm::dvec3 DirectionalLight::getColor() const { return color; }

glm::dvec3 DirectionalLight::getDirection(const glm::dvec3 &) const {
  return -orientation;
}

double PointLight::distanceAttenuation(const glm::dvec3 &P) const {
  // YOUR CODE HERE
  const double distance = glm::length(position - P);
  const double denom =
      constantTerm + linearTerm * distance + quadraticTerm * distance * distance;
  if (denom <= 1e-12) {
    return 1.0;
  }
  return glm::clamp(1.0 / denom, 0.0, 1.0);
}

glm::dvec3 PointLight::getColor() const { return color; }

glm::dvec3 PointLight::getDirection(const glm::dvec3 &P) const {
  return glm::normalize(position - P);
}

glm::dvec3 PointLight::shadowAttenuation(const ray &r, 
                                         const glm::dvec3 &p,
                                         const glm::dvec3 &surface_normal) const {
  (void)r;
  const glm::dvec3 light_vec = position - p;
  const double light_distance = glm::length(light_vec);
  if (light_distance <= RAY_EPSILON) {
    return glm::dvec3(1.0);
  }

  const glm::dvec3 light_dir = light_vec / light_distance;
  const glm::dvec3 n =
      (glm::dot(surface_normal, light_dir) >= 0.0) ? surface_normal : -surface_normal;
  const double eps = RAY_EPSILON;

  ray shadow_ray(p + eps * n, light_dir, glm::dvec3(1.0), ray::SHADOW);
  isect hit;
  if (!getScene()->intersect(shadow_ray, hit) || hit.getT() >= light_distance - eps) {
    return glm::dvec3(1.0);
  }
  const glm::dvec3 kt = glm::clamp(hit.getMaterial().kt(hit), 0.0, 1.0);
  if (glm::length(kt) <= 1e-8) {
    return glm::dvec3(0.0);
  }
  return kt;
}

#define VERBOSE 0
