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
                                               const glm::dvec3 &p) const {
  // YOUR CODE HERE:
  (void)p;
  const Scene* scene = this->getScene();
  ray shadowRay = r;     // copy -> now non-const
  isect hit;
  if (scene->intersect(shadowRay, hit))  // OK: intersect wants ray&
    return glm::dvec3(0.0);

  return glm::dvec3(1.0);
  // You should implement shadow-handling code here.
  // return glm::dvec3(1.0, 1.0, 1.0);
}

glm::dvec3 DirectionalLight::getColor() const { return color; }

glm::dvec3 DirectionalLight::getDirection(const glm::dvec3 &) const {
  return -orientation;
}

double PointLight::distanceAttenuation(const glm::dvec3 &P) const {
  // YOUR CODE HERE
  const double d = glm::length(position - P);
  double denom = constantTerm + linearTerm * d + quadraticTerm * d * d;
  if (denom < 1e-6) denom = 1e-6;
  const double fd = 1.0 / denom;
  return glm::clamp(fd, 0.0, 1.0);

  // You'll need to modify this method to attenuate the intensity
  // of the light based on the distance between the source and the
  // point P.  For now, we assume no attenuation and just return 1.0
  // return 1.0;
}

glm::dvec3 PointLight::getColor() const { return color; }

glm::dvec3 PointLight::getDirection(const glm::dvec3 &P) const {
  return glm::normalize(position - P);
}

glm::dvec3 PointLight::shadowAttenuation(const ray &r,
                                         const glm::dvec3 &p) const {
  // YOUR CODE HERE:
  (void)p;
  const Scene* scene = this->getScene();
  ray shadowRay = r;   // copy so we can pass ray& into intersect(...)
  isect hit;

  // If nothing is hit, light is unblocked
  if (!scene->intersect(shadowRay, hit)) {
    return glm::dvec3(1.0);
  }

  // If we hit something, it only blocks the point light if it's between
  // the shading point and the light position.
  const double distToLight = glm::length(position - shadowRay.getPosition());
  if (hit.getT() < distToLight - RAY_EPSILON) {
    return glm::dvec3(0.0);
  }

  return glm::dvec3(1.0);
  // You should implement shadow-handling code here.
  // return glm::dvec3(1, 1, 1);
}

#define VERBOSE 0

