#include "material.h"
#include "../ui/TraceUI.h"
#include "light.h"
#include "ray.h"
extern TraceUI *traceUI;

#include "../fileio/images.h"
#include <glm/gtx/io.hpp>
#include <iostream>

using namespace std;
extern bool debugMode;

Material::~Material() {}

// Apply the phong model to this point on the surface of the object, returning
// the color of that point.
glm::dvec3 Material::shade(Scene *scene, const ray &r, const isect &i) const {
  // YOUR CODE HERE
  const glm::dvec3 P = r.at(i.getT());
  glm::dvec3 N = glm::normalize(i.getN());

  // Ambient + emissive
  glm::dvec3 color = ke(i) + ka(i) * scene->ambient(); 
  // ambient() is Ia in this codebase

  // View direction: from hit point toward the camera
  const glm::dvec3 V = glm::normalize(-r.getDirection());

  for (const auto &pLight : scene->getAllLights()) {
    // Direction from point to light
    const glm::dvec3 L = glm::normalize(pLight->getDirection(P));

    // Front-face only diffuse/spec (Lambert cosine term)
    const double NdotL = glm::dot(N, L);
    if (NdotL <= 0.0) continue;

    // Shadow ray (blocked/unblocked per Lecture 3)
    const glm::dvec3 shadowOrigin = P + RAY_EPSILON * N;
    ray shadowRay(shadowOrigin, L, glm::dvec3(1.0), ray::SHADOW);
    const glm::dvec3 shadow = pLight->shadowAttenuation(shadowRay, P);

    // Distance attenuation (point lights only; directional returns 1)
    const double fd = pLight->distanceAttenuation(P);

    // Incoming light (per-channel)
    const glm::dvec3 Iin = pLight->getColor() * fd * shadow;

    // Diffuse (Lambert)
    color += kd(i) * Iin * NdotL;

    // Specular (Phong using v·r, as in the lecture slides)
    const glm::dvec3 R = glm::normalize(glm::reflect(-L, N)); // reflect incoming (-L) about N
    const double VdotR = glm::dot(V, R);
    if (VdotR > 0.0) {
      color += ks(i) * Iin * std::pow(VdotR, shininess(i));
    }
  }

  return color;

  // For now, this method just returns the diffuse color of the object.
  // This gives a single matte color for every distinct surface in the
  // scene, and that's it.  Simple, but enough to get you started.
  // (It's also inconsistent with the phong model...)

  // Your mission is to fill in this method with the rest of the phong
  // shading model, including the contributions of all the light sources.
  // You will need to call both distanceAttenuation() and
  // shadowAttenuation()
  // somewhere in your code in order to compute shadows and light falloff.
  //	if( debugMode )
  //		std::cout << "Debugging Phong code..." << std::endl;

  // When you're iterating through the lights,
  // you'll want to use code that looks something
  // like this:
  //
  // for ( const auto& pLight : scene->getAllLights() )
  // {
  //              // pLight has type Light*
  // 		.
  // 		.
  // 		.
  // }
  // return kd(i);
}

TextureMap::TextureMap(string filename) {
  data = readImage(filename.c_str(), width, height);
  if (data.empty()) {
    width = 0;
    height = 0;
    string error("Unable to load texture map '");
    error.append(filename);
    error.append("'.");
    throw TextureMapException(error);
  }
}

glm::dvec3 TextureMap::getMappedValue(const glm::dvec2 &coord) const {
  // YOUR CODE HERE
  if (width <= 0 || height <= 0 || data.empty()) return glm::dvec3(0.0);

  // Clamp UV to [0,1]
  const double u = std::clamp(coord.x, 0.0, 1.0);
  const double v = std::clamp(coord.y, 0.0, 1.0);

  // Map to continuous pixel coordinates
  const double fx = u * (width  - 1);
  const double fy = v * (height - 1);

  const int x0 = (int)std::floor(fx);
  const int y0 = (int)std::floor(fy);
  const int x1 = std::min(x0 + 1, width  - 1);
  const int y1 = std::min(y0 + 1, height - 1);

  const double tx = fx - x0;
  const double ty = fy - y0;

  const glm::dvec3 c00 = getPixelAt(x0, y0);
  const glm::dvec3 c10 = getPixelAt(x1, y0);
  const glm::dvec3 c01 = getPixelAt(x0, y1);
  const glm::dvec3 c11 = getPixelAt(x1, y1);

  const glm::dvec3 c0 = (1.0 - tx) * c00 + tx * c10;
  const glm::dvec3 c1 = (1.0 - tx) * c01 + tx * c11;
  return (1.0 - ty) * c0 + ty * c1;
  //
  // In order to add texture mapping support to the
  // raytracer, you need to implement this function.
  // What this function should do is convert from
  // parametric space which is the unit square
  // [0, 1] x [0, 1] in 2-space to bitmap coordinates,
  // and use these to perform bilinear interpolation
  // of the values.

  // return glm::dvec3(1, 1, 1);
}

glm::dvec3 TextureMap::getPixelAt(int x, int y) const {
  // YOUR CODE HERE
  if (width <= 0 || height <= 0 || data.empty()) return glm::dvec3(0.0);

  x = std::clamp(x, 0, width - 1);
  y = std::clamp(y, 0, height - 1);

  const int idx = 3 * (y * width + x);
  const double r = data[idx + 0] / 255.0;
  const double g = data[idx + 1] / 255.0;
  const double b = data[idx + 2] / 255.0;

  return glm::dvec3(r, g, b);
  //
  // In order to add texture mapping support to the
  // raytracer, you need to implement this function.

  // return glm::dvec3(1, 1, 1);
}

glm::dvec3 MaterialParameter::value(const isect &is) const {
  if (0 != _textureMap)
    return _textureMap->getMappedValue(is.getUVCoordinates());
  else
    return _value;
}

double MaterialParameter::intensityValue(const isect &is) const {
  if (0 != _textureMap) {
    glm::dvec3 value(_textureMap->getMappedValue(is.getUVCoordinates()));
    return (0.299 * value[0]) + (0.587 * value[1]) + (0.114 * value[2]);
  } else
    return (0.299 * _value[0]) + (0.587 * _value[1]) + (0.114 * _value[2]);
}
