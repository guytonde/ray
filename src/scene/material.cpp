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
  // Get material properties at this intersection point
  glm::dvec3 ke_val = ke(i); // Emissive
  glm::dvec3 ka_val = ka(i); // Ambient
  glm::dvec3 kd_val = kd(i); // Diffuse
  glm::dvec3 ks_val = ks(i); // Specular
  double shininess_val = shininess(i);

  // Get intersection point and normal
  glm::dvec3 point = r.at(i.getT());
  glm::dvec3 normal = glm::normalize(i.getN());

  // View direction (from point to camera)
  glm::dvec3 view_dir = glm::normalize(-r.getDirection());
  glm::dvec3 shading_normal = normal;
  const bool true_refraction_ior = index(i) > 1.0 + 1e-6;
  if (r.type() == ray::REFRACTION && true_refraction_ior &&
      glm::dot(shading_normal, view_dir) < 0.0) {
    shading_normal = -shading_normal;
  }
  // Start with emissive and ambient components
  glm::dvec3 color = ke_val + ka_val * scene->ambient();

  // Iterate through all lights
  for (const auto &pLight : scene->getAllLights()) {
    glm::dvec3 light_dir = glm::normalize(pLight->getDirection(point));
    double atten = pLight->distanceAttenuation(point);
    glm::dvec3 shadow_atten(1.0, 1.0, 1.0);
    const bool eta1_refraction_ray =
        (r.type() == ray::REFRACTION) && Trans() &&
        (std::abs(index(i) - 1.0) < 1e-6);
    const bool both_refraction_ray =
        (r.type() == ray::REFRACTION) && Both();
    const bool matte_transparent_visibility =
        (r.type() == ray::VISIBILITY) && Trans() && !Spec();
    if (traceUI->shadowSw() && !eta1_refraction_ray && !both_refraction_ray &&
        !matte_transparent_visibility) {
      shadow_atten = pLight->shadowAttenuation(r, point, shading_normal);
    }

    glm::dvec3 light_contribution = atten * shadow_atten * pLight->getColor();

    const double n_dot_l_raw = glm::dot(shading_normal, light_dir);
    const bool primary_visibility = (r.type() == ray::VISIBILITY);
    const bool secondary_eta1_refraction =
        (r.type() == ray::REFRACTION) && Trans() &&
        (std::abs(index(i) - 1.0) < 1e-6);
    const bool primary_transparent = primary_visibility && Trans();
    const bool secondary_matte_refraction =
        (r.type() == ray::REFRACTION) && Trans() && !Spec();
    const bool one_sided = (!primary_transparent && primary_visibility) ||
                           secondary_eta1_refraction ||
                           secondary_matte_refraction;
    const double n_dot_l =
        one_sided ? glm::max(0.0, n_dot_l_raw) : std::abs(n_dot_l_raw);
    glm::dvec3 diffuse = kd_val * n_dot_l;

    glm::dvec3 specular(0.0, 0.0, 0.0);
    const bool allow_backface_spec = traceUI->backfaceSpecular();
    const bool do_specular =
        shininess_val > 0.0 &&
        (one_sided ? (n_dot_l_raw > 0.0)
                    : (n_dot_l_raw > 0.0 || allow_backface_spec));
    if (do_specular) {
      const double spec_n_dot_l =
          one_sided
              ? n_dot_l_raw
              : (allow_backface_spec ? std::abs(n_dot_l_raw) : n_dot_l_raw);
      glm::dvec3 reflect_dir =
          glm::normalize((2.0 * spec_n_dot_l * shading_normal) - light_dir);
      double r_dot_v = glm::max(0.0, glm::dot(reflect_dir, view_dir));
      if (r_dot_v > 0.0) {
        specular = ks_val * pow(r_dot_v, shininess_val);
      }
    }

    color += light_contribution * (diffuse + specular);
  }

  return color;
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
}

glm::dvec3 TextureMap::getPixelAt(int x, int y) const {
  if (width <= 0 || height <= 0 || data.empty()) return glm::dvec3(0.0);

  x = std::clamp(x, 0, width - 1);
  y = std::clamp(y, 0, height - 1);

  const int idx = 3 * (y * width + x);
  const double r = data[idx + 0] / 255.0;
  const double g = data[idx + 1] / 255.0;
  const double b = data[idx + 2] / 255.0;

  return glm::dvec3(r, g, b);
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
