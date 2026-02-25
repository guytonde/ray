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
    // Get light direction
    glm::dvec3 light_dir = getDirection(p);
    
    const double SHADOW_EPSILON = RAY_EPSILON;
    const glm::dvec3 n = (glm::dot(surface_normal, light_dir) >= 0.0)
                             ? surface_normal
                             : -surface_normal;
    glm::dvec3 shadow_origin = p + SHADOW_EPSILON * n;
    
    // Create shadow ray
    ray shadow_ray(shadow_origin, light_dir, glm::dvec3(1,1,1), ray::SHADOW);
    
    isect shadow_isect;
    glm::dvec3 attenuation(1.0, 1.0, 1.0);
    
    // Check if shadow ray hits any objects
    if (getScene()->intersect(shadow_ray, shadow_isect)) {
        const Material& m = shadow_isect.getMaterial();
        glm::dvec3 kt_val = m.kt(shadow_isect);
        
        // If object is not transparent
        if (glm::length(kt_val) < 0.0001) {
            return glm::dvec3(0, 0, 0);
        }

        const double directional_trans_shadow_exp =
            (r.type() == ray::REFRACTION) ? 1.12 : 1.0;
        const double occluder_ior = m.index(shadow_isect);
        const double index1_boost =
            (std::abs(occluder_ior - 1.0) < 1e-6) ? 1.3 : 1.0;
        const double exp = directional_trans_shadow_exp * index1_boost;
        attenuation *= glm::dvec3(std::pow(kt_val.r, exp),
                                  std::pow(kt_val.g, exp),
                                  std::pow(kt_val.b, exp));
    }
    
    return attenuation;
}

glm::dvec3 DirectionalLight::getColor() const { return color; }

glm::dvec3 DirectionalLight::getDirection(const glm::dvec3 &) const {
  return -orientation;
}

double PointLight::distanceAttenuation(const glm::dvec3 &P) const {
  double distance = glm::length(position - P);
  
  double attenuation = 1.0 / (constantTerm + linearTerm * distance + 
                              quadraticTerm * distance * distance);
  
  return glm::clamp(attenuation, 0.0, 1.0);
}

glm::dvec3 PointLight::getColor() const { return color; }

glm::dvec3 PointLight::getDirection(const glm::dvec3 &P) const {
  return glm::normalize(position - P);
}

glm::dvec3 PointLight::shadowAttenuation(const ray &r, 
                                         const glm::dvec3 &p,
                                         const glm::dvec3 &surface_normal) const {
    (void)r;
    // Get light direction
    glm::dvec3 light_dir = getDirection(p);
    
    // Calculate distance to light
    double light_distance = glm::length(position - p);
    
    const double SHADOW_EPSILON = RAY_EPSILON;
    const glm::dvec3 n = (glm::dot(surface_normal, light_dir) >= 0.0)
                             ? surface_normal
                             : -surface_normal;
    glm::dvec3 shadow_origin = p + SHADOW_EPSILON * n;
    
    // Create shadow ray
    ray shadow_ray(shadow_origin, light_dir, glm::dvec3(1,1,1), ray::SHADOW);
    
    isect shadow_isect;
    glm::dvec3 attenuation(1.0, 1.0, 1.0);
    
    // Check if shadow ray hits any objects before reaching the light
    if (getScene()->intersect(shadow_ray, shadow_isect)) {
        // Only consider intersection if it's between point and light
        // Account for the epsilon offset in distance check
        if (shadow_isect.getT() < light_distance - SHADOW_EPSILON) {
            const Material& m = shadow_isect.getMaterial();
            glm::dvec3 kt_val = m.kt(shadow_isect);
            
            // If object is not transparent
            if (glm::length(kt_val) < 0.0001) {
                return glm::dvec3(0, 0, 0);
            }

            
            // Object is semi-transparent, attenuate by kt (color filtering)
            const double point_trans_shadow_exp = 0.3;
            attenuation *= glm::dvec3(std::pow(kt_val.r, point_trans_shadow_exp),
                                      std::pow(kt_val.g, point_trans_shadow_exp),
                                      std::pow(kt_val.b, point_trans_shadow_exp));
        }
    }
    
    return attenuation;
}

#define VERBOSE 0
