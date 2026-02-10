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
    // Get light direction
    glm::dvec3 light_dir = getDirection(p);
    
    const double SHADOW_EPSILON = RAY_EPSILON;
    glm::dvec3 shadow_origin = p + SHADOW_EPSILON * light_dir;
    
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
        
        // Object is semi-transparent, attenuate by kt (color filtering)
        attenuation *= kt_val;
    }
    
    return attenuation;
}

glm::dvec3 DirectionalLight::getColor() const { return color; }

glm::dvec3 DirectionalLight::getDirection(const glm::dvec3 &) const {
  return -orientation;
}

double PointLight::distanceAttenuation(const glm::dvec3 &P) const {
  // YOUR CODE HERE
  // Calculate distance from light to point
  double distance = glm::length(position - P);
  
  // Apply attenuation formula: f(d) = min(1, 1/(a + b*d + c*d^2))
  double attenuation = 1.0 / (constantTerm + linearTerm * distance + 
                              quadraticTerm * distance * distance);
  
  // Clamp to [0, 1] as per reference solution
  return glm::clamp(attenuation, 0.0, 1.0);


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
    // Get light direction
    glm::dvec3 light_dir = getDirection(p);
    
    // Calculate distance to light
    double light_distance = glm::length(position - p);
    
    // CRITICAL: Offset along LIGHT DIRECTION, not normal
    const double SHADOW_EPSILON = RAY_EPSILON;
    glm::dvec3 shadow_origin = p + SHADOW_EPSILON * light_dir;
    
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
            attenuation *= kt_val;
        }
    }
    
    return attenuation;
}

#define VERBOSE 0

