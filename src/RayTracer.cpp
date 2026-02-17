#pragma warning(disable : 4786)

#include "RayTracer.h"
#include "scene/light.h"
#include "scene/material.h"
#include "scene/ray.h"

#include "parser/JsonParser.h"
#include "parser/Parser.h"
#include "parser/Tokenizer.h"
#include <json.hpp>

#include "ui/TraceUI.h"
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtx/io.hpp>
#include <string.h>

#include <fstream>
#include <iostream>

using namespace std;
extern TraceUI *traceUI;

bool debugMode = false;

glm::dvec3 RayTracer::trace(double x, double y) {
  if (TraceUI::m_debug) {
    scene->clearIntersectCache();
  }

  ray r(glm::dvec3(0, 0, 0), glm::dvec3(0, 0, 0), glm::dvec3(1, 1, 1),
        ray::VISIBILITY);
  scene->getCamera().rayThrough(x, y, r);
  double dummy;
  glm::dvec3 ret =
      traceRay(r, glm::dvec3(1.0, 1.0, 1.0), traceUI->getDepth(), dummy);
  ret = glm::clamp(ret, 0.0, 1.0);
  return ret;
}

glm::dvec3 RayTracer::tracePixel(int i, int j) {
    glm::dvec3 col(0, 0, 0);

    if (!sceneLoaded())
        return col;

    // Support for supersampling
    int sqrt_samples = (int)sqrt(samples);
    
    if (sqrt_samples <= 1) {
        // No anti-aliasing - single sample at pixel center
        double x = double(i) / double(buffer_width);
        double y = double(j) / double(buffer_height);
        col = trace(x, y);
    } else {
        // Anti-aliasing - multiple samples per pixel
        double sample_offset = 1.0 / sqrt_samples;
        
        for (int si = 0; si < sqrt_samples; ++si) {
            for (int sj = 0; sj < sqrt_samples; ++sj) {
                // Sample at center of sub-pixel
                double x = (double(i) + (si + 0.5) * sample_offset) / double(buffer_width);
                double y = (double(j) + (sj + 0.5) * sample_offset) / double(buffer_height);
                
                col += trace(x, y);
            }
        }
        
        // Average all samples
        col /= double(samples);
    }

    unsigned char *pixel = buffer.data() + (i + j * buffer_width) * 3;
    pixel[0] = (int)(255.0 * col[0]);
    pixel[1] = (int)(255.0 * col[1]);
    pixel[2] = (int)(255.0 * col[2]);
    
    return col;
}

// Recursive ray tracing with reflection and refraction
glm::dvec3 RayTracer::traceRay(ray &r, const glm::dvec3 &thresh, int depth,
                               double &t) {
    isect i;
    glm::dvec3 colorC;
#if VERBOSE
    std::cerr << "== current depth: " << depth << std::endl;
#endif
    
    if (scene->intersect(r, i)) {
        // Get material properties
        const Material &m = i.getMaterial();
        
        // Base shading from Phong model
        colorC = m.shade(scene.get(), r, i);
        
        // Get intersection point and normal
        glm::dvec3 point = r.at(i.getT());
        glm::dvec3 normal = i.getN();
        
        // Store intersection distance
        t = i.getT();
        
        // Small epsilon to prevent self-intersection
        const double EPSILON = 1e-6;
        
        // Only recurse if we haven't exceeded depth limit
        if (depth > 0) {
            // Handle reflection
            if (m.Refl()) {
                glm::dvec3 kr_val = m.kr(i);
                
                // Check if contribution is significant
                if (glm::length(kr_val) > 0.0) {
                    // Compute reflection direction: R = D - 2(D·N)N
                    glm::dvec3 reflect_dir = glm::normalize(
                        r.getDirection() - 2.0 * glm::dot(r.getDirection(), normal) * normal
                    );
                    
                    // Create reflection ray with offset to prevent self-intersection
                    glm::dvec3 reflect_origin = point + EPSILON * reflect_dir;
                    ray reflect_ray(reflect_origin, reflect_dir, glm::dvec3(1,1,1), ray::REFLECTION);
                    double dummy_t;
                    
                    // Recursively trace reflection
                    glm::dvec3 reflect_color = traceRay(reflect_ray, thresh, depth - 1, dummy_t);
                    
                    // Add reflection contribution
                    colorC += kr_val * reflect_color;
                }
            }
            
            // Handle refraction (transmission)
            if (m.Trans()) {
                glm::dvec3 kt_val = m.kt(i);
                
                // Check if contribution is significant
                if (glm::length(kt_val) > 0.0) {
                    // Determine if we're entering or exiting the object
                    // If ray and normal point in opposite directions, we're entering
                    glm::dvec3 incident = glm::normalize(r.getDirection());
                    double cos_i = glm::dot(incident, normal);
                    bool entering = cos_i < 0;
                    
                    // Indices of refraction
                    double n1 = entering ? 1.0 : m.index(i);  // From air or from material
                    double n2 = entering ? m.index(i) : 1.0;  // To material or to air
                    double eta = n1 / n2;
                    
                    // For refraction calculation, ensure cos_i is positive
                    double cos_i_abs = std::abs(cos_i);
                    
                    // Compute sin²(theta_t) using Snell's law
                    double sin_t2 = eta * eta * (1.0 - cos_i_abs * cos_i_abs);
                    
                    // Check for total internal reflection
                    if (sin_t2 <= 1.0) {
                        double cos_t = std::sqrt(1.0 - sin_t2);
                        
                        // Compute refracted direction
                        // If entering: refract into material (same hemisphere as -normal)
                        // If exiting: refract into air (same hemisphere as normal)
                        glm::dvec3 refract_dir;
                        if (entering) {
                            refract_dir = glm::normalize(
                                eta * incident + (eta * cos_i_abs - cos_t) * normal
                            );
                        } else {
                            refract_dir = glm::normalize(
                                eta * incident - (eta * cos_i_abs - cos_t) * normal
                            );
                        }
                        
                        // Create refraction ray with offset to prevent self-intersection
                        glm::dvec3 refract_origin = point + EPSILON * refract_dir;
                        ray refract_ray(refract_origin, refract_dir, glm::dvec3(1,1,1), ray::REFRACTION);
                        double dummy_t;
                        
                        // Recursively trace refraction
                        glm::dvec3 refract_color = traceRay(refract_ray, thresh, depth - 1, dummy_t);
                        
                        // Add refraction contribution
                        colorC += kt_val * refract_color;
                    }
                    // If total internal reflection occurs, only reflection contributes
                    // (already handled above if Refl() is true)
                }
            }
        }
    } else {
        // No intersection - check for cube map
        if (traceUI->cubeMap() && traceUI->getCubeMap()) {
            colorC = traceUI->getCubeMap()->getColor(r);
        } else {
            colorC = glm::dvec3(0.0, 0.0, 0.0);
        }
        t = 1000.0;
    }
    
#if VERBOSE
    std::cerr << "== depth: " << depth + 1 << " done, returning: " << colorC
              << std::endl;
#endif
    return colorC;
}

// Anti-aliasing by supersampling
// glm::dvec3 RayTracer::ssPixel(int i, int j, int dim) {
//   glm::dvec3 accum(0.0);

//   if (!sceneLoaded())
//     return accum;

//   const double pixelWidth = 1.0 / double(buffer_width);
//   const double pixelHeight = 1.0 / double(buffer_height);
//   const double subWidth = pixelWidth / double(dim);
//   const double subHeight = pixelHeight / double(dim);

//   for (int sy = 0; sy < dim; ++sy) {
//     for (int sx = 0; sx < dim; ++sx) {
//       const double x = (double(i) + (sx + 0.5) / dim) * pixelWidth;
//       const double y = (double(j) + (sy + 0.5) / dim) * pixelHeight;
//       accum += trace(x, y);
//     }
//   }

//   glm::dvec3 col = accum / double(dim * dim);

//   unsigned char *pixel = buffer.data() + (i + j * buffer_width) * 3;
//   pixel[0] = (int)(255.0 * col[0]);
//   pixel[1] = (int)(255.0 * col[1]);
//   pixel[2] = (int)(255.0 * col[2]);

//   return col;
// }

RayTracer::RayTracer()
    : scene(nullptr), buffer(0), thresh(0), buffer_width(0), buffer_height(0),
      m_bBufferReady(false) {
}

RayTracer::~RayTracer() {}

void RayTracer::getBuffer(unsigned char *&buf, int &w, int &h) {
  buf = buffer.data();
  w = buffer_width;
  h = buffer_height;
}

double RayTracer::aspectRatio() {
  return sceneLoaded() ? scene->getCamera().getAspectRatio() : 1;
}

bool RayTracer::loadScene(const char *fn) {
  ifstream ifs(fn);
  if (!ifs) {
    string msg("Error: couldn't read scene file ");
    msg.append(fn);
    traceUI->alert(msg);
    return false;
  }

  bool isRay = false;
  const char *ext = strrchr(fn, '.');
  if (ext && !strcmp(ext, ".ray"))
    isRay = true;

  string path(fn);
  if (path.find_last_of("\\/") == string::npos)
    path = ".";
  else
    path = path.substr(0, path.find_last_of("\\/"));

  if (isRay) {
    Tokenizer tokenizer(ifs, false);
    Parser parser(tokenizer, path);
    try {
      scene.reset(parser.parseScene());
    } catch (SyntaxErrorException &pe) {
      traceUI->alert(pe.formattedMessage());
      return false;
    } catch (ParserException &pe) {
      string msg("Parser: fatal exception ");
      msg.append(pe.message());
      traceUI->alert(msg);
      return false;
    } catch (TextureMapException e) {
      string msg("Texture mapping exception: ");
      msg.append(e.message());
      traceUI->alert(msg);
      return false;
    }
  } else {
    try {
      JsonParser parser(path, ifs);
      scene.reset(parser.parseScene());
    } catch (ParserException &pe) {
      string msg("Parser: fatal exception ");
      msg.append(pe.message());
      traceUI->alert(msg);
      return false;
    } catch (const json::exception &je) {
      string msg("Invalid JSON encountered ");
      msg.append(je.what());
      traceUI->alert(msg);
      return false;
    }
  }

  if (!sceneLoaded())
    return false;

  scene->buildAcceleration(traceUI->getMaxDepth(), traceUI->getLeafSize());

  return true;
}

void RayTracer::traceSetup(int w, int h) {
    size_t newBufferSize = w * h * 3;
    if (newBufferSize != buffer.size()) {
        bufferSize = newBufferSize;
        buffer.resize(bufferSize);
    }
    buffer_width = w;
    buffer_height = h;
    std::fill(buffer.begin(), buffer.end(), 0);
    m_bBufferReady = true;

    threads = traceUI->getThreads();
    block_size = traceUI->getBlockSize();
    thresh = traceUI->getThreshold();
    samples = traceUI->getSuperSamples();
    aaThresh = traceUI->getAaThreshold();
    
    // Initialize threading
    stopTrace = false;
    // Clear any existing worker threads
    worker_threads.clear();
}

void RayTracer::traceImage(int w, int h) {
    traceSetup(w, h);
    
    // Divide work among threads
    auto trace_rows = [this](int start_row, int end_row) {
        for (int j = start_row; j < end_row && !stopTrace; ++j) {
            for (int i = 0; i < buffer_width && !stopTrace; ++i) {
                tracePixel(i, j);
            }
        }
    };
    
    // Launch worker threads
    int rows_per_thread = buffer_height / threads;
    for (unsigned int t = 0; t < threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == threads - 1) ? buffer_height : (t + 1) * rows_per_thread;
        worker_threads.emplace_back(trace_rows, start_row, end_row);
    }
}

int RayTracer::aaImage() {
    if (!sceneLoaded() || samples <= 1)
        return 0;
    
    int pixels_aa = 0;
    int sqrt_samples = (int)sqrt(samples);
    
    for (int j = 0; j < buffer_height; ++j) {
        for (int i = 0; i < buffer_width; ++i) {
            // Check if this pixel needs anti-aliasing by comparing with neighbors
            glm::dvec3 current = getPixel(i, j);
            bool needs_aa = false;
            
            // Check right neighbor
            if (i < buffer_width - 1) {
                glm::dvec3 right = getPixel(i + 1, j);
                if (glm::length(current - right) > aaThresh) {
                    needs_aa = true;
                }
            }
            
            // Check bottom neighbor
            if (j < buffer_height - 1) {
                glm::dvec3 bottom = getPixel(i, j + 1);
                if (glm::length(current - bottom) > aaThresh) {
                    needs_aa = true;
                }
            }
            
            // If edge detected, apply supersampling
            if (needs_aa) {
                glm::dvec3 col(0, 0, 0);
                double sample_offset = 1.0 / sqrt_samples;
                
                for (int si = 0; si < sqrt_samples; ++si) {
                    for (int sj = 0; sj < sqrt_samples; ++sj) {
                        double x = (double(i) + (si + 0.5) * sample_offset) / double(buffer_width);
                        double y = (double(j) + (sj + 0.5) * sample_offset) / double(buffer_height);
                        col += trace(x, y);
                    }
                }
                
                col /= double(samples);
                setPixel(i, j, col);
                pixels_aa++;
            }
        }
    }
    
    return pixels_aa;
}

bool RayTracer::checkRender() {
    // Check if all threads are done
    if (worker_threads.empty())
        return true;
        
    for (auto& thread : worker_threads) {
        if (thread.joinable())
            return false;
    }
    
    return true;
}

void RayTracer::waitRender() {
    // Join all worker threads
    for (auto& thread : worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    worker_threads.clear();
}

glm::dvec3 RayTracer::getPixel(int i, int j) {
  unsigned char *pixel = buffer.data() + (i + j * buffer_width) * 3;
  return glm::dvec3((double)pixel[0] / 255.0, (double)pixel[1] / 255.0,
                    (double)pixel[2] / 255.0);
}

void RayTracer::setPixel(int i, int j, glm::dvec3 color) {
  unsigned char *pixel = buffer.data() + (i + j * buffer_width) * 3;

  pixel[0] = (int)(255.0 * color[0]);
  pixel[1] = (int)(255.0 * color[1]);
  pixel[2] = (int)(255.0 * color[2]);
}
