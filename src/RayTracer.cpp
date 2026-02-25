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

namespace {
int samplesPerDimension(int configuredSamples) {
  if (configuredSamples <= 1) {
    return 1;
  }
  // GUI slider semantics: 1..4 means 1x1, 2x2, 3x3, 4x4.
  if (configuredSamples <= 4) {
    return configuredSamples;
  }
  // Backward compatibility: allow legacy total sample values 9 or 16.
  const int root = static_cast<int>(std::sqrt(configuredSamples));
  if (root * root == configuredSamples) {
    return root;
  }
  return configuredSamples;
}
} // namespace

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

    const int samples_per_dim =
        (traceUI != nullptr && traceUI->aaSwitch())
            ? samplesPerDimension(samples)
            : 1;
    const int total_samples = samples_per_dim * samples_per_dim;

    if (samples_per_dim <= 1) {
        double x = double(i) / double(buffer_width);
        double y = double(j) / double(buffer_height);
        col = trace(x, y);
    } else {
        double sample_offset = 1.0 / samples_per_dim;
        
        for (int si = 0; si < samples_per_dim; ++si) {
            for (int sj = 0; sj < samples_per_dim; ++sj) {
                // Sample at center of sub-pixel
                double x = (double(i) + (si + 0.5) * sample_offset) / double(buffer_width);
                double y = (double(j) + (sj + 0.5) * sample_offset) / double(buffer_height);
                
                col += trace(x, y);
            }
        }
        
        col /= double(total_samples);
    }

    col = glm::clamp(col, 0.0, 1.0);

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
        const Material &m = i.getMaterial();
        colorC = m.shade(scene.get(), r, i);

        const glm::dvec3 point = r.at(i.getT());
        const glm::dvec3 normal = glm::normalize(i.getN());
        const glm::dvec3 incident = glm::normalize(r.getDirection());
        t = i.getT();

        const double REFLECT_EPSILON = 1e-4;
        const double REFRACT_EPSILON = 1e-7;
        if (depth > 0) {
            if (m.Refl()) {
                const glm::dvec3 kr_val = m.kr(i);
                if (glm::length(kr_val) > 0.0) {
                    const glm::dvec3 reflect_dir =
                        glm::normalize(incident - 2.0 * glm::dot(incident, normal) * normal);
                    ray reflect_ray(point + REFLECT_EPSILON * reflect_dir, reflect_dir,
                                    glm::dvec3(1.0), ray::REFLECTION);
                    double dummy_t = 0.0;
                    colorC +=
                        kr_val * traceRay(reflect_ray, thresh, depth - 1, dummy_t);
                }
            }

            if (m.Trans()) {
                const glm::dvec3 kt_val = m.kt(i);
                if (glm::length(kt_val) > 0.0) {
                    glm::dvec3 n = normal;
                    double n1 = 1.0;
                    double n2 = m.index(i);
                    if (n2 <= 0.0) {
                        n2 = 1.0;
                    }

                    double d_dot_n = glm::dot(incident, n);
                    if (d_dot_n > 0.0) {
                        n = -n;
                        n1 = n2;
                        n2 = 1.0;
                        d_dot_n = glm::dot(incident, n);
                    }

                    glm::dvec3 refract_dir(incident);
                    bool has_refract = true;

                    if (std::abs(n1 - n2) > 1e-9) {
                        const double eta = n1 / n2;
                        const double cos_theta_i = -d_dot_n;
                        const double sin_t2 =
                            eta * eta * (1.0 - cos_theta_i * cos_theta_i);
                        if (sin_t2 <= 1.0 + 1e-6) {
                            const double cos_t =
                                std::sqrt(std::max(0.0, 1.0 - sin_t2));
                            refract_dir = glm::normalize(
                                eta * incident + (eta * cos_theta_i - cos_t) * n);
                        } else {
                            has_refract = false;
                        }
                    }

                    if (has_refract) {
                        ray refract_ray(point + REFRACT_EPSILON * refract_dir, refract_dir,
                                        glm::dvec3(1.0), ray::REFRACTION);
                        double dummy_t = 0.0;
                        colorC += traceRay(refract_ray, thresh, depth - 1, dummy_t);
                    }
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

    if (sceneLoaded()) {
        scene->buildAcceleration(traceUI->getMaxDepth(), traceUI->getLeafSize());
    }
    
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
    // Supersampling is applied directly in tracePixel().
    return 0;
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
