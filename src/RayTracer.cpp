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
#include <cstdlib>
#include <cctype>
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

struct HarmonicSample {
  double value;
  glm::dvec3 grad;
  double singularDistance;
};

double realPartSqrtComplex(double x, double y) {
  const double r = std::sqrt(x * x + y * y);
  return std::sqrt(std::max(0.0, 0.5 * (r + x)));
}

double evaluateRiemann(const glm::dvec3 &p) {
  return p.z - realPartSqrtComplex(p.x, p.y);
}

HarmonicSample sampleRiemann(const glm::dvec3 &p) {
  HarmonicSample s;
  s.value = evaluateRiemann(p);

  const double r2 = p.x * p.x + p.y * p.y;
  const double r = std::sqrt(r2);
  s.singularDistance = r;

  if (r < 1e-8) {
    s.grad = glm::dvec3(0.0, 0.0, 1.0);
    return s;
  }

  const double u = realPartSqrtComplex(p.x, p.y);
  if (u < 1e-8) {
    const double eps = 1e-4;
    const glm::dvec3 ex(eps, 0.0, 0.0);
    const glm::dvec3 ey(0.0, eps, 0.0);
    const glm::dvec3 ez(0.0, 0.0, eps);
    s.grad = glm::dvec3(
        (evaluateRiemann(p + ex) - evaluateRiemann(p - ex)) / (2.0 * eps),
        (evaluateRiemann(p + ey) - evaluateRiemann(p - ey)) / (2.0 * eps),
        (evaluateRiemann(p + ez) - evaluateRiemann(p - ez)) / (2.0 * eps));
    return s;
  }

  const double drdx = p.x / r;
  const double drdy = p.y / r;
  const double dudx = (drdx + 1.0) / (4.0 * u);
  const double dudy = drdy / (4.0 * u);
  s.grad = glm::dvec3(-dudx, -dudy, 1.0);
  return s;
}

double evaluateGyroid(const glm::dvec3 &p) {
  return std::sin(p.x) * std::cos(p.y) + std::sin(p.y) * std::cos(p.z) +
         std::sin(p.z) * std::cos(p.x);
}

HarmonicSample sampleGyroid(const glm::dvec3 &p) {
  HarmonicSample s;
  s.value = evaluateGyroid(p);
  s.grad = glm::dvec3(std::cos(p.x) * std::cos(p.y) - std::sin(p.z) * std::sin(p.x),
                      -std::sin(p.x) * std::sin(p.y) + std::cos(p.y) * std::cos(p.z),
                      -std::sin(p.y) * std::sin(p.z) + std::cos(p.z) * std::cos(p.x));
  s.singularDistance = 1e9;
  return s;
}

HarmonicSample sampleField(const glm::dvec3 &p, RayTracer::HarmonicMode mode) {
  return (mode == RayTracer::HarmonicMode::GYROID) ? sampleGyroid(p)
                                                   : sampleRiemann(p);
}

glm::dvec3 baseColorAt(const glm::dvec3 &p, RayTracer::HarmonicMode mode) {
  if (mode == RayTracer::HarmonicMode::GYROID) {
    const glm::dvec3 c(0.5 + 0.5 * std::cos(0.9 * p.x + 0.0),
                       0.5 + 0.5 * std::cos(0.9 * p.y + 2.1),
                       0.5 + 0.5 * std::cos(0.9 * p.z + 4.2));
    return glm::clamp(c, 0.0, 1.0);
  }

  const double phase = std::atan2(p.y, p.x);
  glm::dvec3 c(0.5 + 0.45 * std::cos(phase + 0.0),
               0.5 + 0.45 * std::cos(phase + 2.09439510239),
               0.5 + 0.45 * std::cos(phase + 4.18879020479));
  const double stripes = 0.5 + 0.5 * std::cos(8.0 * p.z + phase);
  c = 0.8 * c + 0.2 * glm::dvec3(stripes);
  return glm::clamp(c, 0.0, 1.0);
}

glm::dvec3 shadeHarmonicPoint(const Scene *scene, const ray &cameraRay,
                              const glm::dvec3 &p, const glm::dvec3 &n,
                              const glm::dvec3 &baseColor) {
  const glm::dvec3 viewDir = glm::normalize(-cameraRay.getDirection());
  glm::dvec3 color = 0.1 * baseColor + 0.25 * scene->ambient() * baseColor;

  for (const auto *light : scene->getAllLights()) {
    const glm::dvec3 l = glm::normalize(light->getDirection(p));
    const double atten = light->distanceAttenuation(p);
    if (atten <= 0.0) {
      continue;
    }

    glm::dvec3 shadow(1.0);
    if (traceUI && traceUI->shadowSw()) {
      shadow = light->shadowAttenuation(cameraRay, p, n);
    }

    const double ndotl = std::max(0.0, glm::dot(n, l));
    const glm::dvec3 diffuse = ndotl * baseColor;

    const glm::dvec3 r = glm::normalize(2.0 * glm::dot(n, l) * n - l);
    const double spec = std::pow(std::max(0.0, glm::dot(r, viewDir)), 72.0);
    const glm::dvec3 specular = 0.3 * glm::dvec3(spec);

    color += atten * shadow * light->getColor() * (diffuse + specular);
  }

  return glm::clamp(color, 0.0, 1.0);
}

double harnackStepSize(const HarmonicSample &s, const glm::dvec3 &dir) {
  const double absValue = std::abs(s.value);
  const double directionalGrad = std::abs(glm::dot(s.grad, dir));
  const double newtonStep = absValue / std::max(1e-6, directionalGrad);

  const double radiusCap = std::max(2e-4, 0.45 * s.singularDistance);
  const double harnackStep = radiusCap * (absValue / (1.0 + absValue));

  const double step = std::min(newtonStep, harnackStep);
  return std::clamp(step, 2e-4, 0.5);
}

std::string toLowerCopy(std::string text) {
  for (char &c : text) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return text;
}
} // namespace

glm::dvec3 RayTracer::trace(double x, double y) {
  if (TraceUI::m_debug) {
    scene->clearIntersectCache();
  }

  ray r(glm::dvec3(0, 0, 0), glm::dvec3(0, 0, 0), glm::dvec3(1, 1, 1),
        ray::VISIBILITY);
  scene->getCamera().rayThrough(x, y, r);
  if (harmonicTracingEnabled) {
    return traceHarmonic(r);
  }

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

glm::dvec3 RayTracer::traceHarmonic(const ray &cameraRay) const {
  const double hitEps = 1e-4;
  const double tMin = 1e-4;
  const double tMax = 120.0;
  const int maxSteps = 256;

  double t = tMin;
  double prevT = tMin;
  HarmonicSample prevSample = sampleField(cameraRay.at(t), harmonicMode);

  for (int step = 0; step < maxSteps && t < tMax; ++step) {
    const glm::dvec3 p = cameraRay.at(t);
    const HarmonicSample s = sampleField(p, harmonicMode);

    if (std::abs(s.value) <= hitEps) {
      const glm::dvec3 n = glm::normalize(s.grad);
      return shadeHarmonicPoint(scene.get(), cameraRay, p, n,
                                baseColorAt(p, harmonicMode));
    }

    if (step > 0 && s.value * prevSample.value < 0.0) {
      double lo = prevT;
      double hi = t;
      double flo = prevSample.value;
      for (int i = 0; i < 24; ++i) {
        const double mid = 0.5 * (lo + hi);
        const double fm = sampleField(cameraRay.at(mid), harmonicMode).value;
        if (flo * fm <= 0.0) {
          hi = mid;
        } else {
          lo = mid;
          flo = fm;
        }
      }

      const double rootT = 0.5 * (lo + hi);
      const glm::dvec3 hitP = cameraRay.at(rootT);
      const HarmonicSample hitS = sampleField(hitP, harmonicMode);
      const glm::dvec3 n = glm::normalize(hitS.grad);
      return shadeHarmonicPoint(scene.get(), cameraRay, hitP, n,
                                baseColorAt(hitP, harmonicMode));
    }

    const double stepSize = harnackStepSize(s, cameraRay.getDirection());
    prevT = t;
    prevSample = s;
    t += stepSize;
  }

  const double v = 0.5 * (cameraRay.getDirection().y + 1.0);
  return glm::clamp((1.0 - v) * glm::dvec3(0.01, 0.01, 0.02) +
                        v * glm::dvec3(0.08, 0.09, 0.12),
                    0.0, 1.0);
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
        const SceneObject *hit_object = i.getObject();
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
                    ray refract_ray(point, incident, glm::dvec3(1.0), ray::REFRACTION);

                    if (!overlapRefractionEnabled) {
                        if (d_dot_n > 0.0) {
                            n = -n;
                            n1 = n2;
                            n2 = 1.0;
                            d_dot_n = glm::dot(incident, n);
                        }
                    } else {
                        const bool exiting_by_normal = (d_dot_n > 0.0);
                        if (exiting_by_normal) {
                            n = -n;
                            d_dot_n = glm::dot(incident, n);
                        }

                        const double object_ior = std::max(1.0, m.index(i));
                        const bool was_inside_hit_object =
                            (hit_object != nullptr) && r.containsMedium(hit_object);

                        const bool legacy_air_to_object =
                            !was_inside_hit_object && !exiting_by_normal &&
                            (std::abs(r.currentMediumIor() - 1.0) <= 1e-9);
                        const bool legacy_object_to_air =
                            was_inside_hit_object && exiting_by_normal &&
                            (r.mediumDepth() == 1);

                        if (legacy_air_to_object) {
                            n1 = 1.0;
                            n2 = object_ior;
                            refract_ray.pushMedium(hit_object, object_ior);
                        } else if (legacy_object_to_air) {
                            n1 = object_ior;
                            n2 = 1.0;
                            refract_ray.setMediaFrom(r);
                            refract_ray.removeMedium(hit_object);
                        } else {
                            refract_ray.setMediaFrom(r);
                            n1 = r.currentMediumIor();
                            if (hit_object != nullptr) {
                                if (was_inside_hit_object) {
                                    refract_ray.removeMedium(hit_object);
                                } else {
                                    refract_ray.pushMedium(hit_object, object_ior);
                                }
                            }
                            n2 = refract_ray.currentMediumIor();
                        }
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
                        refract_ray.setPosition(point + REFRACT_EPSILON * refract_dir);
                        refract_ray.setDirection(refract_dir);
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
      m_bBufferReady(false), overlapRefractionEnabled(false),
      harmonicTracingEnabled(false), harmonicMode(HarmonicMode::RIEMANN) {
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
  overlapRefractionEnabled = false;
  harmonicTracingEnabled = false;
  harmonicMode = HarmonicMode::RIEMANN;
  loadedScenePath = (fn != nullptr) ? std::string(fn) : std::string();

  if (const char *envValue = std::getenv("RAY_ENABLE_OVERLAP_REFRACTION")) {
    overlapRefractionEnabled = strcmp(envValue, "0") != 0;
  }
  if (const char *envValue = std::getenv("RAY_ENABLE_HARMONIC_TRACING")) {
    harmonicTracingEnabled = strcmp(envValue, "0") != 0;
  }
  if (const char *envMode = std::getenv("RAY_HARMONIC_MODE")) {
    const std::string mode = toLowerCopy(std::string(envMode));
    if (mode == "gyroid") {
      harmonicMode = HarmonicMode::GYROID;
    } else if (mode == "riemann") {
      harmonicMode = HarmonicMode::RIEMANN;
    }
  } else {
    const std::string loweredPath = toLowerCopy(loadedScenePath);
    if (loweredPath.find("gyroid") != std::string::npos) {
      harmonicMode = HarmonicMode::GYROID;
    } else if (loweredPath.find("riemann") != std::string::npos) {
      harmonicMode = HarmonicMode::RIEMANN;
    }
  }

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
