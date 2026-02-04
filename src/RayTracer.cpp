// The main ray tracer.

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
#include <string.h> // for memset

#include <limits>   // std::numeric_limits
#include <algorithm> // std::max

#include <fstream>
#include <iostream>

// std::vector<std::thread> workers;
// std::queue<std::pair<int,int>> workQueue;
// std::mutex workMutex;

// std::atomic<int> blocksDone{0};
// int blocksTotal = 0;
// std::atomic<bool> rendering{false};

using namespace std;
extern TraceUI *traceUI;

// Use this variable to decide if you want to print out debugging messages. Gets
// set in the "trace single ray" mode in TraceGLWindow, for example.
bool debugMode = false;

// Trace a top-level ray through pixel(i,j), i.e. normalized window coordinates
// (x,y), through the projection plane, and out into the scene. All we do is
// enter the main ray-tracing method, getting things started by plugging in an
// initial ray weight of (0.0,0.0,0.0) and an initial recursion depth of 0.

glm::dvec3 RayTracer::trace(double x, double y) {
  // Clear out the ray cache in the scene for debugging purposes,
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

  double x = double(i) / double(buffer_width);
  double y = double(j) / double(buffer_height);

  col = trace(x, y);
  col = glm::clamp(col, 0.0, 1.0);

  setPixel(i, j, col);   // single write path
  return col;
}

#define VERBOSE 0

// Do recursive ray tracing! You'll want to insert a lot of code here (or places
// called from here) to handle reflection, refraction, etc etc.
glm::dvec3 RayTracer::traceRay(ray &r, const glm::dvec3 &thresh, int depth, double &length) {
  isect i;
  glm::dvec3 colorC;

#if VERBOSE
  std::cerr << "== current depth: " << depth << std::endl;
#endif

  // Hard cutoff using the TraceUI threshold value synced into RayTracer::thresh in traceSetup()
  if (glm::compMax(thresh) < this->thresh) {
    length = std::numeric_limits<double>::infinity();
    return glm::dvec3(0.0, 0.0, 0.0);
  }

  if (scene->intersect(r, i)) {
    // Record hit distance for the caller.
    length = i.getT();

    // Local illumination (Phong etc. handled in Material::shade)
    const Material &m = i.getMaterial();
    colorC = m.shade(scene.get(), r, i);

    // Recursive rays
    if (depth > 0) {
      const glm::dvec3 P = r.at(i.getT());
      glm::dvec3 N = glm::normalize(i.getN());
      const glm::dvec3 D = glm::normalize(r.getDirection());

      // Make N oppose the incoming direction (common convention)
      if (glm::dot(D, N) > 0.0) N = -N;

      // ---- Reflection ----
      const glm::dvec3 Kr = m.kr(i);
      if (glm::compMax(Kr) > 0.0) {
        const glm::dvec3 newThresh = thresh * Kr;

        if (glm::compMax(newThresh) >= this->thresh) {
          const glm::dvec3 Rdir = glm::normalize(D - 2.0 * glm::dot(D, N) * N);
          ray rr(P + RAY_EPSILON * N, Rdir, r.getAtten() * Kr, ray::REFLECTION);

          double tR;
          colorC += Kr * traceRay(rr, newThresh, depth - 1, tR);
        }
      }

      // ---- Refraction (Snell) ----
      const glm::dvec3 Kt = m.kt(i);
      if (glm::compMax(Kt) > 0.0) {
        const glm::dvec3 newThresh = thresh * Kt;

        if (glm::compMax(newThresh) >= this->thresh) {
          double n1 = 1.0;
          double n2 = m.index(i);

          glm::dvec3 Nn = N;
          double cosI = -glm::dot(D, Nn);

          // If we're exiting, flip normal and swap indices
          if (cosI < 0.0) {
            cosI = -cosI;
            Nn = -Nn;
            std::swap(n1, n2);
          }

          const double eta = n1 / n2;
          const double k = 1.0 - eta * eta * (1.0 - cosI * cosI);

          // Total internal reflection => no transmitted ray
          if (k >= 0.0) {
            const glm::dvec3 Tdir =
                glm::normalize(eta * D + (eta * cosI - std::sqrt(k)) * Nn);

            // Start on transmitted side
            ray tr(P - RAY_EPSILON * Nn, Tdir, r.getAtten() * Kt, ray::REFRACTION);

            double tT;
            colorC += Kt * traceRay(tr, newThresh, depth - 1, tT);
          }
        }
      }
    }

  } else {
    // No intersection: background (CubeMap if enabled, else black).
    length = std::numeric_limits<double>::infinity();

    if (traceUI->cubeMap() && traceUI->getCubeMap() != nullptr) {
      colorC = traceUI->getCubeMap()->getColor(r);
    } else {
      colorC = glm::dvec3(0.0, 0.0, 0.0);
    }
  }

#if VERBOSE
  std::cerr << "== depth: " << depth + 1 << " done, returning: " << colorC
            << std::endl;
#endif
  return colorC;
    /*isect i;
    glm::dvec3 colorC;
  #if VERBOSE
    std::cerr << "== current depth: " << depth << std::endl;
  #endif

    if (scene->intersect(r, i)) {
      // YOUR CODE HERE

      // An intersection occurred!  We've got work to do. For now, this code gets
      // the material for the surface that was intersected, and asks that material
      // to provide a color for the ray.

      // This is a great place to insert code for recursive ray tracing. Instead
      // of just returning the result of shade(), add some more steps: add in the
      // contributions from reflected and refracted rays.

      const Material &m = i.getMaterial();
      colorC = m.shade(scene.get(), r, i);
    } else {
      // No intersection. This ray travels to infinity, so we color
      // it according to the background color, which in this (simple)
      // case is just black.
      //
      // FIXME: Add CubeMap support here.
      // TIPS: CubeMap object can be fetched from
      // traceUI->getCubeMap();
      //       Check traceUI->cubeMap() to see if cubeMap is loaded
      //       and enabled.

      colorC = glm::dvec3(0.0, 0.0, 0.0);
    }
  #if VERBOSE
    std::cerr << "== depth: " << depth + 1 << " done, returning: " << colorC
              << std::endl;
  #endif
    return colorC;*/
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

  // Check if fn ends in '.ray'
  bool isRay = false;
  const char *ext = strrchr(fn, '.');
  if (ext && !strcmp(ext, ".ray"))
    isRay = true;

  // Strip off filename, leaving only the path:
  string path(fn);
  if (path.find_last_of("\\/") == string::npos)
    path = ".";
  else
    path = path.substr(0, path.find_last_of("\\/"));

  if (isRay) {
    // .ray Parsing Path
    // Call this with 'true' for debug output from the tokenizer
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
    // JSON Parsing Path
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

  /*
   * Sync with TraceUI
   */

  threads = traceUI->getThreads();
  block_size = traceUI->getBlockSize();
  thresh = traceUI->getThreshold();
  samples = traceUI->getSuperSamples();
  aaThresh = traceUI->getAaThreshold();

  // YOUR CODE HERE
  // Additional init
  stopTrace = false;

  // Reset work queue + progress
  {
    std::lock_guard<std::mutex> lock(workMutex);
    while (!workQueue.empty()) workQueue.pop();
  }
  blocksDone = 0;
  blocksTotal = 0;
  rendering = false;

  const int bs = std::max(1, block_size);
  for (int y0 = 0; y0 < buffer_height; y0 += bs) {
    for (int x0 = 0; x0 < buffer_width; x0 += bs) {
      std::lock_guard<std::mutex> lock(workMutex);
      workQueue.push({x0, y0});
      blocksTotal++;
    }
  }
  // FIXME: Additional initializations
}

/*
 * RayTracer::traceImage
 *
 *	Trace the image and store the pixel data in RayTracer::buffer.
 *
 *	Arguments:
 *		w:	width of the image buffer
 *		h:	height of the image buffer
 *
 */
void RayTracer::traceImage(int w, int h) {
  // Always call traceSetup before rendering anything.
  traceSetup(w, h);

  // YOUR CODE HERE
  if (!sceneLoaded()) {
    rendering = false;
    return;
  }

  // join any previous run
  waitRender();
  stopTrace = false;
  rendering = true;

  const unsigned int nThreads =
      std::max(1u, std::min(threads, (unsigned int)MAX_THREADS)); // MAXTHREADS exists [file:1]

  workers.clear();
  workers.reserve(nThreads);

  auto workerFn = [&](unsigned int tid) {
    ray_thread_id = tid;                      // per-thread ray counter id [file:1]
    const int bs = std::max(1, block_size); // or blocksize in the starter [file:1]

    while (!stopTrace) {
      std::pair<int,int> block;
      {
        std::lock_guard<std::mutex> lock(workMutex);
        if (workQueue.empty()) break;
        block = workQueue.front();
        workQueue.pop();
      }

      const int x0 = block.first;
      const int y0 = block.second;

      for (int j = y0; j < std::min(y0 + bs, buffer_height) && !stopTrace; ++j) {
        for (int i = x0; i < std::min(x0 + bs, buffer_width) && !stopTrace; ++i) {
          tracePixel(i, j); // tracePixel computes + writes the RGB bytes [file:1]
        }
      }

      blocksDone++;
    }
  };

  for (unsigned int t = 0; t < nThreads; ++t)
    workers.emplace_back(workerFn, t);

  // return immediately (async), GUI will poll checkRender() [file:1]
  // FIXME: Start one or more threads for ray tracing
  //
  // TIPS: Ideally, the traceImage should be executed asynchronously,
  //       i.e. returns IMMEDIATELY after working threads are launched.
  //
  //       An asynchronous traceImage lets the GUI update your results
  //       while rendering.
}

int RayTracer::aaImage() {
  // YOUR CODE HERE
  if (!sceneLoaded()) return 0;

  const int n = std::max(1, samples);          // samples per side
  if (n == 1) return 0;

  int extraRays = 0;

  auto luminance = [](const glm::dvec3& c) {
    return 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
  };

  for (int j = 0; j < buffer_height && !stopTrace; ++j) {
    for (int i = 0; i < buffer_width && !stopTrace; ++i) {

      const double w = double(buffer_width);
      const double h = double(buffer_height);

      // 4 quick samples to detect an edge
      glm::dvec3 c0 = trace((i + 0.25) / w, (j + 0.25) / h);
      glm::dvec3 c1 = trace((i + 0.75) / w, (j + 0.25) / h);
      glm::dvec3 c2 = trace((i + 0.25) / w, (j + 0.75) / h);
      glm::dvec3 c3 = trace((i + 0.75) / w, (j + 0.75) / h);

      const double l0 = luminance(c0), l1 = luminance(c1), l2 = luminance(c2), l3 = luminance(c3);
      const double lmin = std::min(std::min(l0, l1), std::min(l2, l3));
      const double lmax = std::max(std::max(l0, l1), std::max(l2, l3));

      glm::dvec3 out = (c0 + c1 + c2 + c3) * 0.25;

      // If contrast is high, do full n*n stratified samples
      if ((lmax - lmin) > aaThresh) {
        glm::dvec3 sum(0.0);
        int cnt = 0;

        for (int sy = 0; sy < n; ++sy) {
          for (int sx = 0; sx < n; ++sx) {
            const double x = (i + (sx + 0.5) / double(n)) / w;
            const double y = (j + (sy + 0.5) / double(n)) / h;
            sum += trace(x, y);
            cnt++;
          }
        }

        out = sum / double(cnt);
        extraRays += (n * n - 4);
      }

      setPixel(i, j, glm::clamp(out, 0.0, 1.0));
    }
  }

  return extraRays;
  // FIXME: Implement Anti-aliasing here
  //
  // TIP: samples and aaThresh have been synchronized with TraceUI by
  //      RayTracer::traceSetup() function
  // return 0;
}

bool RayTracer::checkRender() {
  // YOUR CODE HERE
  if (!rendering) return true;
  if (stopTrace) return true;
  return blocksDone >= blocksTotal;
  // FIXME: Return true if tracing is done.
  //        This is a helper routine for GUI.
  //
  // TIPS: Introduce an array to track the status of each worker thread.
  //       This array is maintained by the worker threads.
  // return true;
}

void RayTracer::waitRender() {
  // YOUR CODE HERE
  for (auto &th : workers) {
    if (th.joinable()) th.join();
  }
  workers.clear();
  rendering = false;
  // FIXME: Wait until the rendering process is done.
  //        This function is essential if you are using an asynchronous
  //        traceImage implementation.
  //
  // TIPS: Join all worker threads here.
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
