#include <cmath>

#include "../SceneObjects/Portal.h"
#include "../ui/TraceUI.h"
#include "light.h"
#include "scene.h"
#include <glm/gtx/extended_min_max.hpp>
#include <glm/gtx/io.hpp>
#include <iostream>

using namespace std;
extern TraceUI *traceUI;

bool Geometry::intersect(ray &r, isect &i) const {
  double tmin, tmax;
  if (hasBoundingBoxCapability() && !(bounds.intersect(r, tmin, tmax)))
    return false;
  // Transform the ray into the object's local coordinate space
  glm::dvec3 pos = transform.globalToLocalCoords(r.getPosition());
  glm::dvec3 dir =
      transform.globalToLocalCoords(r.getPosition() + r.getDirection()) - pos;
  double length = glm::length(dir);
  dir = glm::normalize(dir);
  // Backup World pos/dir, and switch to local pos/dir
  glm::dvec3 Wpos = r.getPosition();
  glm::dvec3 Wdir = r.getDirection();
  r.setPosition(pos);
  r.setDirection(dir);
  bool rtrn = false;
  if (intersectLocal(r, i)) {
    // Transform the intersection point & normal returned back into
    // global space.
    i.setN(glm::normalize(transform.localToGlobalCoordsNormal(i.getN())));
    i.setT(i.getT() / length);
    rtrn = true;
  }
  // Restore World pos/dir
  r.setPosition(Wpos);
  r.setDirection(Wdir);
  return rtrn;
}

bool Geometry::hasBoundingBoxCapability() const {
  // by default, primitives do not have to specify a bounding box. If this
  // method returns true for a primitive, then either the ComputeBoundingBox()
  // or the ComputeLocalBoundingBox() method must be implemented.

  // If no bounding box capability is supported for an object, that object will
  // be checked against every single ray drawn. This should be avoided whenever
  // possible, but this possibility exists so that new primitives will not have
  // to have bounding boxes implemented for them.
  return false;
}

void Geometry::ComputeBoundingBox() {
  // take the object's local bounding box, transform all 8 points on it,
  // and use those to find a new bounding box.

  BoundingBox localBounds = ComputeLocalBoundingBox();

  glm::dvec3 min = localBounds.getMin();
  glm::dvec3 max = localBounds.getMax();

  glm::dvec4 v, newMax, newMin;

  v = transform.localToGlobalCoords(glm::dvec4(min[0], min[1], min[2], 1));
  newMax = v;
  newMin = v;
  v = transform.localToGlobalCoords(glm::dvec4(max[0], min[1], min[2], 1));
  newMax = glm::max(newMax, v);
  newMin = glm::min(newMin, v);
  v = transform.localToGlobalCoords(glm::dvec4(min[0], max[1], min[2], 1));
  newMax = glm::max(newMax, v);
  newMin = glm::min(newMin, v);
  v = transform.localToGlobalCoords(glm::dvec4(max[0], max[1], min[2], 1));
  newMax = glm::max(newMax, v);
  newMin = glm::min(newMin, v);
  v = transform.localToGlobalCoords(glm::dvec4(min[0], min[1], max[2], 1));
  newMax = glm::max(newMax, v);
  newMin = glm::min(newMin, v);
  v = transform.localToGlobalCoords(glm::dvec4(max[0], min[1], max[2], 1));
  newMax = glm::max(newMax, v);
  newMin = glm::min(newMin, v);
  v = transform.localToGlobalCoords(glm::dvec4(min[0], max[1], max[2], 1));
  newMax = glm::max(newMax, v);
  newMin = glm::min(newMin, v);
  v = transform.localToGlobalCoords(glm::dvec4(max[0], max[1], max[2], 1));
  newMax = glm::max(newMax, v);
  newMin = glm::min(newMin, v);

  bounds.setMax(glm::dvec3(newMax));
  bounds.setMin(glm::dvec3(newMin));
}

Scene::Scene() : bvh(nullptr) { ambientIntensity = glm::dvec3(0, 0, 0); }

Scene::~Scene() {
  for (auto &obj : objects)
    delete obj;
  for (auto &light : lights)
    delete light;
}

void Scene::add(Geometry *obj) {
  obj->ComputeBoundingBox();
  sceneBounds.merge(obj->getBoundingBox());
  objects.emplace_back(obj);
  bvh.reset();
  nonBoundedObjects.clear();
}

void Scene::add(Light *light) { lights.emplace_back(light); }

void Scene::buildAcceleration(int maxDepth, int leafSize) {
  std::vector<Geometry *> boundedObjects;
  boundedObjects.reserve(objects.size());
  nonBoundedObjects.clear();
  nonBoundedObjects.reserve(objects.size());

  for (auto *obj : objects) {
    obj->buildAcceleration(maxDepth, leafSize);
    if (obj->hasBoundingBoxCapability()) {
      boundedObjects.push_back(obj);
    } else {
      nonBoundedObjects.push_back(obj);
    }
  }

  if (boundedObjects.empty()) {
    bvh.reset();
    return;
  }

  bvh = std::make_unique<BVH<Geometry>>(boundedObjects, maxDepth, leafSize);
}

// Get any intersection with an object.  Return information about the
// intersection through the reference parameter.
bool Scene::intersect(ray &r, isect &i) const {
  const bool useBvh =
      (traceUI != nullptr) && traceUI->kdSwitch() && bvh && !bvh->empty();

  auto traceNearest = [&](ray &queryRay, isect &hitIsect) -> bool {
    bool haveOne = false;
    if (useBvh) {
      haveOne = bvh->intersect(queryRay, hitIsect);
      for (const auto *obj : nonBoundedObjects) {
        isect cur;
        if (obj->intersect(queryRay, cur)) {
          if (!haveOne || cur.getT() < hitIsect.getT()) {
            hitIsect = cur;
            haveOne = true;
          }
        }
      }
    } else {
      for (const auto &obj : objects) {
        isect cur;
        if (obj->intersect(queryRay, cur)) {
          if (!haveOne || (cur.getT() < hitIsect.getT())) {
            hitIsect = cur;
            haveOne = true;
          }
        }
      }
    }

    if (!haveOne) {
      hitIsect.setT(1000.0);
    }
    return haveOne;
  };

  constexpr int MAX_PORTAL_HOPS = 64;
  ray segmentRay = r;

  for (int hop = 0; hop <= MAX_PORTAL_HOPS; ++hop) {
    isect curHit;
    const bool haveOne = traceNearest(segmentRay, curHit);
    if (!haveOne) {
      i = curHit;
      r = segmentRay;
      if (TraceUI::m_debug) {
        addToIntersectCache(std::make_pair(new ray(r), new isect(i)));
      }
      return false;
    }

    const SceneObject *hitObj = curHit.getObject();
    const auto *portal = dynamic_cast<const Portal *>(hitObj);
    if (portal && portal->isLinked()) {
      ray teleportedRay = segmentRay;
      if (portal->teleportRay(segmentRay, curHit.getT(), teleportedRay)) {
        segmentRay = teleportedRay;
        continue;
      }
    }

    i = curHit;
    r = segmentRay;
    if (TraceUI::m_debug) {
      addToIntersectCache(std::make_pair(new ray(r), new isect(i)));
    }
    return true;
  }

  i.setT(1000.0);
  r = segmentRay;
  if (TraceUI::m_debug) {
    addToIntersectCache(std::make_pair(new ray(r), new isect(i)));
  }
  return false;
}

TextureMap *Scene::getTexture(string name) {
  auto itr = textureCache.find(name);
  if (itr == textureCache.end()) {
    textureCache[name].reset(new TextureMap(name));
    return textureCache[name].get();
  }
  return itr->second.get();
}
