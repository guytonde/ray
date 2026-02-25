#include "Portal.h"

#include <cmath>

#include <glm/geometric.hpp>

bool Portal::intersectLocal(ray &r, isect &i) const {
  const glm::dvec3 p = r.getPosition();
  const glm::dvec3 d = r.getDirection();

  constexpr double PLANE_EPS = 1e-12;
  if (std::abs(d.z) <= PLANE_EPS) {
    return false;
  }

  const double t = -p.z / d.z;
  if (t <= RAY_EPSILON) {
    return false;
  }

  const glm::dvec3 hit = r.at(t);
  bool inside = false;
  if (shapeType == ShapeType::Circle) {
    const double rr = radius * radius;
    inside = (hit.x * hit.x + hit.y * hit.y) <= rr;
  } else {
    inside = std::abs(hit.x) <= halfWidth && std::abs(hit.y) <= halfHeight;
  }

  if (!inside) {
    return false;
  }

  i.setObject(this);
  i.setMaterial(this->getMaterial());
  i.setT(t);
  i.setN(d.z > 0.0 ? glm::dvec3(0.0, 0.0, -1.0) : glm::dvec3(0.0, 0.0, 1.0));

  if (shapeType == ShapeType::Circle) {
    const double invR = (radius > 0.0) ? (1.0 / radius) : 0.0;
    i.setUVCoordinates(glm::dvec2(hit.x * 0.5 * invR + 0.5,
                                  hit.y * 0.5 * invR + 0.5));
  } else {
    const double invW = (halfWidth > 0.0) ? (1.0 / (2.0 * halfWidth)) : 0.0;
    const double invH = (halfHeight > 0.0) ? (1.0 / (2.0 * halfHeight)) : 0.0;
    i.setUVCoordinates(
        glm::dvec2(hit.x * invW + 0.5, hit.y * invH + 0.5));
  }

  return true;
}

BoundingBox Portal::ComputeLocalBoundingBox() {
  constexpr double zPad = 1e-6;
  BoundingBox box;

  if (shapeType == ShapeType::Circle) {
    box.setMin(glm::dvec3(-radius, -radius, -zPad));
    box.setMax(glm::dvec3(radius, radius, zPad));
  } else {
    box.setMin(glm::dvec3(-halfWidth, -halfHeight, -zPad));
    box.setMax(glm::dvec3(halfWidth, halfHeight, zPad));
  }

  return box;
}

bool Portal::teleportRay(const ray &inRay, double hitT, ray &outRay) const {
  if (!linked) {
    return false;
  }

  const glm::dvec3 inPosWorld = inRay.getPosition();
  const glm::dvec3 inEndWorld = inPosWorld + inRay.getDirection();

  const glm::dvec3 srcPosLocal = transform.globalToLocalCoords(inPosWorld);
  const glm::dvec3 srcEndLocal = transform.globalToLocalCoords(inEndWorld);
  glm::dvec3 srcDirLocal = srcEndLocal - srcPosLocal;
  if (glm::length(srcDirLocal) <= 1e-12) {
    return false;
  }
  srcDirLocal = glm::normalize(srcDirLocal);

  const glm::dvec3 hitWorld = inRay.at(hitT);
  const glm::dvec3 srcHitLocal = transform.globalToLocalCoords(hitWorld);

  glm::dvec3 dstPosLocal(srcHitLocal.x, srcHitLocal.y, 0.0);
  glm::dvec3 dstDirLocal(srcDirLocal.x, srcDirLocal.y, -srcDirLocal.z);
  if (glm::length(dstDirLocal) <= 1e-12) {
    return false;
  }
  dstDirLocal = glm::normalize(dstDirLocal);

  constexpr double EXIT_EPS = 1e-5;
  dstPosLocal.z = (dstDirLocal.z >= 0.0) ? EXIT_EPS : -EXIT_EPS;

  const glm::dvec3 outPosWorld = linked->transform.localToGlobalCoords(dstPosLocal);
  const glm::dvec3 outEndWorld =
      linked->transform.localToGlobalCoords(dstPosLocal + dstDirLocal);
  glm::dvec3 outDirWorld = outEndWorld - outPosWorld;
  if (glm::length(outDirWorld) <= 1e-12) {
    return false;
  }
  outDirWorld = glm::normalize(outDirWorld);

  outRay = ray(outPosWorld, outDirWorld, inRay.getAtten(), inRay.type());
  return true;
}
