#ifndef PORTAL_H__
#define PORTAL_H__

#include "../scene/scene.h"

class Portal : public SceneObject {
public:
  enum class ShapeType { Circle, Rectangle };

  Portal(Scene *scene, Material *mat, const MatrixTransform &transform,
         ShapeType shape, double radius, double width, double height)
      : SceneObject(scene, mat), shapeType(shape), radius(radius),
        halfWidth(width * 0.5), halfHeight(height * 0.5), linked(nullptr) {
    setTransform(transform);
  }

  bool intersectLocal(ray &r, isect &i) const override;
  bool hasBoundingBoxCapability() const override { return true; }
  BoundingBox ComputeLocalBoundingBox() override;

  void setLinkedPortal(Portal *other) { linked = other; }
  Portal *getLinkedPortal() const { return linked; }
  bool isLinked() const { return linked != nullptr; }

  bool teleportRay(const ray &inRay, double hitT, ray &outRay) const;

private:
  ShapeType shapeType;
  double radius;
  double halfWidth;
  double halfHeight;
  Portal *linked;
};

#endif // PORTAL_H__

