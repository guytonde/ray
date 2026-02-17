#pragma once

#include "bbox.h"
#include "ray.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

template <typename Obj> class BVH {
public:
  BVH() = default;

  BVH(const std::vector<Obj *> &objects, int maxDepth, int leafSize)
      : maxDepth(maxDepth), leafSize(leafSize) {
    std::vector<Obj *> candidates;
    candidates.reserve(objects.size());
    for (Obj *obj : objects) {
      if (obj->hasBoundingBoxCapability()) {
        candidates.push_back(obj);
      }
    }
    if (!candidates.empty()) {
      root = build(candidates, 0);
    }
  }

  bool intersectClosest(ray &r, isect &best) const {
    if (!root) {
      return false;
    }
    double bestT = std::numeric_limits<double>::infinity();
    bool hit = traverse(*root, r, best, bestT);
    if (hit) {
      best.setT(bestT);
    }
    return hit;
  }

private:
  struct Node {
    BoundingBox bounds;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
    std::vector<Obj *> objects;
    bool isLeaf = false;
  };

  std::unique_ptr<Node> root;
  int maxDepth = 15;
  int leafSize = 10;

  static BoundingBox mergedBounds(const std::vector<Obj *> &objects) {
    BoundingBox box;
    for (Obj *obj : objects) {
      box.merge(obj->getBoundingBox());
    }
    return box;
  }

  static int widestAxis(const BoundingBox &box) {
    glm::dvec3 extent = box.getMax() - box.getMin();
    if (extent.x >= extent.y && extent.x >= extent.z) {
      return 0;
    }
    if (extent.y >= extent.z) {
      return 1;
    }
    return 2;
  }

  static double centerOnAxis(const BoundingBox &box, int axis) {
    return 0.5 * (box.getMin()[axis] + box.getMax()[axis]);
  }

  std::unique_ptr<Node> build(std::vector<Obj *> &objects, int depth) const {
    auto node = std::make_unique<Node>();
    node->bounds = mergedBounds(objects);

    if (depth >= maxDepth || static_cast<int>(objects.size()) <= leafSize) {
      node->isLeaf = true;
      node->objects = objects;
      return node;
    }

    int axis = widestAxis(node->bounds);
    std::sort(objects.begin(), objects.end(), [axis](Obj *a, Obj *b) {
      return centerOnAxis(a->getBoundingBox(), axis) <
             centerOnAxis(b->getBoundingBox(), axis);
    });

    std::size_t mid = objects.size() / 2;
    if (mid == 0 || mid >= objects.size()) {
      node->isLeaf = true;
      node->objects = objects;
      return node;
    }

    std::vector<Obj *> leftList(objects.begin(), objects.begin() + mid);
    std::vector<Obj *> rightList(objects.begin() + mid, objects.end());
    node->left = build(leftList, depth + 1);
    node->right = build(rightList, depth + 1);
    return node;
  }

  static bool nodeBoxHit(const BoundingBox &box, const ray &r, double bestT,
                         double &tNear) {
    double tmin = 0.0;
    double tmax = 0.0;
    if (!box.intersect(r, tmin, tmax)) {
      return false;
    }
    tNear = std::max(tmin, 0.0);
    return tNear <= bestT;
  }

  bool traverse(const Node &node, ray &r, isect &best, double &bestT) const {
    double nodeNear = 0.0;
    if (!nodeBoxHit(node.bounds, r, bestT, nodeNear)) {
      return false;
    }

    bool hit = false;

    if (node.isLeaf) {
      for (Obj *obj : node.objects) {
        isect cur;
        if (obj->intersect(r, cur) && cur.getT() < bestT) {
          best = cur;
          bestT = cur.getT();
          hit = true;
        }
      }
      return hit;
    }

    const Node *first = node.left.get();
    const Node *second = node.right.get();

    double leftNear = std::numeric_limits<double>::infinity();
    double rightNear = std::numeric_limits<double>::infinity();
    bool leftHit =
        first ? nodeBoxHit(first->bounds, r, bestT, leftNear) : false;
    bool rightHit =
        second ? nodeBoxHit(second->bounds, r, bestT, rightNear) : false;

    if (leftHit && rightHit && rightNear < leftNear) {
      std::swap(first, second);
      std::swap(leftNear, rightNear);
    }

    if (first && (leftHit || rightHit)) {
      hit |= traverse(*first, r, best, bestT);
    }
    if (second) {
      double nearSecond = std::numeric_limits<double>::infinity();
      if (nodeBoxHit(second->bounds, r, bestT, nearSecond)) {
        hit |= traverse(*second, r, best, bestT);
      }
    }

    return hit;
  }
};
