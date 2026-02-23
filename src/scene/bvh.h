#pragma once

#include "bbox.h"
#include "ray.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

template <typename Primitive> class BVH {
public:
  BVH() = default;

  BVH(const std::vector<Primitive *> &primitives, int maxDepth, int leafSize) {
    build(primitives, maxDepth, leafSize);
  }

  void build(const std::vector<Primitive *> &primitives, int maxDepth,
             int leafSize) {
    primitives_.clear();
    nodes_.clear();

    maxDepth_ = std::max(1, maxDepth);
    leafSize_ = std::max(1, leafSize);

    primitives_.reserve(primitives.size());
    for (Primitive *p : primitives) {
      if (p != nullptr) {
        primitives_.push_back(p);
      }
    }

    if (primitives_.empty()) {
      return;
    }

    nodes_.reserve(primitives_.size() * 2);
    buildNode(0, primitives_.size(), 0);
  }

  bool empty() const { return nodes_.empty(); }

  bool intersect(ray &r, isect &i) const {
    if (nodes_.empty()) {
      i.setT(1000.0);
      return false;
    }

    bool haveHit = false;
    double bestT = std::numeric_limits<double>::infinity();
    std::vector<int> stack;
    stack.reserve(64);
    stack.push_back(0);

    while (!stack.empty()) {
      const int nodeIndex = stack.back();
      stack.pop_back();
      const Node &node = nodes_[nodeIndex];

      double tmin, tmax;
      if (!node.bounds.intersect(r, tmin, tmax)) {
        continue;
      }
      if (haveHit && tmin > bestT) {
        continue;
      }

      if (node.isLeaf()) {
        const size_t end = node.start + node.count;
        for (size_t idx = node.start; idx < end; ++idx) {
          isect cur;
          if (primitives_[idx]->intersect(r, cur)) {
            const double curT = cur.getT();
            if (!haveHit || curT < bestT) {
              bestT = curT;
              i = cur;
              haveHit = true;
            }
          }
        }
        continue;
      }

      const Node &left = nodes_[node.left];
      const Node &right = nodes_[node.right];

      double ltmin = 0.0, ltmax = 0.0;
      double rtmin = 0.0, rtmax = 0.0;
      const bool hitLeft = left.bounds.intersect(r, ltmin, ltmax);
      const bool hitRight = right.bounds.intersect(r, rtmin, rtmax);

      if (hitLeft && hitRight) {
        const bool leftFirst = ltmin <= rtmin;
        const int first = leftFirst ? node.left : node.right;
        const int second = leftFirst ? node.right : node.left;
        const double firstT = leftFirst ? ltmin : rtmin;
        const double secondT = leftFirst ? rtmin : ltmin;

        if (!haveHit || secondT <= bestT) {
          stack.push_back(second);
        }
        if (!haveHit || firstT <= bestT) {
          stack.push_back(first);
        }
      } else if (hitLeft) {
        if (!haveHit || ltmin <= bestT) {
          stack.push_back(node.left);
        }
      } else if (hitRight) {
        if (!haveHit || rtmin <= bestT) {
          stack.push_back(node.right);
        }
      }
    }

    if (!haveHit) {
      i.setT(1000.0);
    }
    return haveHit;
  }

private:
  static constexpr int kNumBins = 12;
  static constexpr double kTraversalCost = 1.0;
  static constexpr double kIntersectCost = 1.0;

  struct Node {
    BoundingBox bounds;
    int left = -1;
    int right = -1;
    size_t start = 0;
    size_t count = 0;

    bool isLeaf() const { return left < 0 && right < 0; }
  };

  static glm::dvec3 center(const BoundingBox &b) {
    return 0.5 * (b.getMin() + b.getMax());
  }

  static double surfaceArea(const BoundingBox &b) {
    const glm::dvec3 d = b.getMax() - b.getMin();
    if (d.x <= 0.0 || d.y <= 0.0 || d.z <= 0.0) {
      return 0.0;
    }
    return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x);
  }

  size_t medianSplit(size_t begin, size_t end, int axis) {
    const size_t mid = begin + (end - begin) / 2;
    auto beginIt = primitives_.begin() + static_cast<std::ptrdiff_t>(begin);
    auto midIt = primitives_.begin() + static_cast<std::ptrdiff_t>(mid);
    auto endIt = primitives_.begin() + static_cast<std::ptrdiff_t>(end);
    std::nth_element(
        beginIt, midIt, endIt,
        [axis](const Primitive *a, const Primitive *b) {
          return center(a->getBoundingBox())[axis] <
                 center(b->getBoundingBox())[axis];
        });
    return mid;
  }

  bool findSahSplit(size_t begin, size_t end, const BoundingBox &nodeBounds,
                    const BoundingBox &centroidBounds, int &bestAxis,
                    double &bestSplitPosition) const {
    const size_t primitiveCount = end - begin;
    const double parentArea = surfaceArea(nodeBounds);
    if (parentArea <= 0.0) {
      return false;
    }

    const double leafCost = kIntersectCost * double(primitiveCount);
    double bestCost = std::numeric_limits<double>::infinity();
    bool found = false;

    for (int axis = 0; axis < 3; ++axis) {
      const double cmin = centroidBounds.getMin()[axis];
      const double cmax = centroidBounds.getMax()[axis];
      const double extent = cmax - cmin;
      if (extent <= 1e-12) {
        continue;
      }

      struct BinData {
        BoundingBox bounds;
        size_t count = 0;
      };
      std::array<BinData, kNumBins> bins;

      for (size_t idx = begin; idx < end; ++idx) {
        const double c = center(primitives_[idx]->getBoundingBox())[axis];
        int bin = int(((c - cmin) / extent) * kNumBins);
        bin = std::clamp(bin, 0, kNumBins - 1);
        bins[bin].count++;
        bins[bin].bounds.merge(primitives_[idx]->getBoundingBox());
      }

      std::array<BoundingBox, kNumBins> leftBounds;
      std::array<BoundingBox, kNumBins> rightBounds;
      std::array<size_t, kNumBins> leftCounts{};
      std::array<size_t, kNumBins> rightCounts{};

      BoundingBox leftAccum;
      size_t leftCount = 0;
      for (int i = 0; i < kNumBins; ++i) {
        leftCount += bins[i].count;
        if (bins[i].count > 0) {
          leftAccum.merge(bins[i].bounds);
        }
        leftBounds[i] = leftAccum;
        leftCounts[i] = leftCount;
      }

      BoundingBox rightAccum;
      size_t rightCount = 0;
      for (int i = kNumBins - 1; i >= 0; --i) {
        rightCount += bins[i].count;
        if (bins[i].count > 0) {
          rightAccum.merge(bins[i].bounds);
        }
        rightBounds[i] = rightAccum;
        rightCounts[i] = rightCount;
      }

      for (int split = 0; split < kNumBins - 1; ++split) {
        const size_t lCount = leftCounts[split];
        const size_t rCount = rightCounts[split + 1];
        if (lCount == 0 || rCount == 0) {
          continue;
        }

        const double lArea = surfaceArea(leftBounds[split]);
        const double rArea = surfaceArea(rightBounds[split + 1]);
        const double splitCost =
            kTraversalCost + kIntersectCost * ((lArea / parentArea) * lCount +
                                               (rArea / parentArea) * rCount);
        if (splitCost < bestCost) {
          bestCost = splitCost;
          bestAxis = axis;
          bestSplitPosition =
              cmin + (extent * (double(split + 1) / double(kNumBins)));
          found = true;
        }
      }
    }

    if (!found) {
      return false;
    }
    return bestCost < leafCost;
  }

  int buildNode(size_t begin, size_t end, int depth) {
    Node node;
    for (size_t idx = begin; idx < end; ++idx) {
      node.bounds.merge(primitives_[idx]->getBoundingBox());
    }

    const size_t count = end - begin;
    const int nodeIndex = static_cast<int>(nodes_.size());
    nodes_.push_back(node);

    if (count <= static_cast<size_t>(leafSize_) || depth >= maxDepth_) {
      nodes_[nodeIndex].start = begin;
      nodes_[nodeIndex].count = count;
      return nodeIndex;
    }

    BoundingBox centroidBounds;
    for (size_t idx = begin; idx < end; ++idx) {
      const glm::dvec3 c = center(primitives_[idx]->getBoundingBox());
      centroidBounds.merge(BoundingBox(c, c));
    }

    const glm::dvec3 diag = centroidBounds.getMax() - centroidBounds.getMin();
    int axis = 0;
    if (diag.y > diag.x && diag.y >= diag.z) {
      axis = 1;
    } else if (diag.z > diag.x && diag.z >= diag.y) {
      axis = 2;
    }

    if (std::abs(diag[axis]) < 1e-12) {
      nodes_[nodeIndex].start = begin;
      nodes_[nodeIndex].count = count;
      return nodeIndex;
    }

    int sahAxis = -1;
    double splitPos = 0.0;
    bool useSah =
        findSahSplit(begin, end, node.bounds, centroidBounds, sahAxis, splitPos);

    size_t mid = begin;
    if (useSah) {
      axis = sahAxis;
      auto beginIt = primitives_.begin() + static_cast<std::ptrdiff_t>(begin);
      auto endIt = primitives_.begin() + static_cast<std::ptrdiff_t>(end);
      auto midIt = std::partition(
          beginIt, endIt, [axis, splitPos](const Primitive *p) {
            return center(p->getBoundingBox())[axis] < splitPos;
          });
      mid = static_cast<size_t>(std::distance(primitives_.begin(), midIt));
      if (mid == begin || mid == end) {
        mid = medianSplit(begin, end, axis);
      }
    } else {
      mid = medianSplit(begin, end, axis);
    }

    if (mid == begin || mid == end) {
      nodes_[nodeIndex].start = begin;
      nodes_[nodeIndex].count = count;
      return nodeIndex;
    }

    const int left = buildNode(begin, mid, depth + 1);
    const int right = buildNode(mid, end, depth + 1);
    nodes_[nodeIndex].left = left;
    nodes_[nodeIndex].right = right;
    nodes_[nodeIndex].start = 0;
    nodes_[nodeIndex].count = 0;
    return nodeIndex;
  }

  int maxDepth_ = 1;
  int leafSize_ = 1;
  std::vector<Primitive *> primitives_;
  std::vector<Node> nodes_;
};
