#include "trimesh.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <float.h>
#include <string.h>
#include "../ui/TraceUI.h"
extern TraceUI *traceUI;
extern TraceUI *traceUI;

using namespace std;

Trimesh::~Trimesh() {
  for (auto f : faces)
    delete f;
}

// must add vertices, normals, and materials IN ORDER
void Trimesh::addVertex(const glm::dvec3 &v) { vertices.emplace_back(v); }

void Trimesh::addNormal(const glm::dvec3 &n) { normals.emplace_back(n); }

void Trimesh::addColor(const glm::dvec3 &c) { vertColors.emplace_back(c); }

void Trimesh::addUV(const glm::dvec2 &uv) { uvCoords.emplace_back(uv); }

// Returns false if the vertices a,b,c don't all exist
bool Trimesh::addFace(int a, int b, int c) {
  int vcnt = vertices.size();

  if (a >= vcnt || b >= vcnt || c >= vcnt)
    return false;

  TrimeshFace *newFace = new TrimeshFace(this, a, b, c);
  if (!newFace->degen)
    faces.push_back(newFace);
  else
    delete newFace;

  // Don't add faces to the scene's object list so we can cull by bounding
  // box
  return true;
}

// Check to make sure that if we have per-vertex materials or normals
// they are the right number.
const char *Trimesh::doubleCheck() {
  if (!vertColors.empty() && vertColors.size() != vertices.size())
    return "Bad Trimesh: Wrong number of vertex colors.";
  if (!uvCoords.empty() && uvCoords.size() != vertices.size())
    return "Bad Trimesh: Wrong number of UV coordinates.";
  if (!normals.empty() && normals.size() != vertices.size())
    return "Bad Trimesh: Wrong number of normals.";

  return 0;
}

bool Trimesh::intersectLocal(ray &r, isect &i) const {
  bool have_one = false;
  for (auto face : faces) {
    isect cur;
    if (face->intersectLocal(r, cur)) {
      if (!have_one || (cur.getT() < i.getT())) {
        i = cur;
        have_one = true;
      }
    }
  }
  if (!have_one)
    i.setT(1000.0);
  return have_one;
}

bool TrimeshFace::intersect(ray &r, isect &i) const {
  return intersectLocal(r, i);
}


// Intersect ray r with the triangle abc.  If it hits returns true,
// and put the parameter in t and the barycentric coordinates of the
// intersection in u (alpha) and v (beta).
bool TrimeshFace::intersectLocal(ray &r, isect &i) const {
    // Get vertices of the triangle
    const glm::dvec3& a = parent->vertices[ids[0]];
    const glm::dvec3& b = parent->vertices[ids[1]];
    const glm::dvec3& c = parent->vertices[ids[2]];
    
    // Compute edges
    glm::dvec3 edge1 = b - a;
    glm::dvec3 edge2 = c - a;
    
    // Begin calculating determinant using Möller–Trumbore algorithm
    glm::dvec3 h = glm::cross(r.getDirection(), edge2);
    double det = glm::dot(edge1, h);
    
    // If determinant is near zero, ray lies in plane of triangle
    const double EPSILON = 0.0000001;
    if (det > -EPSILON && det < EPSILON)
        return false;
    
    double inv_det = 1.0 / det;
    glm::dvec3 s = r.getPosition() - a;
    double u = inv_det * glm::dot(s, h);
    
    // Check if intersection is outside triangle (barycentric u)
    if (u < 0.0 || u > 1.0)
        return false;
    
    glm::dvec3 q = glm::cross(s, edge1);
    double v = inv_det * glm::dot(r.getDirection(), q);
    
    // Check if intersection is outside triangle (barycentric v)
    if (v < 0.0 || u + v > 1.0)
        return false;
    
    // Compute t to find out where the intersection point is on the line
    double t = inv_det * glm::dot(edge2, q);
    
    // Ray intersection
    if (t > EPSILON) {
        i.setT(t);
        i.setObject(this->parent);
        
        // Barycentric coordinates: alpha = 1-u-v, beta = u, gamma = v
        double alpha = 1.0 - u - v;
        double beta = u;
        double gamma = v;
        
        // Interpolate normal if parent mesh has per-vertex normals
        if (!parent->normals.empty()) {
            const glm::dvec3& na = parent->normals[ids[0]];
            const glm::dvec3& nb = parent->normals[ids[1]];
            const glm::dvec3& nc = parent->normals[ids[2]];
            
            glm::dvec3 interpolated_normal = alpha * na + beta * nb + gamma * nc;
            i.setN(glm::normalize(interpolated_normal));
        } else {
            // Use face normal
            i.setN(normal);
        }
        
        // Handle UV coordinates if present
        if (!parent->uvCoords.empty()) {
            const glm::dvec2& uva = parent->uvCoords[ids[0]];
            const glm::dvec2& uvb = parent->uvCoords[ids[1]];
            const glm::dvec2& uvc = parent->uvCoords[ids[2]];
            
            glm::dvec2 interpolated_uv = alpha * uva + beta * uvb + gamma * uvc;
            i.setUVCoordinates(interpolated_uv);
        }
        // Handle vertex colors if present (and UV coords not present)
        else if (!parent->vertColors.empty()) {
            const glm::dvec3& ca = parent->vertColors[ids[0]];
            const glm::dvec3& cb = parent->vertColors[ids[1]];
            const glm::dvec3& cc = parent->vertColors[ids[2]];
            
            glm::dvec3 interpolated_color = alpha * ca + beta * cb + gamma * cc;
            
            // Create a new material with the interpolated color
            Material* new_mat = new Material(parent->getMaterial());
            new_mat->setDiffuse(interpolated_color);
            i.setMaterial(*new_mat);
        }
        
        return true;
    }
    
    return false; // Line intersection but not ray intersection
}


// Once all the verts and faces are loaded, per vertex normals can be
// generated by averaging the normals of the neighboring faces.
void Trimesh::generateNormals() {
  int cnt = vertices.size();
  normals.resize(cnt);
  std::vector<int> numFaces(cnt, 0);

  for (auto face : faces) {
    glm::dvec3 faceNormal = face->getNormal();

    for (int i = 0; i < 3; ++i) {
      normals[(*face)[i]] += faceNormal;
      ++numFaces[(*face)[i]];
    }
  }

  for (int i = 0; i < cnt; ++i) {
    if (numFaces[i])
      normals[i] /= numFaces[i];
  }

  vertNorms = true;
}

