#include <algorithm>
#include <filesystem>
#include <memory>
#define _CRT_SECURE_NO_WARNINGS 1
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <omp.h>
#include <random>
#include <vector>

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#ifndef M_PI
#define M_PI 3.14159265358979323856
#endif

static std::default_random_engine engine[256];
thread_local std::uniform_real_distribution<double> uniform(0, 1);

const double eps = 1e-10;

inline double sqr(double x) { return x * x; };

class Vector {
  public:
    explicit Vector(double x = 0, double y = 0, double z = 0) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
    }
    double norm() const { return sqrt(norm2()); }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
        data[2] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double &operator[](int i) { return data[i]; };

  private:
    double data[3];
};

inline Vector operator+(const Vector &a, const Vector &b) {
    return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
inline Vector operator-(const Vector &a, const Vector &b) {
    return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
inline Vector operator*(const double a, const Vector &b) {
    return Vector(a * b[0], a * b[1], a * b[2]);
}
inline Vector operator*(const Vector &a, const double b) { return b * a; }
inline Vector operator*(const Vector &a, const Vector &b) {
    return Vector(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}
inline Vector operator/(const Vector &a, const double b) {
    return Vector(a[0] / b, a[1] / b, a[2] / b);
}
inline double dot(const Vector &a, const Vector &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline Vector cross(const Vector &a, const Vector &b) {
    return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                  a[0] * b[1] - a[1] * b[0]);
}

class Matrix {
  public:
    Vector operator[](int i) const { return data[i]; }
    Vector &operator[](int i) { return data[i]; }

  private:
    Vector data[3];
};

Matrix create_rotation_matrix(Vector k,
                              double theta) { // theta in radians pls
    k.normalize();                            // the axis must be a unit vector

    double c = cos(theta);
    double s = sin(theta);

    double k_x = k[0], k_y = k[1], k_z = k[2];

    // reference: Rodrigues rotation formula Wikipedia right at the end
    Matrix m;
    m[0][0] = c + (1 - c) * k_x * k_x;
    m[0][1] = (1 - c) * k_x * k_y - s * k_z;
    m[0][2] = (1 - c) * k_x * k_z + s * k_y;

    m[1][0] = (1 - c) * k_x * k_y + s * k_z;
    m[1][1] = c + (1 - c) * k_y * k_y;
    m[1][2] = (1 - c) * k_y * k_z - s * k_x;

    m[2][0] = (1 - c) * k_x * k_z - s * k_y;
    m[2][1] = (1 - c) * k_y * k_z + s * k_x;
    m[2][2] = c + (1 - c) * k_z * k_z;

    return m;
}

Vector operator*(const Matrix &a, const Vector &b) {
    return Vector(dot(a[0], b), dot(a[1], b), dot(a[2], b));
}

class Ray {
  public:
    Ray(const Vector &origin, const Vector &unit_direction, double time)
        : O(origin), u(unit_direction), time(time){};
    Vector O, u;
    double time;
};

class Object {
  public:
    Object(const Vector &albedo, bool mirror = false, bool transparent = false,
           bool is_light = false, double n = 1.5,
           bool invert_normals = false) // by default glass
        : albedo(albedo), mirror(mirror), transparent(transparent),
          is_light(is_light), n(n), invert_normals(invert_normals){};

    virtual Object *clone() const = 0;
    virtual bool intersect(const Ray &ray, Vector &P, double &t, Vector &N,
                           Vector &albedo) const = 0;
    virtual void scale_translate(double s, const Vector &t) = 0;
    virtual void rotate(const Matrix &m) = 0;
    virtual void compute_centroid() = 0;
    virtual void compute_bounding_box() = 0;

    Vector albedo, velocity = Vector(0, 0, 0);
    bool mirror, transparent, is_light;
    double n;
    bool invert_normals;
    struct BBox {
        Vector min, max;
        int longest_axis() const {
            Vector diag = max - min;
            if (diag[0] > diag[1] - eps && diag[0] > diag[2] - eps)
                return 0;
            if (diag[1] > diag[0] - eps && diag[1] > diag[2] - eps)
                return 1;
            return 2;
        }
        bool intersect_ray(const Ray &ray, double &t) const {
            double tx_min = (min[0] - ray.O[0]) / ray.u[0];
            double tx_max = (max[0] - ray.O[0]) / ray.u[0];
            if (tx_max < tx_min - eps)
                std::swap(tx_min, tx_max);
            double ty_min = (min[1] - ray.O[1]) / ray.u[1];
            double ty_max = (max[1] - ray.O[1]) / ray.u[1];
            if (ty_max < ty_min - eps)
                std::swap(ty_min, ty_max);
            double tz_min = (min[2] - ray.O[2]) / ray.u[2];
            double tz_max = (max[2] - ray.O[2]) / ray.u[2];
            if (tz_max < tz_min - eps)
                std::swap(tz_min, tz_max);
            double t_min = std::max(tx_min, std::max(ty_min, tz_min));
            double t_max = std::min(tx_max, std::min(ty_max, tz_max));
            if (t_max < 0 || t_min > t_max)
                return false;
            t = t_min;
            return true;
        }
    } bbox;
    Vector centroid{0, 0, 0};
};

class Sphere : public Object {
  public:
    Sphere(const Vector &center, double radius, const Vector &albedo,
           bool mirror = false, bool transparent = false, bool is_light = false,
           double n = 1.5, bool invert_normals = false)
        : ::Object(albedo, mirror, transparent, is_light, n, invert_normals),
          C(center), R(radius) {
        compute_centroid();
        compute_bounding_box();
    };

    Sphere *clone() const override { return new Sphere(*this); }
    // returns true iif there is an intersection between the ray and the sphere
    // if there is an intersection, also computes the point of intersection P,
    // t>=0 the distance between the ray origin and P (i.e., the parameter along
    // the ray) and the unit normal N
    bool intersect(const Ray &ray, Vector &P, double &t, Vector &N,
                   Vector &albedo) const override {
        Vector real_C = C + velocity * ray.time;
        double delta = sqr(dot(ray.u, ray.O - real_C)) -
                       ((ray.O - real_C).norm2() - sqr(R));
        if (delta < -eps)
            return false;
        if (-eps < delta && delta < eps)
            delta = 0;
        double t1 = dot(ray.u, real_C - ray.O) + sqrt(delta);
        double t2 = dot(ray.u, real_C - ray.O) - sqrt(delta);
        if (t1 < eps)
            return false;
        if (t2 > eps)
            t = t2;
        else
            t = t1;
        P = ray.O + t * ray.u;
        N = P - real_C;
        N.normalize();
        if (invert_normals)
            N = N * -1.0;
        albedo = this->albedo;
        return true;
    }
    void scale_translate(double s, const Vector &t) override {
        C = C * s + t;
        R *= s;
        bbox.min = bbox.min * s + t;
        bbox.max = bbox.max * s + t;
        centroid = C;
    }
    void rotate(const Matrix &) override {
        return; // sphere doesn't do anything when rotated
    }
    void compute_centroid() override { centroid = C; }
    void compute_bounding_box() override {
        bbox.max = C + R * Vector(1, 1, 1);
        bbox.min = C - R * Vector(1, 1, 1);
    }
    Vector C;
    double R;
};

// Class only used in labs 3 and 4
class TriangleIndices {
  public:
    TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1,
                    int nj = -1, int nk = -1, int uvi = -1, int uvj = -1,
                    int uvk = -1, int group = -1) {
        vtx[0] = vtxi;
        vtx[1] = vtxj;
        vtx[2] = vtxk;
        uv[0] = uvi;
        uv[1] = uvj;
        uv[2] = uvk;
        n[0] = ni;
        n[1] = nj;
        n[2] = nk;
        this->group = group;
    };
    int vtx[3]; // indices within the vertex coordinates array
    int uv[3];  // indices within the uv coordinates array
    int n[3];   // indices within the normals array
    int group;  // face group
};

// Class only used in labs 3 and 4
class TriangleMesh : public Object {
  public:
    TriangleMesh(const Vector &albedo, bool mirror = false,
                 bool transparent = false, bool is_light = false,
                 double n = 1.5, bool invert_normals = false)
        : ::Object(albedo, mirror, transparent, is_light, n, invert_normals){};

    TriangleMesh *clone() const override { return new TriangleMesh(*this); }
    // first scale and then translate the current object
    void scale_translate(double s, const Vector &t) override {
        for (size_t i = 0; i < vertices.size(); i++) {
            vertices[i] = vertices[i] * s + t;
        }
        bbox.min = bbox.min * s + t;
        bbox.max = bbox.max * s + t;
        centroid = centroid * s + t;
        for (auto &node : bvh_nodes) {
            node.bbox.min = node.bbox.min * s + t;
            node.bbox.max = node.bbox.max * s + t;
        }
    }
    void center_scale_translate(double s, const Vector &t) {
        compute_centroid();
        for (auto &vertex : vertices) {
            vertex = (vertex - centroid) * s + t;
        }
        compute_centroid();
        compute_bounding_box();
        build_BVH();
    }

    // read an .obj file
    void readOBJ(const char *obj, bool flip_uvs = false) {
        std::ifstream f(obj);
        if (!f)
            return;

        std::map<std::string, int> mtls;
        int curGroup = -1, maxGroup = -1;

        // OBJ indices are 1-based and can be negative (relative), this
        // normalizes them
        auto resolveIdx = [](int i, int size) {
            return i < 0 ? size + i : i - 1;
        };

        auto setFaceVerts = [&](TriangleIndices &t, int i0, int i1, int i2) {
            t.vtx[0] = resolveIdx(i0, vertices.size());
            t.vtx[1] = resolveIdx(i1, vertices.size());
            t.vtx[2] = resolveIdx(i2, vertices.size());
        };
        auto setFaceUVs = [&](TriangleIndices &t, int j0, int j1, int j2) {
            t.uv[0] = resolveIdx(j0, uvs.size());
            t.uv[1] = resolveIdx(j1, uvs.size());
            t.uv[2] = resolveIdx(j2, uvs.size());
        };
        auto setFaceNormals = [&](TriangleIndices &t, int k0, int k1, int k2) {
            t.n[0] = resolveIdx(k0, normals.size());
            t.n[1] = resolveIdx(k1, normals.size());
            t.n[2] = resolveIdx(k2, normals.size());
        };

        std::string line;
        while (std::getline(f, line)) {
            // Trim trailing whitespace
            line.erase(line.find_last_not_of(" \r\t\n") + 1);
            if (line.empty())
                continue;

            const char *s = line.c_str();

            if (line.rfind("usemtl ", 0) == 0) {
                std::string matname = line.substr(7);
                auto result = mtls.emplace(matname, maxGroup + 1);
                if (result.second) {
                    curGroup = ++maxGroup;
                } else {
                    curGroup = result.first->second;
                }
            } else if (line.rfind("vn ", 0) == 0) {
                Vector v;
                sscanf(s, "vn %lf %lf %lf", &v[0], &v[1], &v[2]);
                normals.push_back(v);
            } else if (line.rfind("vt ", 0) == 0) {
                Vector v;
                sscanf(s, "vt %lf %lf", &v[0], &v[1]);
                if (flip_uvs)
                    v[1] = 1 - v[1];
                uvs.push_back(v);
            } else if (line.rfind("v ", 0) == 0) {
                Vector pos, col;
                if (sscanf(s, "v %lf %lf %lf %lf %lf %lf", &pos[0], &pos[1],
                           &pos[2], &col[0], &col[1], &col[2]) == 6) {
                    for (int i = 0; i < 3; i++)
                        col[i] = std::min(1.0, std::max(0.0, col[i]));
                    vertexcolors.push_back(col);
                } else {
                    sscanf(s, "v %lf %lf %lf", &pos[0], &pos[1], &pos[2]);
                }
                vertices.push_back(pos);
            } else if (line[0] == 'f') {
                int i[4], j[4], k[4], offset, nn;
                const char *cur = s + 1;
                TriangleIndices t;
                t.group = curGroup;

                // Try each face format: v/vt/vn, v/vt, v//vn, v
                if ((nn = sscanf(cur, "%d/%d/%d %d/%d/%d %d/%d/%d%n", &i[0],
                                 &j[0], &k[0], &i[1], &j[1], &k[1], &i[2],
                                 &j[2], &k[2], &offset)) == 9) {
                    setFaceVerts(t, i[0], i[1], i[2]);
                    setFaceUVs(t, j[0], j[1], j[2]);
                    setFaceNormals(t, k[0], k[1], k[2]);
                } else if ((nn = sscanf(cur, "%d/%d %d/%d %d/%d%n", &i[0],
                                        &j[0], &i[1], &j[1], &i[2], &j[2],
                                        &offset)) == 6) {
                    setFaceVerts(t, i[0], i[1], i[2]);
                    setFaceUVs(t, j[0], j[1], j[2]);
                } else if ((nn = sscanf(cur, "%d//%d %d//%d %d//%d%n", &i[0],
                                        &k[0], &i[1], &k[1], &i[2], &k[2],
                                        &offset)) == 6) {
                    setFaceVerts(t, i[0], i[1], i[2]);
                    setFaceNormals(t, k[0], k[1], k[2]);
                } else if ((nn = sscanf(cur, "%d %d %d%n", &i[0], &i[1], &i[2],
                                        &offset)) == 3) {
                    setFaceVerts(t, i[0], i[1], i[2]);
                } else
                    continue;

                indices.push_back(t);
                cur += offset;

                // Fan triangulation for polygon faces (4+ vertices)
                while (*cur && *cur != '\n') {
                    TriangleIndices t2;
                    t2.group = curGroup;
                    if ((nn = sscanf(cur, " %d/%d/%d%n", &i[3], &j[3], &k[3],
                                     &offset)) == 3) {
                        setFaceVerts(t2, i[0], i[2], i[3]);
                        setFaceUVs(t2, j[0], j[2], j[3]);
                        setFaceNormals(t2, k[0], k[2], k[3]);
                    } else if ((nn = sscanf(cur, " %d/%d%n", &i[3], &j[3],
                                            &offset)) == 2) {
                        setFaceVerts(t2, i[0], i[2], i[3]);
                        setFaceUVs(t2, j[0], j[2], j[3]);
                    } else if ((nn = sscanf(cur, " %d//%d%n", &i[3], &k[3],
                                            &offset)) == 2) {
                        setFaceVerts(t2, i[0], i[2], i[3]);
                        setFaceNormals(t2, k[0], k[2], k[3]);
                    } else if ((nn = sscanf(cur, " %d%n", &i[3], &offset)) ==
                               1) {
                        setFaceVerts(t2, i[0], i[2], i[3]);
                    } else {
                        cur++;
                        continue;
                    }

                    indices.push_back(t2);
                    cur += offset;
                    i[2] = i[3];
                    j[2] = j[3];
                    k[2] = k[3];
                }
            }
        }
        center_scale_translate(1.0, Vector(0, 0, 0));
    }

    void update_bbox(BBox &bbox, Vector &vertex) {
        bbox.max[0] = std::max(bbox.max[0], vertex[0]);
        bbox.max[1] = std::max(bbox.max[1], vertex[1]);
        bbox.max[2] = std::max(bbox.max[2], vertex[2]);
        bbox.min[0] = std::min(bbox.min[0], vertex[0]);
        bbox.min[1] = std::min(bbox.min[1], vertex[1]);
        bbox.min[2] = std::min(bbox.min[2], vertex[2]);
    }

    BBox compute_bounding_box(int start_index, int end_index) {
        BBox bbox;
        bbox.min = std::numeric_limits<double>::max() * Vector(1.0, 1.0, 1.0);
        bbox.max = -1 * bbox.min;
        for (int i = start_index; i < end_index; i++) {
            update_bbox(bbox, vertices[indices[i].vtx[0]]);
            update_bbox(bbox, vertices[indices[i].vtx[1]]);
            update_bbox(bbox, vertices[indices[i].vtx[2]]);
        }
        return bbox;
    }
    void compute_bounding_box() override {
        bbox.min = std::numeric_limits<double>::max() * Vector(1.0, 1.0, 1.0);
        bbox.max = -1 * bbox.min;
        for (auto vertex : vertices) {
            update_bbox(bbox, vertex);
        }
    }
    void compute_centroid() override {
        // computes centroid weighted by triangle area
        Vector sum_midpoints(0, 0, 0);
        double total_area = 0.0;

        for (const auto &triangle : indices) {
            Vector A = vertices[triangle.vtx[0]];
            Vector B = vertices[triangle.vtx[1]];
            Vector C = vertices[triangle.vtx[2]];

            Vector e1 = B - A;
            Vector e2 = C - A;
            double area = 0.5 * cross(e1, e2).norm();

            Vector midpoint = (A + B + C) / 3.0;

            sum_midpoints = sum_midpoints + (midpoint * area);
            total_area += area;
        }

        if (total_area > eps) {
            centroid = sum_midpoints / total_area;
        } else {
            centroid = Vector(0, 0, 0);
        }
    }

    // ray-mesh intersection (labs 3 and 4)
    bool intersect(const Ray &ray, Vector &P, double &t, Vector &N,
                   Vector &albedo) const override {

        // lab 3 : for each triangle, compute the ray-triangle intersection with
        // Moller-Trumbore algorithm lab 3 : once done, speed it up by first
        // checking against the mesh bounding box lab 4 : recursively apply the
        // bounding-box test from a BVH datastructure
        double t_prime;
        if (!this->bbox.intersect_ray(ray, t_prime))
            return false;
        std::vector<int> nodes_to_visit;
        nodes_to_visit.push_back(0);
        t = std::numeric_limits<double>::max();
        bool found = 0;
        while (!nodes_to_visit.empty()) {
            BVHNode node = bvh_nodes[nodes_to_visit.back()];
            nodes_to_visit.pop_back();
            if (!node.is_leaf()) {
                if (bvh_nodes[node.left].bbox.intersect_ray(ray, t_prime)) {
                    if (t_prime < t) {
                        nodes_to_visit.push_back(node.left);
                    }
                }
                if (bvh_nodes[node.right].bbox.intersect_ray(ray, t_prime)) {
                    if (t_prime < t) {
                        nodes_to_visit.push_back(node.right);
                    }
                }
            } else {
                for (size_t i = node.start_index; i < node.end_index; i++) {
                    TriangleIndices triangle = indices[i];
                    Vector A = vertices[triangle.vtx[0]];
                    Vector B = vertices[triangle.vtx[1]];
                    Vector C = vertices[triangle.vtx[2]];
                    Vector e1 = B - A;
                    Vector e2 = C - A;
                    Vector N_prime = cross(e1, e2);
                    double ray_dot_N = dot(ray.u, N_prime);
                    if (abs(ray_dot_N) < eps) // ray tangential to object
                        continue;
                    Vector A_cross_ray = cross(A - ray.O, ray.u);
                    double t_prime = dot(A - ray.O, N_prime) / ray_dot_N;
                    double beta = dot(e2, A_cross_ray) / ray_dot_N;
                    double gamma = -dot(e1, A_cross_ray) / ray_dot_N;
                    double alfa = 1 - beta - gamma;
                    if (eps < t_prime && t_prime < t &&
                        -eps < std::min(alfa, std::min(beta, gamma)) &&
                        std::max(alfa, std::max(beta, gamma)) < 1 + eps) {
                        N = N_prime;
                        // smoothen the object
                        /*Vector normal_A = normals[triangle.n[0]];
                        Vector normal_B = normals[triangle.n[1]];
                        Vector normal_C = normals[triangle.n[2]];
                        N = alfa * normal_A + beta * normal_B +
                            gamma * normal_C;*/
                        // can be deleted
                        N.normalize();
                        if (dot(ray.u, N) > 0) {
                            N = -1 * N;
                        }
                        P = alfa * A + beta * B + gamma * C;
                        if (triangle.group != -1 &&
                            triangle.group < (int)textures.size()) {
                            // inshallah albedo calculation
                            Vector uv_A = uvs[triangle.uv[0]];
                            Vector uv_B = uvs[triangle.uv[1]];
                            Vector uv_C = uvs[triangle.uv[2]];
                            Vector uv_P =
                                alfa * uv_A + beta * uv_B + gamma * uv_C;
                            // make them modulo 1 - just the fractional part
                            double u = uv_P[0] - std::floor(uv_P[0]);
                            double v = uv_P[1] - std::floor(uv_P[1]);
                            Texture texture = textures[triangle.group];
                            int x = (int)(u * texture.W);
                            int y = (int)(v * texture.H);
                            x = std::max(0, std::min(texture.W - 1, x));
                            y = std::max(0,
                                         std::min(texture.H - 1, y)); // safety
                            int idx = ((y * texture.W) + x) *
                                      texture.C; // texture has .C channels i.e.
                                                 // .C entries for every pixel
                            if (texture.C >= 4 && texture.data[idx + 3] < 128) {
                                continue; // this point is more transparent than
                                          // opaque
                            }
                            albedo = Vector(
                                std::pow(texture.data[idx] / 255.0, 2.2),
                                std::pow(texture.data[idx + 1] / 255.0, 2.2),
                                std::pow(texture.data[idx + 2] / 255.0,
                                         2.2)); // gamma correction thing, TODO
                                                // remove hardcoding
                        } else {
                            albedo = this->albedo;
                        }
                        found = true;
                        t = t_prime;
                    }
                }
            }
        }
        if (invert_normals)
            N = N * -1.0;
        return found;
    }

    void rotate(const Matrix &m) override {
        for (auto &vertex : vertices) {
            vertex = vertex - centroid;
            vertex = m * vertex;
            vertex = vertex + centroid;
        }
        for (auto &n : normals) {
            n = m * n;
            n.normalize();
        }
        compute_bounding_box();
        build_BVH();
    }

    void add_textures(
        const char *filename) { // IMPORTANT: add textures in the correct order
        int w, h, c;
        stbi_set_flip_vertically_on_load(true); // to not need v = 1 - v
        unsigned char *data = stbi_load(filename, &w, &h, &c, 4);
        if (data) {
            textures.push_back({data, w, h, 4});
        }
    }

    Vector get_barycenter(const TriangleIndices &t) {
        return 1.0 / 3.0 *
               (vertices[t.vtx[0]] + vertices[t.vtx[1]] + vertices[t.vtx[2]]);
    }

    int build_BVH_recursion(int start_index, int end_index) {
        // oh boy
        BVHNode cur;

        cur.start_index = start_index;
        cur.end_index = end_index;
        cur.bbox = compute_bounding_box(start_index, end_index);

        int num_triangles = end_index - start_index;

        if (num_triangles <= 2) { // stopping criterion
            cur.left = -1;
            cur.right = -1;
            bvh_nodes.push_back(cur);
            return bvh_nodes.size() - 1;
        }

        int axis = cur.bbox.longest_axis();

        int mid = start_index + num_triangles / 2;
        std::nth_element(
            indices.begin() + start_index, indices.begin() + mid,
            indices.begin() + end_index,
            [axis, this](const TriangleIndices &a, const TriangleIndices &b) {
                return get_barycenter(a)[axis] < get_barycenter(b)[axis];
            });

        int pos = bvh_nodes.size();
        bvh_nodes.push_back(cur);

        int left = build_BVH_recursion(start_index, mid);
        int right = build_BVH_recursion(mid, end_index);

        bvh_nodes[pos].left = left;
        bvh_nodes[pos].right = right;

        return pos;
    }

    void build_BVH() {
        bvh_nodes.clear();
        build_BVH_recursion(0, indices.size());
    }

    std::vector<TriangleIndices> indices;
    std::vector<Vector> vertices;
    std::vector<Vector> normals;
    std::vector<Vector> uvs;
    std::vector<Vector> vertexcolors;
    // texture things
    struct Texture {
        unsigned char *data;
        int W, H, C; // texture resolution and nb of channels
    };
    std::vector<Texture> textures;

    struct BVHNode {
        int left, right; // indices in bvh_nodes vector
        BBox bbox;
        size_t start_index, end_index;
        bool is_leaf() const { return left == -1; }
    };
    std::vector<BVHNode> bvh_nodes;
};

class Scene {
  public:
    Scene(){};
    void addObject(const Object *obj) { objects.push_back(obj); }
    void removeObject(const Object *obj) { // TODO make cleaner
        for (size_t i = 0; i < objects.size(); i++) {
            if (objects[i] == obj) {
                objects.erase(objects.begin() + i);
                return;
            }
        }
    }

    // returns true iif there is an intersection between the ray and any object
    // in the scene if there is an intersection, also computes the point of the
    // *nearest* intersection P, t>=0 the distance between the ray origin and P
    // (i.e., the parameter along the ray) and the unit normal N. Also returns
    // the index of the object within the std::vector objects in object_id
    bool intersect(const Ray &ray, Vector &P, double &t, Vector &N,
                   Vector &albedo, int &object_id) const {
        t = std::numeric_limits<double>::max();
        object_id = -1;
        Vector P_func, N_func;
        double t_func;
        Vector albedo_func;
        for (size_t i = 0; i < objects.size(); i++) {
            if (objects[i]->intersect(ray, P_func, t_func, N_func,
                                      albedo_func)) {
                if (t_func < t) {
                    t = t_func;
                    P = P_func;
                    N = N_func;
                    albedo = albedo_func;
                    object_id = i;
                }
            }
        }
        return (object_id != -1);
    }

    // return the radiance (color) along ray
    Vector getColor(const Ray &ray, int recursion_depth,
                    bool last_bounce_was_diffuse = false) {

        if (recursion_depth >= max_light_bounce)
            return Vector(0, 0, 0);

        Vector P, N;
        double t;
        int object_id;
        Vector direct_light;
        Vector albedo;
        if (intersect(ray, P, t, N, albedo, object_id)) {
            if (objects[object_id]->is_light) {
                if (last_bounce_was_diffuse) {
                    return Vector(0.0, 0.0,
                                  0.0); // we don't want to count light twice
                }
                if (light_radius < eps) // basically 0
                    return Vector(1.0, 1.0, 1.0);
                double area = 4.0 * M_PI * light_radius * light_radius;
                return light_intensity / area * albedo; // TODO smarter formula
            }
            if (objects[object_id]->mirror) {
                // return getColor in the reflected direction, with
                // recursion_depth+1 (recursively)
                Vector new_vec = ray.u - 2 * dot(ray.u, N) * N;
                new_vec.normalize();
                return getColor(Ray(P + eps * N, new_vec, ray.time),
                                recursion_depth + 1) *
                       albedo;
            }

            if (objects[object_id]->transparent) {
                // return getColor in the refraction direction, with
                // recursion_depth+1 (recursively)
                double n1 = 1.0; // air
                double n2 =
                    objects[object_id]->n; // the object's refraction index

                Vector N_ref = N;
                double cosi = dot(ray.u, N);

                if (cosi > 0) { // acute angle, ray is coming out of the glass
                    N_ref = N * -1.0;  // normal now points inside
                    std::swap(n1, n2); // from glass to air
                } else {
                    // ray is coming in the glass
                    cosi = -cosi; // real cos of incidence angle
                }

                double n = n1 / n2;
                double sin2_r = n * n * (1.0 - cosi * cosi);

                if (sin2_r > 1.0) { // total reflection
                    Vector ref =
                        ray.u +
                        2 * cosi * N_ref; // the - from the reflection formula
                                          // above is embedded in cosi
                    ref.normalize();

                    // offset on the same side
                    return getColor(Ray(P + eps * N_ref, ref, ray.time),
                                    recursion_depth + 1) *
                           albedo;
                } else {
                    // ADDITION: Fresnel law
                    double k0 = sqr((n1 - n2) / (n1 + n2));
                    double R = k0 + (1.0 - k0) * std::pow(1.0 - cosi, 5.0);
                    int tid = omp_get_thread_num();
                    if (uniform(engine[tid]) < R) { // reflect
                        // same code as above
                        Vector ref = ray.u + 2 * cosi * N_ref;
                        ref.normalize();

                        return getColor(Ray(P + eps * N_ref, ref, ray.time),
                                        recursion_depth + 1);
                    } else { // refract
                        double cos_r = sqrt(1.0 - sin2_r);
                        Vector ref = n * ray.u + (n * cosi - cos_r) * N_ref;
                        ref.normalize();

                        // offset on the opposite side
                        return getColor(Ray(P - eps * N_ref, ref, ray.time),
                                        recursion_depth + 1) *
                               albedo;
                    }
                }
            }

            // ADDITION: if we treat the light source as spherical rather than a
            // point, we need to shoot a ray at a random point on the sphere
            int tid = omp_get_thread_num();
            double r1 = uniform(engine[tid]);
            double r2 = uniform(engine[tid]);

            double z = 1.0 - 2.0 * r1;
            double r_xy = sqrt(std::max(0.0, 1.0 - z * z));
            double x = r_xy * cos(2 * M_PI * r2);
            double y = r_xy * sin(2 * M_PI * r2);
            Vector random_light_point;
            random_light_point =
                light_position + Vector(x, y, z) * light_radius;
            // test if there is a shadow by sending a new ray
            P = P + eps * N;
            Ray new_ray(
                P, (random_light_point - P) / (random_light_point - P).norm(),
                ray.time);
            Vector P1, N1;
            int object_id1;
            double t1;
            Vector albedo1;
            bool shadow = 0;
            if (intersect(new_ray, P1, t1, N1, albedo1, object_id1)) {
                if (!objects[object_id1]->is_light &&
                    (P1 - P).norm2() <
                        (random_light_point - P)
                            .norm2()) { // otherwise the light source would cast
                                        // a shadow on itself. shadow!!!
                    direct_light = Vector(0, 0, 0);
                    shadow = true;
                }
            }
            // if there is no shadow, compute the formula with dot products etc.
            if (!shadow)
                direct_light =
                    light_intensity /
                    (4 * M_PI * (random_light_point - P).norm2()) *
                    (albedo / M_PI) *
                    std::max(0.0, dot(N, (random_light_point - P) /
                                             (random_light_point - P).norm()));
            // (lab 2) : add indirect lighting component with a recursive
            // call
            r1 = uniform(engine[tid]);
            r2 = uniform(engine[tid]);
            x = cos(2 * M_PI * r1) * sqrt(1 - r2);
            y = sin(2 * M_PI * r1) * sqrt(1 - r2);
            z = sqrt(r2);
            Vector T1;
            if (abs(N[0]) < abs(N[1]) + eps && abs(N[0]) < abs(N[2]) + eps)
                T1 = Vector(0, -N[2], N[1]);
            else if (abs(N[1]) < abs(N[0]) + eps && abs(N[1]) < abs(N[2]) + eps)
                T1 = Vector(N[2], 0, -N[0]);
            else if (abs(N[2]) < abs(N[0]) + eps && abs(N[2]) < abs(N[1]) + eps)
                T1 = Vector(-N[1], N[0], 0);
            else
                exit(1);
            T1.normalize();
            Vector T2 = cross(N, T1);
            Vector dir = x * T1 + y * T2 + z * N;
            dir.normalize();
            Vector next_color =
                getColor(Ray(P, dir, ray.time), recursion_depth + 1, true);
            Vector indirect_light = next_color * albedo;
            return direct_light + indirect_light;
        }
        return Vector(0, 0, 0);
    }

    void generate_image(int W, int H, std::vector<unsigned char> &image,
                        int N = 200, double sigma = 0.5) {

#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                double miu_x = j - W / 2.0 + 0.5;
                double miu_y = H / 2.0 - i - 0.5;
                double miu_z = -W / (2 * tan(fov / 2));

                // (lab 2) : add Monte Carlo / averaging of random ray
                // contributions here
                // (lab 2) : add antialiasing by altering the ray_direction
                // here
                // (lab 2) : add depth of field effect by altering the ray
                // origin (and direction) here
                Vector color(0, 0, 0);
                int tid = omp_get_thread_num();
                for (int l = 0; l < N; l++) {
                    double r1 = uniform(engine[tid]);
                    if (r1 < eps)
                        r1 = eps;
                    double r2 = uniform(engine[tid]);

                    Vector ray_direction = Vector(
                        miu_x + sigma * sqrt(-2 * log(r1)) * cos(2 * M_PI * r2),
                        miu_y + sigma * sqrt(-2 * log(r1)) * sin(2 * M_PI * r2),
                        miu_z);
                    ray_direction.normalize();

                    // depth of field
                    double t_focal =
                        focal_distance / std::abs(ray_direction[2]);
                    Vector focal_point =
                        camera_center +
                        ray_direction * t_focal; // the point on the focal plane
                                                 // that this ray hits

                    double r_lens = uniform(engine[tid]);
                    double theta_lens = uniform(engine[tid]) * 2 * M_PI;
                    double dx = lens_radius * sqrt(r_lens) * cos(theta_lens);
                    double dy = lens_radius * sqrt(r_lens) * sin(theta_lens);

                    // we shoot a ray from a random point on the lens
                    Vector new_origin = camera_center + Vector(dx, dy, 0);
                    Vector final_dir = focal_point - new_origin;
                    final_dir.normalize();

                    color = color + getColor(Ray(new_origin, final_dir,
                                                 uniform(engine[tid]) *
                                                     camera_shutter_time),
                                             0);
                }
                color = color / N;

                image[(i * W + j) * 3 + 0] = std::min(
                    255.,
                    std::max(0., 255. * std::pow(color[0] / 255., 1. / gamma)));
                image[(i * W + j) * 3 + 1] = std::min(
                    255.,
                    std::max(0., 255. * std::pow(color[1] / 255., 1. / gamma)));
                image[(i * W + j) * 3 + 2] = std::min(
                    255.,
                    std::max(0., 255. * std::pow(color[2] / 255., 1. / gamma)));
            }
        }
    }

    void _rotate_generate_gif(int W, int H, Object &obj, int rot, Vector axis,
                              int frame_nb, int &total_frames,
                              const char *folder) {
        if (frame_nb % rot) {
            frame_nb += rot - frame_nb % rot;
        }
        for (int frame = 0; frame < frame_nb / rot; frame++) {
            Matrix m = create_rotation_matrix(
                axis,
                rot * 2 * M_PI * frame /
                    frame_nb); // rot full rotations around axis

            // I'm doing it every time because I don't want the errors when I
            // rotate multiple times in a row, I just do one big rotation, the
            // time complexity doesn't come from here anyway
            std::unique_ptr<Object> obj_cur(obj.clone());
            obj_cur->rotate(m);

            addObject(obj_cur.get());

            std::vector<unsigned char> image(W * H * 3, 0);
            generate_image(W, H, image, 64);

            for (int j = 0; j < rot; j++) {
                char filename[256];
                snprintf(filename, sizeof(filename), "%s/frame_%03d.png",
                         folder, total_frames + j * (frame_nb / rot) + frame);

                stbi_write_png(filename, W, H, 3, &image[0], 0);
            }

            removeObject(obj_cur.get());
        }
        total_frames += frame_nb;
    }

    void _move_generate_gif(int W, int H, Object &obj, Vector translation,
                            int frame_nb, int &total_frames,
                            const char *folder) {
        // move it top right
        for (int frame = 0; frame < frame_nb; frame++) {

            std::unique_ptr<Object> obj_cur(obj.clone());
            obj_cur->scale_translate(1.0, translation * frame / frame_nb);

            addObject(obj_cur.get());

            char filename[256];
            snprintf(filename, sizeof(filename), "%s/frame_%03d.png", folder,
                     total_frames + frame);

            std::vector<unsigned char> image(W * H * 3, 0);
            generate_image(W, H, image, 64);
            stbi_write_png(filename, W, H, 3, &image[0], 0);

            removeObject(obj_cur.get());
        }
        total_frames += frame_nb;
    }
    void generate_gif(int W, int H, Object &obj, const char *folder) {
        std::filesystem::create_directories(folder);
        int total_frames = 0;
        constexpr int frame_nb = 48;
        _rotate_generate_gif(W, H, obj, 6, Vector(0, 1, 0), frame_nb,
                             total_frames, folder);
        Vector translation{7, 7, -10};
        _move_generate_gif(W, H, obj, translation, frame_nb, total_frames,
                           folder);
        obj.scale_translate(1.0, translation);
        _rotate_generate_gif(W, H, obj, 6, Vector(0, 1, 0), frame_nb,
                             total_frames, folder);
    }

    std::vector<const Object *> objects;

    Vector camera_center, light_position;
    double light_radius = 0; // 0 if point source
    double focal_distance, lens_radius, camera_shutter_time;
    double fov, gamma, light_intensity;
    int max_light_bounce;
};

int main() {
    int W = 512;
    int H = 512;

    for (int i = 0; i < 256; i++) {
        engine[i].seed(i);
    }

    // these recreate the pic with all spheres aligned
    // Sphere left_sphere(Vector(-20, 0, 0), 10., Vector(1.0, 1.0, 1.0), true,
    //                    false);
    // Sphere center_sphere(Vector(0, 0, 0), 10., Vector(1.0, 1.0, 1.0), false,
    //                      true);
    // Sphere right_sphere_outer(Vector(20, 0, 0), 10.0, Vector(1.0, 1.0, 1.0),
    //                           false, true);
    // Sphere right_sphere_inner(Vector(20, 0, 0), 9.5, Vector(1.0, 1.0, 1.0),
    //                           false, true, false, 1.5, true);

    Sphere left_sphere(Vector(-15, 0, 20), 6, Vector(1.0, 0.76, 0.33), true,
                       false);
    left_sphere.velocity = Vector(0, 400, 0);
    Sphere center_sphere(Vector(0, 0, 0), 10., Vector(1.0, 0.0, 0.0), false,
                         true);
    Sphere right_sphere_outer(Vector(20, 20, -10), 15.0, Vector(1.0, 1.0, 1.0),
                              false, true);
    Sphere right_sphere_inner(Vector(20, 20, -10), 14.5, Vector(1.0, 1.0, 1.0),
                              false, true, false, 1.5, true);

    Sphere wall_left(Vector(-1000, 0, 0), 940, Vector(0.8, 0.2, 0.8));
    Sphere wall_right(Vector(1000, 0, 0), 940, Vector(0.8, 0.8, 0.2));
    Sphere wall_front(Vector(0, 0, -1000), 940, Vector(0.2, 0.8, 0.8));
    Sphere wall_behind(Vector(0, 0, 1000), 940, Vector(0.9, 0.6, 0.5));
    Sphere ceiling(Vector(0, 1000, 0), 940, Vector(0.3, 0.5, 0.8));
    Sphere floor(Vector(0, -1000, 0), 990, Vector(0.2, 0.3, 0.8));

    TriangleMesh cat(Vector(1.0, 1.0, 1.0));
    cat.readOBJ("cat/Models_F0202A090/cat.obj");
    cat.scale_translate(0.6, Vector(0, 0, 0));
    // cat.add_textures("cat/Models_F0202A090/cat_diff.png"); // textured car!

    // cat.readOBJ("maxwell_cat/dingus.obj", true);
    // cat.scale_translate(0.01, Vector(0, 0, 0));
    // cat.add_textures("maxwell_cat/dingus_nowhiskers.jpg");
    // cat.add_textures("maxwell_cat/dingus_whiskers.tga.png"); // textured car!
    // Matrix m = create_rotation_matrix(Vector(0, 1, 0), -M_PI / 4);
    // cat.rotate(m);

    Scene scene;
    scene.camera_center = Vector(0, 0, 55);
    scene.light_position = Vector(-10, 20, 40);
    // scene.light_radius = 5.0;
    scene.light_intensity = 1E7;
    scene.focal_distance = 55.0;
    scene.lens_radius = 0;
    scene.camera_shutter_time =
        1.00 / 48; // from the cinematographic shutter formula on the shutter
                   // speed Wikipedia

    scene.fov = 60 * M_PI / 180.;
    scene.gamma = 2.2;
    scene.max_light_bounce = 5;

    // Sphere light_sphere(scene.light_position, scene.light_radius,
    //                     Vector(1.0, 1.0, 1.0), false, false, true);

    // scene.addObject(&light_sphere);

    // scene.addObject(&left_sphere);
    // scene.addObject(&center_sphere);
    // scene.addObject(&right_sphere_outer);
    // scene.addObject(&right_sphere_inner); // ADDITION: the trick

    scene.addObject(&cat);

    scene.addObject(&wall_left);
    scene.addObject(&wall_right);
    scene.addObject(&wall_front);
    scene.addObject(&wall_behind);
    scene.addObject(&ceiling);
    // scene.addObject(&floor);

    std::vector<unsigned char> image(W * H * 3, 0);
    scene.generate_image(W, H, image, 64);
    stbi_write_png("whole_project.png", W, H, 3, &image[0], 0);

    // scene.generate_gif(W, H, cat, "cat_gif_new");

    return 0;
}
