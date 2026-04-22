#define _CRT_SECURE_NO_WARNINGS 1
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <omp.h>
#include <random>
#include <vector>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#ifndef M_PI
#define M_PI 3.14159265358979323856
#endif

static std::default_random_engine engine[256];
thread_local std::uniform_real_distribution<double> uniform(0, 1);

const double eps = 1e-10;

double sqr(double x) { return x * x; };

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
    double data[3];
};

Vector operator+(const Vector &a, const Vector &b) {
    return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector &a, const Vector &b) {
    return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const double a, const Vector &b) {
    return Vector(a * b[0], a * b[1], a * b[2]);
}
Vector operator*(const Vector &a, const double b) { return b * a; }
Vector operator*(const Vector &a, const Vector &b) {
    return Vector(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}
Vector operator/(const Vector &a, const double b) {
    return Vector(a[0] / b, a[1] / b, a[2] / b);
}
double dot(const Vector &a, const Vector &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector &a, const Vector &b) {
    return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                  a[0] * b[1] - a[1] * b[0]);
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
           bool is_light = false, double n = 1.5) // by default glass
        : albedo(albedo), mirror(mirror), transparent(transparent),
          is_light(is_light), n(n){};

    virtual bool intersect(const Ray &ray, Vector &P, double &t,
                           Vector &N) const = 0;

    Vector albedo, velocity = Vector(0, 0, 0);
    bool mirror, transparent, is_light;
    double n;
};

class Sphere : public Object {
  public:
    Sphere(const Vector &center, double radius, const Vector &albedo,
           bool mirror = false, bool transparent = false, bool is_light = false,
           double n = 1.5, bool invert_normals = false)
        : ::Object(albedo, mirror, transparent, is_light, n), C(center),
          R(radius), invert_normals(invert_normals){};

    // returns true iif there is an intersection between the ray and the sphere
    // if there is an intersection, also computes the point of intersection P,
    // t>=0 the distance between the ray origin and P (i.e., the parameter along
    // the ray) and the unit normal N
    bool intersect(const Ray &ray, Vector &P, double &t, Vector &N) const {
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
        return true;
    }

    Vector C;
    double R;
    bool invert_normals;
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
                 bool transparent = false)
        : ::Object(albedo, mirror, transparent){};

    // first scale and then translate the current object
    void scale_translate(double s, const Vector &t) {
        for (size_t i = 0; i < vertices.size(); i++) {
            vertices[i] = vertices[i] * s + t;
        }
    }

    // read an .obj file
    void readOBJ(const char *obj) {
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
    }

    void compute_bounding_box() {
        B_min = std::numeric_limits<double>::max() * Vector(1.0, 1.0, 1.0);
        B_max = -1 * B_min;
        for (auto vertex : vertices) {
            B_max.data[0] = std::max(B_max.data[0], vertex.data[0]);
            B_max.data[1] = std::max(B_max.data[1], vertex.data[1]);
            B_max.data[2] = std::max(B_max.data[2], vertex.data[2]);
            B_min.data[0] = std::min(B_min.data[0], vertex.data[0]);
            B_min.data[1] = std::min(B_min.data[1], vertex.data[1]);
            B_min.data[2] = std::min(B_min.data[2], vertex.data[2]);
        }
    }

    // TODO ray-mesh intersection (labs 3 and 4)
    bool intersect(const Ray &ray, Vector &P, double &t, Vector &N) const {

        // lab 3 : for each triangle, compute the ray-triangle intersection with
        // Moller-Trumbore algorithm lab 3 : once done, speed it up by first
        // checking against the mesh bounding box lab 4 : recursively apply the
        // bounding-box test from a BVH datastructure

        double tx_min = (B_min.data[0] - ray.O.data[0]) / ray.u.data[0];
        double tx_max = (B_max.data[0] - ray.O.data[0]) / ray.u.data[0];
        if (tx_max < tx_min - eps)
            std::swap(tx_min, tx_max);
        double ty_min = (B_min.data[1] - ray.O.data[1]) / ray.u.data[1];
        double ty_max = (B_max.data[1] - ray.O.data[1]) / ray.u.data[1];
        if (ty_max < ty_min - eps)
            std::swap(ty_min, ty_max);
        double tz_min = (B_min.data[2] - ray.O.data[2]) / ray.u.data[2];
        double tz_max = (B_max.data[2] - ray.O.data[2]) / ray.u.data[2];
        if (tz_max < tz_min - eps)
            std::swap(tz_min, tz_max);
        double t_min = std::max(tx_min, std::max(ty_min, tz_min));
        double t_max = std::min(tx_max, std::min(ty_max, tz_max));
        if (t_min > t_max + eps)
            return false;

        bool found = 0;
        t = std::numeric_limits<double>::max();
        for (auto triangle : indices) {
            Vector A = vertices[triangle.vtx[0]];
            Vector B = vertices[triangle.vtx[1]];
            Vector C = vertices[triangle.vtx[2]];
            Vector e1 = B - A;
            Vector e2 = C - A;
            Vector N_prime = cross(e1, e2);
            double t_prime = dot(A - ray.O, N_prime) / dot(ray.u, N_prime);
            double beta =
                dot(e2, cross(A - ray.O, ray.u)) / dot(ray.u, N_prime);
            double gamma =
                -dot(e1, cross(A - ray.O, ray.u)) / dot(ray.u, N_prime);
            double alfa = 1 - beta - gamma;
            if (-eps < t_prime && t_prime < t &&
                -eps < std::min(alfa, std::min(beta, gamma)) &&
                std::max(alfa, std::max(beta, gamma)) < 1 + eps) {
                found = true;
                t = t_prime;
                N = N_prime;
                N.normalize();
                P = alfa * A + beta * B + gamma * C;
            }
        }

        return found;
    }

    std::vector<TriangleIndices> indices;
    std::vector<Vector> vertices;
    std::vector<Vector> normals;
    std::vector<Vector> uvs;
    std::vector<Vector> vertexcolors;
    Vector B_min, B_max;
};

class Scene {
  public:
    Scene(){};
    void addObject(const Object *obj) { objects.push_back(obj); }

    // returns true iif there is an intersection between the ray and any object
    // in the scene if there is an intersection, also computes the point of the
    // *nearest* intersection P, t>=0 the distance between the ray origin and P
    // (i.e., the parameter along the ray) and the unit normal N. Also returns
    // the index of the object within the std::vector objects in object_id
    bool intersect(const Ray &ray, Vector &P, double &t, Vector &N,
                   int &object_id) const {
        t = std::numeric_limits<double>::max();
        object_id = -1;
        Vector P_func, N_func;
        double t_func;
        for (size_t i = 0; i < objects.size(); i++) {
            if (objects[i]->intersect(ray, P_func, t_func, N_func)) {
                if (t_func < t) {
                    t = t_func;
                    P = P_func;
                    N = N_func;
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
        if (intersect(ray, P, t, N, object_id)) {
            if (objects[object_id]->is_light) {
                if (last_bounce_was_diffuse) {
                    return Vector(0.0, 0.0,
                                  0.0); // we don't want to count light twice
                }
                if (light_radius < eps) // basically 0
                    return Vector(1.0, 1.0, 1.0);
                double area = 4.0 * M_PI * light_radius * light_radius;
                return light_intensity / area *
                       objects[object_id]->albedo; // TODO smarter formula
            }
            if (objects[object_id]->mirror) {
                // return getColor in the reflected direction, with
                // recursion_depth+1 (recursively)
                Vector new_vec = ray.u - 2 * dot(ray.u, N) * N;
                new_vec.normalize();
                return getColor(Ray(P + eps * N, new_vec, ray.time),
                                recursion_depth + 1) *
                       objects[object_id]->albedo;
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
                           objects[object_id]->albedo;
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
                               objects[object_id]->albedo;
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
            if (light_radius > eps)
                random_light_point =
                    light_position + Vector(x, y, z) * light_radius;
            else
                random_light_point = light_position;
            // test if there is a shadow by sending a new ray
            P = P + eps * N;
            Ray new_ray(
                P, (random_light_point - P) / (random_light_point - P).norm(),
                ray.time);
            Vector P1, N1;
            int object_id1;
            double t1;
            bool shadow = 0;
            if (intersect(new_ray, P1, t1, N1, object_id1)) {
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
                    (objects[object_id]->albedo / M_PI) *
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
            if (abs(N.data[0]) < abs(N.data[1]) + eps &&
                abs(N.data[0]) < abs(N.data[2]) + eps)
                T1 = Vector(0, -N.data[2], N.data[1]);
            else if (abs(N.data[1]) < abs(N.data[0]) + eps &&
                     abs(N.data[1]) < abs(N.data[2]) + eps)
                T1 = Vector(N.data[2], 0, -N.data[0]);
            else if (abs(N.data[2]) < abs(N.data[0]) + eps &&
                     abs(N.data[2]) < abs(N.data[1]) + eps)
                T1 = Vector(-N.data[1], N.data[0], 0);
            else
                exit(1);
            T1.normalize();
            Vector T2 = cross(N, T1);
            Vector dir = x * T1 + y * T2 + z * N;
            dir.normalize();
            Vector next_color =
                getColor(Ray(P, dir, ray.time), recursion_depth + 1, true);
            Vector indirect_light = next_color * objects[object_id]->albedo;
            return direct_light + indirect_light;
        }
        return Vector(0, 0, 0);
    }

    std::vector<const Object *> objects;

    Vector camera_center, light_position;
    double light_radius; // 0 if point source
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

    TriangleMesh cat(Vector(1.0, 1.0, 1.0)); // white cat for now

    cat.readOBJ("cat/Models_F0202A090/cat.obj");
    cat.scale_translate(0.6, Vector(0.0, -5.0, 0.0));
    cat.compute_bounding_box();

    Scene scene;
    scene.camera_center = Vector(0, 0, 55);
    scene.light_position = Vector(-10, 20, 40);
    scene.light_radius = 5.0;
    scene.light_intensity = 1E7;
    scene.focal_distance = 55.0;
    scene.lens_radius = 0.3;
    scene.camera_shutter_time =
        1.00 / 48; // from the cinematographic shutter formula on the shutter
                   // speed Wikipedia

    scene.fov = 60 * M_PI / 180.;
    scene.gamma = 2.2;
    scene.max_light_bounce = 1;

    Sphere light_sphere(scene.light_position, scene.light_radius,
                        Vector(1.0, 1.0, 1.0), false, false, true);

    scene.addObject(&light_sphere);

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
    scene.addObject(&floor);

    std::vector<unsigned char> image(W * H * 3, 0);

    int N = 32;
    double sigma = 0.5;

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            double miu_x = j - W / 2.0 + 0.5;
            double miu_y = H / 2.0 - i - 0.5;
            double miu_z = -W / (2 * tan(scene.fov / 2));

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
                    scene.focal_distance / std::abs(ray_direction.data[2]);
                Vector focal_point =
                    scene.camera_center +
                    ray_direction * t_focal; // the point on the focal plane
                                             // that this ray hits

                double r_lens = uniform(engine[tid]);
                double theta_lens = uniform(engine[tid]) * 2 * M_PI;
                double dx = scene.lens_radius * sqrt(r_lens) * cos(theta_lens);
                double dy = scene.lens_radius * sqrt(r_lens) * sin(theta_lens);

                // we shoot a ray from a random point on the lens
                Vector new_origin = scene.camera_center + Vector(dx, dy, 0);
                Vector final_dir = focal_point - new_origin;
                final_dir.normalize();

                color =
                    color + scene.getColor(Ray(new_origin, final_dir,
                                               uniform(engine[tid]) *
                                                   scene.camera_shutter_time),
                                           0);
            }
            color = color / N;

            image[(i * W + j) * 3 + 0] =
                std::min(255., std::max(0., 255. * std::pow(color[0] / 255.,
                                                            1. / scene.gamma)));
            image[(i * W + j) * 3 + 1] =
                std::min(255., std::max(0., 255. * std::pow(color[1] / 255.,
                                                            1. / scene.gamma)));
            image[(i * W + j) * 3 + 2] =
                std::min(255., std::max(0., 255. * std::pow(color[2] / 255.,
                                                            1. / scene.gamma)));
        }
    }
    stbi_write_png("lab3.png", W, H, 3, &image[0], 0);

    return 0;
}
