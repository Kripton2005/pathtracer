#define _CRT_SECURE_NO_WARNINGS 1
#include <cmath>
#include <random>
#include <vector>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#ifndef M_PI
#define M_PI 3.14159265358979323856
#endif

static std::default_random_engine engine[32];
static std::uniform_real_distribution<double> uniform(0, 1);

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
  Ray(const Vector &origin, const Vector &unit_direction)
      : O(origin), u(unit_direction) {};
  Vector O, u;
  double n;
};

class Object {
public:
  Object(const Vector &albedo, bool mirror = false, bool transparent = false)
      : albedo(albedo), mirror(mirror), transparent(transparent) {};

  virtual bool intersect(const Ray &ray, Vector &P, double &t,
                         Vector &N) const = 0;

  Vector albedo;
  bool mirror, transparent;
  double n;
};

class Sphere : public Object {
public:
  Sphere(const Vector &center, double radius, const Vector &albedo,
         bool mirror = false, bool transparent = false)
      : ::Object(albedo, mirror, transparent), C(center), R(radius) {};

  // returns true iif there is an intersection between the ray and the sphere
  // if there is an intersection, also computes the point of intersection P,
  // t>=0 the distance between the ray origin and P (i.e., the parameter along
  // the ray) and the unit normal N
  bool intersect(const Ray &ray, Vector &P, double &t, Vector &N) const {
    double delta = sqr(dot(ray.u, ray.O - C)) - ((ray.O - C).norm2() - sqr(R));
    if (delta < -eps)
      return false;
    if (-eps < delta && delta < eps)
      delta = 0;
    double t1 = dot(ray.u, C - ray.O) + sqrt(delta);
    double t2 = dot(ray.u, C - ray.O) - sqrt(delta);
    if (t1 < eps)
      return false;
    if (t2 > eps)
      t = t2;
    else
      t = t1;
    P = ray.O + t * ray.u;
    N = P - C;
    N.normalize();
    return true;
  }

  double R;
  Vector C;
};

// I will provide you with an obj mesh loader (labs 3 and 4)
class TriangleMesh : public Object {
public:
  TriangleMesh(const Vector &albedo, bool mirror = false,
               bool transparent = false)
      : ::Object(albedo, mirror, transparent) {};

  bool intersect(const Ray &ray, Vector &P, double &t, Vector &N) const {
    // TODO (labs 3 and 4)
    return false;
  }
};

class Scene {
public:
  Scene() {};
  void addObject(const Object *obj) { objects.push_back(obj); }

  // returns true iif there is an intersection between the ray and any object in
  // the scene
  // if there is an intersection, also computes the point of the *nearest*
  // intersection P, t>=0 the distance between the ray origin and P (i.e., the
  // parameter along the ray) and the unit normal N. Also returns the index of
  // the object within the std::vector objects in object_id
  bool intersect(const Ray &ray, Vector &P, double &t, Vector &N,
                 int &object_id) const {
    t = 2e9;
    object_id = -1;
    Vector P_func, N_func;
    double t_func;
    for (int i = 0; i < objects.size(); i++) {
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
  Vector getColor(const Ray &ray, int recursion_depth) {

    if (recursion_depth >= max_light_bounce)
      return Vector(0, 0, 0);

    Vector P, N;
    double t;
    int object_id;
    if (intersect(ray, P, t, N, object_id)) {
      if (objects[object_id]->mirror) {
        // return getColor in the reflected direction, with recursion_depth+1
        // (recursively)
        Vector new_vec = ray.u - 2 * dot(ray.u, N) * N;
        new_vec.normalize();
        return getColor(Ray(P + eps * N, new_vec), recursion_depth + 1);
      } // else

      if (objects[object_id]->transparent) {
        // return getColor in the refraction direction, with recursion_depth+1
        // (recursively)
        // TODO
        /*Vector new_vec_t = 1.0003 / 1.53 * (ray.u - dot(ray.u, N) * N);
        Vector new_vec_n = -1 * N *
                           sqrt(1 - sqr(ray.n - objects[object_id]->n) *
                                        (1 - sqr(dot(ray.u, N))));
        Vector new_vec = new_vec_t + new_vec_n;
        new_vec.normalize();
        return getColor(Ray(P + eps * N, new_vec), recursion_depth + 1);*/
      } // else

      // test if there is a shadow by sending a new ray
      P = P + eps * N;
      Ray new_ray(P, (light_position - P) / (light_position - P).norm());
      Vector P1, N1;
      int object_id1;
      double t1;
      if (intersect(new_ray, P1, t1, N1, object_id1)) {
        if ((P1 - P).norm2() < (light_position - P).norm2()) // shadow!!!
          return Vector(0, 0, 0);
      }
      // if there is no shadow, compute the formula with dot products etc.
      return light_intensity / (4 * M_PI * (light_position - P).norm2()) *
             (objects[object_id]->albedo / M_PI) *
             std::max(0.0, dot(N, (light_position - P) /
                                      (light_position - P).norm()));
      // TODO (lab 2) : add indirect lighting component with a recursive call
    }

    return Vector(0, 0, 0);
  }

  std::vector<const Object *> objects;

  Vector camera_center, light_position;
  double fov, gamma, light_intensity;
  int max_light_bounce;
};

int main() {
  int W = 512;
  int H = 512;

  for (int i = 0; i < 32; i++) {
    engine[i].seed(i);
  }

  Sphere center_sphere(Vector(0, 0, 0), 10., Vector(0.8, 0.8, 0.8), true);
  Sphere wall_left(Vector(-1000, 0, 0), 940, Vector(0.5, 0.8, 0.1));
  Sphere wall_right(Vector(1000, 0, 0), 940, Vector(0.9, 0.2, 0.3));
  Sphere wall_front(Vector(0, 0, -1000), 940, Vector(0.1, 0.6, 0.7));
  Sphere wall_behind(Vector(0, 0, 1000), 940, Vector(0.8, 0.2, 0.9));
  Sphere ceiling(Vector(0, 1000, 0), 940, Vector(0.3, 0.5, 0.3));
  Sphere floor(Vector(0, -1000, 0), 990, Vector(0.6, 0.5, 0.7));

  Scene scene;
  scene.camera_center = Vector(0, 0, 55);
  scene.light_position = Vector(-10, 20, 40);
  scene.light_intensity = 3E7;
  scene.fov = 60 * M_PI / 180.;
  scene.gamma = 2.2;
  scene.max_light_bounce = 5;

  scene.addObject(&center_sphere);

  scene.addObject(&wall_left);
  scene.addObject(&wall_right);
  scene.addObject(&wall_front);
  scene.addObject(&wall_behind);
  scene.addObject(&ceiling);
  scene.addObject(&floor);

  std::vector<unsigned char> image(W * H * 3, 0);

#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      Vector color;

      Vector ray_direction = Vector(j - W / 2.0 + 0.5, H / 2.0 - i - 0.5,
                                    -W / (2 * tan(scene.fov / 2)));
      ray_direction.normalize();
      Ray ray(scene.camera_center, ray_direction);

      // TODO (lab 2) : add Monte Carlo / averaging of random ray contributions
      // here
      // TODO (lab 2) : add antialiasing by altering the ray_direction here
      // TODO (lab 2) : add depth of field effect by altering the ray origin
      // (and direction) here

      color = scene.getColor(ray, 0);

      image[(i * W + j) * 3 + 0] = std::min(
          255.,
          std::max(0., 255. * std::pow(color[0] / 255., 1. / scene.gamma)));
      image[(i * W + j) * 3 + 1] = std::min(
          255.,
          std::max(0., 255. * std::pow(color[1] / 255., 1. / scene.gamma)));
      image[(i * W + j) * 3 + 2] = std::min(
          255.,
          std::max(0., 255. * std::pow(color[2] / 255., 1. / scene.gamma)));
    }
  }
  stbi_write_png("lab1.png", W, H, 3, &image[0], 0);

  return 0;
}
