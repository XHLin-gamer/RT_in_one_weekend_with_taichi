from random import random
import taichi as ti


use_f64 = False


float_type = ti.f32
if use_f64:
    float_type = ti.f64
vec3 = ti.types.vector(3, float_type)


@ti.struct_class
# @ti.data_oriented
class ray:
    origin: vec3
    direction: vec3

    @ti.func
    def __init__(self, origin: float_type, direction: float_type):
        self.origin = origin
        self.direction = direction.normalized()

    @ti.func
    def normalized(self):
        self.direction = self.direction.normalized()

    @ti.func
    def at(self, t: float_type) -> ti.Vector:
        return self.origin + t * self.direction


@ti.struct_class
class sphere:
    center: vec3
    radius: float_type
    material: int
    # lambertian 0
    # metal 1
    # dielectric 2
    color: vec3
    fuzz: float_type
    etai_over_etat: float_type

    @ti.func
    def hit(self, ray, t_min, t_max):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        half_b = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius**2

        discriminant = half_b**2 - a*c
        is_hit = False
        root: float_type = 0.0
        if discriminant > 0:
            is_hit = True
            sqrtd = ti.sqrt(discriminant)

            root = (-half_b - sqrtd) / a
            if root < t_min or t_max < root:
                root = (-half_b + sqrtd) / a
                if root < t_min or t_max < root:
                    is_hit = False
        position = ray.at(root)
        normal = (position - self.center)/self.radius

        return is_hit, root, position, normal, self.color, self.material, self.fuzz, self.etai_over_etat


@ti.func
def random_init_sphere():
    a = ti.random() * 3.1415926
    b = ti.random() * 3.1415926
    return ti.Vector([ti.cos(a)*ti.cos(b), ti.sin(a)*ti.cos(b), ti.sin(b)])


@ti.func
def random_in_hemi_sphere(direction):
    r1 = random_init_sphere()
    sign = r1.dot(direction)
    if ti.abs(sign) < 1e-9:
        sign = 1.0
        r1 = direction
    return r1*(sign/ti.abs(sign))


@ti.data_oriented
class hittable_list:
    def __init__(self, max_sphere_nums):
        self.sphere_field = sphere.field(shape=(max_sphere_nums,))
        self.sphere_loaded = 0

    def add_sphere(self, center, radius, material, color=ti.Vector([.2, .3, .4]), fuzz=0, etai_over_etat=0):
        self.sphere_field[self.sphere_loaded].center = center
        self.sphere_field[self.sphere_loaded].radius = radius
        self.sphere_field[self.sphere_loaded].material = material
        self.sphere_field[self.sphere_loaded].color = color
        self.sphere_field[self.sphere_loaded].fuzz = fuzz
        self.sphere_field[self.sphere_loaded].etai_over_etat = etai_over_etat
        self.sphere_loaded += 1

    @ti.func
    def hit(self, ray, t_min, t_max):
        root: float_type = 2.0**20
        position = ti.Vector([0.0, 0.0, 0.0], dt=float_type)
        normal = ti.Vector([0.0, 0.0, 0.0], dt=float_type)
        color = ti.Vector([0.0, 0.0, 0.0], dt=float_type)
        material = -1
        is_hit = False
        front_face = True
        fuzz: float_type = 0.0
        etai_over_etat: float_type = 0.0
        for i in range(self.sphere_loaded):
            _is_hit, _root, _position, _normal, _color, _material, _fuzz, _etai_over_etat = self.sphere_field[i].hit(
                ray, t_min, t_max)
            if _is_hit and root > _root:
                is_hit = _is_hit
                root = _root
                position = _position
                normal = _normal
                color = _color
                material = _material
                fuzz = _fuzz
                etai_over_etat = _etai_over_etat
                if ray.direction.dot(normal) > 0.0:
                    front_face = False
                    normal = -normal
                else:
                    front_face = True

        return is_hit, root, position, normal, front_face, color, material, fuzz, etai_over_etat


@ti.func
def reflect(v, n):
    return v - 2*v.dot(n)*n


@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = ti.min(-n.dot(uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta*n)
    r_out_parallel = -ti.sqrt(ti.abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel


@ti.func
def reflectance(cosine, ref_idx):
    r0 = (1-ref_idx) / (1+ref_idx)
    r0 = r0**2
    return r0 + (1 - r0)*((1-cosine)**5)


@ti.func
def random_in_unit_disk():
    ttt = ti.random() * 3.1415926 * 2
    return ti.Vector([ti.cos(ttt), ti.sin(ttt), 0.0])


@ti.data_oriented
class camera:
    def __init__(self, look_from, look_at, vup, vfov, aspect_ratio, aperture=1.0, focus_dist=1.0):
        theta = vfov/180.0*3.14159265
        h = ti.tan(theta/2)
        viewport_height = 2.0*h
        viewport_width = aspect_ratio * viewport_height

        self.w = (look_from - look_at).normalized()
        self.u = vup.cross(self.w).normalized()
        self.v = self.w.cross(self.u)

        self.origin = look_from
        self.horizental = viewport_width * self.u * focus_dist
        self.vertical = viewport_height * self.v * focus_dist
        self.lower_left_corner = self.origin - \
            self.horizental/2 - self.vertical/2 - self.w*focus_dist
        self.lens_radius = aperture / 2.0

    @ti.func
    def get_ray(self, s, t):
        rd = self.lens_radius * random_in_unit_disk()
        offset = self.u*rd[0] + self.v*rd[1]
        new_ray = ray(origin=self.origin + offset,
                      direction=self.lower_left_corner + s*self.horizental + t*self.vertical - self.origin - offset)
        new_ray.normalized()
        return new_ray
