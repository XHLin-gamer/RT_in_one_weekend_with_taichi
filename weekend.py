from random import random
from ray_module import *
import taichi as ti
import time

ti.init(arch=ti.gpu)
res = 1920, 1080

# rays pools
# BVH

output_image = False
output_image_target_iteration = 1000

color_buffer = ti.Vector.field(3, dtype=float_type, shape=res)
frame_buffer = ti.Vector.field(3, dtype=float_type, shape=res)
iteration = ti.field(dtype=int, shape=())
iteration[None] = 0
viewport_height = 2.0
aspect_ratio = res[0] / res[1]
viewport_width = aspect_ratio * viewport_height
sampels_per_pixel = ti.field(dtype=ti.int32, shape=())
sampels_per_pixel[None] = 4
depth = 32
# camera
# focus_lenth = 1.0
# origin = ti.Vector([0.0, 0.0, 0.0])
# horizontal = ti.Vector([viewport_width, 0, 0])
# vertical = ti.Vector([0, viewport_height, 0])
# lower_left_corner = origin - 0.5 * horizontal - \
#     0.5*vertical - ti.Vector([0, 0, focus_lenth])

# test_sphere = sphere(center=ti.Vector([0, 0, -1]), radius=0.5)
# another_sphere = sphere(center=ti.Vector([0, -100.5, -1]), radius=100.0)

scene = hittable_list(max_sphere_nums=2**9)


def init_scene():
    # ground
    scene.add_sphere(center=ti.Vector(
        [0, -1000.0, -1]), radius=1000.0, material=0, color=ti.Vector([0.5, 0.5, 0.5]))
    for _a in range(22):
        for _b in range(22):
            rrr = random()
            center = ti.Vector(
                [_a - 11.0 + 0.9*random(), 0.2, _b - 11.0 + 0.9*random()])
            if (center - ti.Vector([4.0, 0.2, 0.0])).dot(center - ti.Vector([4.0, 0.2, 0.0])) > 0.8:
                if rrr < 0.8:
                    color = ti.Vector([random(), random(), random()])
                    scene.add_sphere(center=center, radius=0.2,
                                     material=0, color=color)
                elif rrr < 0.95:
                    color = ti.Vector(
                        [random(), random(), random()])*0.5 + ti.Vector([0.5, 0.5, 0.5])
                    fuzz = random()*0.5
                    scene.add_sphere(center=center, radius=.2,
                                     material=1, fuzz=fuzz)
                else:
                    scene.add_sphere(center=center, radius=0.2,
                                     material=2, etai_over_etat=1.5)

    scene.add_sphere(ti.Vector([0.0, 1.0, 0.0]),
                     radius=1.0, material=2, etai_over_etat=1.5)
    scene.add_sphere(ti.Vector([-4.0, 1., 0.0]), radius=1.0,
                     material=0, color=ti.Vector([0.4, 0.2, 0.1]))
    scene.add_sphere(ti.Vector([4., 1., 0.]), 1.0, material=1, color=ti.Vector(
        [0.7, 0.6, 0.5]), fuzz=0.05)


init_scene()

look_from = vec3([16.0, 3.0, 3.0])
look_at = vec3([0.0, .0, 0.0])
vup = vec3([0.0, 1.0, 0.0])

aperture = .1
# dist_to_focus =(look_from - look_at).dot(look_from - look_at)
dist_to_focus = 12.0
cam = camera(look_from=look_from, look_at=look_at, vup=vup, vfov=20.0,
             aspect_ratio=aspect_ratio, focus_dist=dist_to_focus, aperture=aperture)


@ti.func
def ray_color(r) -> ti.Vector:
    recursive_origin: float_type = r.origin
    recursive_direction: float_type = r.direction
    # r = ray(origin=ti.Vector([0.0, 0.0, 0.0]), direction=ti.Vector([1.0, 1.0, 1.0],dt=ti.f32))
    p_RR = 0.6
    black = ti.Vector([.0, 0.0, 0.0], dt=float_type)
    brightness = black
    if ti.random() < p_RR:
        brightness = ti.Vector([1.0, 1.0, 1.0], dt=float_type)/p_RR
        for d in range(depth):
            # brightness = ti.Vector([0.0, 0.0, 0.0])
            # break
            if d == (depth-1):
                brightness = black
            new_ray = ray(origin=recursive_origin,
                          direction=recursive_direction)
            new_ray.normalized()
            is_hit, root, position, normal, front_face, color, material, fuzz, etai_over_etat = scene.hit(
                new_ray, 0.0001, 1000000)
            # 使用hierarchy 求交
            t = root
            if is_hit:
                if material == 0:
                    # normal = normal.normalized()
                    # fac = ti.exp(-root)/4 + 0.75
                    # brightness *= color * fac + ti.Vector([1.0, 1.0, 1.0], dt=float_type)*(1-fac)
                    brightness *= color
                    # r_p = random_init_sphere()
                    r_p = random_in_hemi_sphere(normal)
                    recursive_direction = normal.normalized() + r_p
                    recursive_origin = position
                elif material == 1:
                    recursive_direction = reflect(
                        recursive_direction, normal) + fuzz * random_init_sphere()
                    recursive_origin = position
                    brightness *= color
                elif material == 2:
                    if d == (depth - 2):
                        brightness = ti.Vector([1.0, 1.0, 1.0])
                        break
                    refraction_ratio = 1.5
                    if front_face:
                        refraction_ratio = 0.666
                    cos_theta = ti.min(-normal.dot(recursive_direction.normalized()), 1.0)
                    sin_theta = ti.sqrt(1.0 - cos_theta**2)
                    cannot_refract = refraction_ratio * sin_theta > 1.0
                    if not cannot_refract or reflectance(cos_theta, refraction_ratio) < ti.random():
                        recursive_direction = refract(
                            recursive_direction.normalized(), normal, refraction_ratio)
                    else:
                        recursive_direction = reflect(
                            recursive_direction.normalized(), normal)
                    recursive_origin = position
            else:
                t = 0.5*(r.direction[1] + 1.0)
                brightness *= (1-t)*ti.Vector([1.0, 1.0, 1.0]
                                              ) + t*ti.Vector([0.5, 0.7, 1.0])
                break
    return brightness


@ti.kernel
def render():
    print(f'iteration: {iteration[None]}')
    iteration[None] += 1
    fa = sampels_per_pixel[None]*iteration[None]
    for i, j in color_buffer:
        for s in range(sampels_per_pixel[None]):
            u = (i+ti.random())/(res[0] - 1)
            v = (j+ti.random())/(res[1] - 1)
            color_buffer[i, j] += ray_color(r=cam.get_ray(u, v))
        frame_buffer[i, j] = color_buffer[i, j] / fa
        frame_buffer[i, j] = ti.sqrt(frame_buffer[i, j])


if output_image:
    for _ in range(output_image_target_iteration):
        render()
    window = ti.GUI(name='weekend', res=res,
                    background_color=0xAAAAAA, show_gui=False)
    window.set_image(frame_buffer)
    window.show(str(int(time.time()))+'.png')
else:
    window = ti.GUI(name='weekend', res=res, background_color=0xAAAAAA)
    while window.running:
        render()
        window.set_image(frame_buffer)
        window.show()
