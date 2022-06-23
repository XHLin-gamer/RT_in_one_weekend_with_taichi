# RT_in_one_weekend_with_taichi
![pic1](./1655910860.png)

用taichi搓了一个[Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)的简易光线追踪程序。用taichi进行光线追踪的变成雀食轻松不少，替换了原文中使用C++智能指针的部分，编写程序的时候不用想着这个智能指针咋用来着之类的问题。而且比起用openMP多线程，GPU的毕竟并行程度高，跑得快。

### 遇到的几个问题：
- 数值精度问题
  
taichi默认使用32为浮点数。上面的那副展示图中，用来表示地面的那个球体半径太大了，撞到了数值精度的问题，在32位浮点数表示的情况下，会产生神秘的纹理。如下所示![pic2](./float_acc.png)

两个解决方法：1，把表示地面的那个球半径调小一些，实测从1000缩小到600左右就没有问题了；2.使用64位浮点数
```python
# ray_module.py , line 5:
use_f64 = False # switch True to use Double-precision floating-point 
float_type = ti.f32
if use_f64:
    float_type = ti.f64
vec3 = ti.types.vector(3, float_type)
```
但是64位浮点数算的炒鸡慢

- ray类改写成ti.struct_class地面变黄

![yellow.png](./yellow.png)

原因是taichi的```struct_class```不会自动调用python的class的```__init__```函数,导致ray的direction没有被normalized.手动normalized可以解决

- GPU运行效率并不高

程序运行的时候GPU使用率非常低，不如说就是0%

想不明白为啥，可能对内存访问效率太差了？或者使用了python-scope的构造函数？或者每条光线求交并没有充分使用GPU的并行性？
## reference

- [bsavery/ray-tracing-one-weekend-taichi](https://github.com/bsavery/ray-tracing-one-weekend-taichi)
- [erizmr/taichi_ray_tracing](https://github.com/erizmr/taichi_ray_tracing)
- [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
