from ast import Return
from PyRT_Common import *
from PyRT_Core import *
from random import randint


# -------------------------------------------------Integrator Classes
# the integrators also act like a scene class in that-
# it stores all the primitives that are to be ray traced.
class Integrator(ABC):
    # Initializer - creates object list
    def __init__(self, filename_, experiment_name=''):
        # self.primitives = []
        self.filename = filename_ + experiment_name
        # self.env_map = None  # not initialized
        self.scene = None

    @abstractmethod
    def compute_color(self, ray):
        pass

    # def add_environment_map(self, env_map_path):
    #    self.env_map = EnvironmentMap(env_map_path)
    def add_scene(self, scene):
        self.scene = scene

    def get_filename(self):
        return self.filename

    # Simple render loop: launches 1 ray per pixel
    def render(self):
        # YOU MUST CHANGE THIS METHOD IN ASSIGNMENTS 1.1 and 1.2:
        cam = self.scene.camera  # camera object
        # ray = Ray()
        print('Rendering Image: ' + self.get_filename())
        for x in range(0, cam.width):
            for y in range(0, cam.height):
                # pixel=GREEN
                # pixel = RGBColor(x/cam.width,y/cam.height,0)
                ray = Ray(Vector3D(0,0,0),cam.get_direction(x,y))
                pixel = self.compute_color(ray)
                # self.scene.any_hit(ray)
                self.scene.set_pixel(pixel, x, y)  # save pixel to pixel array
            progress = (x / cam.width) * 100
            print('\r\tProgress: ' + str(progress) + '%', end='')
        # save image to file
        print('\r\tProgress: 100% \n\t', end='')
        full_filename = self.get_filename()
        self.scene.save_image(full_filename)


class LazyIntegrator(Integrator):
    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        return BLACK


class IntersectionIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        # ASSIGNMENT 1.2: PUT YOUR CODE HERE
        if self.scene.any_hit(ray):
            return RED
        else:
            return BLACK


class DepthIntegrator(Integrator):

    def __init__(self, filename_, max_depth_=10):
        super().__init__(filename_ + '_Depth')
        self.max_depth = max_depth_

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit = self.scene.closest_hit(ray)
        if hit.has_hit:
            radiometry = max(0,1-hit.hit_distance/self.max_depth)
            return RGBColor(radiometry,radiometry,radiometry)
        else:
            return BLACK


class NormalIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Normal')

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit = self.scene.closest_hit(ray)
        if hit.has_hit:
            return RGBColor((1+hit.normal.x)/2,(1+hit.normal.y)/2,(1+hit.normal.z)/2)
        else:
            return BLACK


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')

    def compute_color(self, ray):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        hit=self.scene.closest_hit(ray)

        if hit.has_hit:
            wo = ray.d*-1
            lamb = self.scene.object_list[hit.primitive_index].get_BRDF()
            # Amibient
            La= lamb.kd.multiply(self.scene.i_a)

            # Diffuse
            # Ld = Kd * I / d^2 * max(0, nL)
            Ld = RGBColor(0,0,0)
            # Specular
            # Ls = Ks * I / d^2 * max(0,vr)^s
            # Ls = RGBColor(0,0,0)
            for light in self.scene.pointLights:
                wi = Ray(hit.hit_point,light.pos-hit.hit_point)
                dist = Length(light.pos-hit.hit_point)
                hit_closest = self.scene.closest_hit(wi)
                # shadow test
                if not self.scene.any_hit(wi) or hit_closest.hit_distance >= dist:
                    Lwi = light.intensity / dist / dist
                    Ld += Lwi.multiply(lamb.get_value(light.pos-hit.hit_point,wo,hit.normal))
                    # Ls += Lwi.multiply(lamb.get_value_s(light.pos-hit.hit_point,wo,hit.normal))
            return La+Ld
            
        else:
            return BLACK


class CMCIntegrator(Integrator):  # Classic Monte Carlo Integrator

    def __init__(self, n, filename_, experiment_name=''):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n

    def compute_color(self, ray):
        pass


class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n, myGP, filename_, experiment_name=''):
        filename_bmc = filename_ + '_BMC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.myGP = myGP

    def compute_color(self, ray):
        pass
