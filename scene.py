import copy
import math

import numpy as np
import scipy
from manim import *

from sympy import sin, cos, exp, Symbol, lambdify, latex, print_latex
from manim.mobject.geometry.tips import ArrowSquareTip
from scipy.optimize import minimize_scalar
from manim.mobject.geometry.tips import ArrowTriangleFilledTip

l_font_size = 90
font_size_partial_der = 170
three_d_f_font_size = 30
font_size_big = 60
axis_label_font_size = 180


def align_top(obj, template):
    current_coords = obj.get_center()
    x = current_coords[0]
    y = current_coords[1]
    z = current_coords[2]
    shift = template.get_top()[1] - obj.get_top()[1]
    obj.move_to(np.array([x, y + shift, z]))
    return obj


def align_left(obj, template):
    current_coords = obj.get_center()
    x = current_coords[0]
    y = current_coords[1]
    z = current_coords[2]
    shift = template.get_left()[0] - obj.get_left()[0]
    obj.move_to(np.array([x + shift, y, z]))
    return obj


def get_anchor_point(xx, yy, ax):
    return ax.plot(lambda x: np.sin(x - xx) + yy,
                   x_range=[xx - 1, xx + 1, 0.01], color=RED, stroke_width=0)


def get_opt_point(opt_point_size, dummy_graph):
    return Circle(color=YELLOW, fill_opacity=1, radius=opt_point_size).move_to(dummy_graph)


def get_arrow_on_graph(ax, x1, y1, x2, y2, stroke, dummy_center):
    start_ = get_anchor_point(x1, y1, ax)
    end_ = get_anchor_point(x2, y2, ax)
    return Line(start=start_.get_center(), end=end_.get_center(), color=RED, buff=0.05).add_tip(
        tip_shape=ArrowTriangleFilledTip)

def get_arrow_new(point1, point2, ax=None, color=None, buff=None, add_pointer=None):
    if ax is not None:
        point1 = ax.c2p(point1[0], point1[1], point1[2])
        point2 = ax.c2p(point2[0], point2[1], point2[2])
    color_ = YELLOW if color is None else color
    buff_ = 0.05 if buff is None else buff
    line = Line(start=point1, end=point2, color=color_, buff=buff_)
    if add_pointer:
        line.add_tip(tip_shape=ArrowTriangleFilledTip)
    return line


class FirstGraph(MovingCameraScene):

    def func_circle(self, t):
        return np.array((np.sin(t), np.cos(t), 0))

    def construct(self):
        x = 2.5
        y = 7
        d = 20
        v = 15
        h = 0.3
        l_shift = 4
        l_font_size = 90
        x_y_label_font = 45
        r_time = 2
        t_before_opt_point = 3.5
        axes_reveal_t = 0.01
        t_between_graphs = 0.3
        opt_point_size = 0.2
        camera_scale = 1.5
        label_write_time = 1.5
        time_before_reveal_graph = 3
        slide_x_min = -2
        x_y_label_dist = 0.2
        x_y_label_shift_right = 1.2
        x_y_label_shift_up = 2

        self.play(
            self.camera.frame.animate.set_width(self.camera.frame.get_width() * camera_scale), run_time=0.001
        )

        axes = Axes(
            x_range=[-x, x],
            y_range=[-y * h, y],
            axis_config={"color": BLUE,
                         "include_ticks": False},
        ).shift(DOWN*0.8)

        dummy_graph = axes.plot(lambda x: np.sin(x), color=WHITE)

        def f(x):
            return x ** 2

        # Create Graph
        graph = axes.plot(f, color=WHITE)
        label = MathTex(r'f(x) = x^2', font_size=l_font_size).shift(l_shift * UP * 1)
        coord_labels = axes.get_axis_labels(x_label="x", y_label="y")

        opt_point_starting_location = get_anchor_point(slide_x_min, f(slide_x_min), axes)
        opt_point = Circle(color=YELLOW, fill_opacity=1, radius=opt_point_size).move_to(opt_point_starting_location)

        # Arc tracing by the optiml point.
        graph_arc = axes.plot(f, color=WHITE, x_range=[slide_x_min, 0.001, 0.001])

        def get_coord_lables():
            opt_coord_labels = VGroup()
            label_opt_x = MathTex(r'x: ' + str('{:2.2f}'.format(opt_point.get_center()[0])), font_size=x_y_label_font)
            label_opt_y = MathTex(r'y: ' + str('{:2.2f}'.format(f(opt_point.get_center()[0]))),
                                  font_size=x_y_label_font)
            opt_coord_labels_inner = VGroup()
            opt_coord_labels_inner.add(label_opt_x)
            opt_coord_labels_inner.add(label_opt_y)
            opt_coord_labels.add(opt_coord_labels_inner)
            opt_coord_labels.add(opt_point_starting_location)
            opt_coord_labels_inner.arrange(DOWN, center=False, aligned_edge=LEFT, buff=x_y_label_dist)
            opt_coord_labels_inner.move_to(opt_point_starting_location.get_center()
                                           + RIGHT * x_y_label_shift_right
                                           + UP * x_y_label_shift_up)
            return opt_coord_labels

        opt_coord_labels = always_redraw(lambda: get_coord_lables())

        self.play(Write(label), run_time=label_write_time)
        self.play(Create(axes), run_time=axes_reveal_t)
        self.add(coord_labels)
        self.play(Create(graph), run_time=r_time)
        self.play(Create(opt_point))
        self.add(graph_arc)

        # Animate
        self.play(MoveAlongPath(opt_coord_labels, graph_arc), MoveAlongPath(opt_point, graph_arc), run_time=3.8)


class OneDimensionalGradientDescent(MovingCameraScene):
    def construct(self):
        x = 4
        y = 17
        d = 20
        v = 15
        h = 0.3
        l_shift = 4
        l_font_size = 90
        r_time = 1.7
        axes_reveal_t = 0.01
        t_between_graphs = 0.3
        camera_scale = 1.5
        time_before_reveal_graph = 1.5
        t_between_steps = 0.2
        time_step_reveal = 0.15
        time_segment_reveal = 0.3
        opt_point_size_regular = 0.05
        opt_point_size_final = 0.1
        stroke_width_segment = 5
        x_iters = [-0.95, -0.65, -0.4, -0.25, -0.11, 0]

        self.play(
            self.camera.frame.animate.set_width(self.camera.frame.get_width() * camera_scale), run_time=0.001
        )

        axes = Axes(
            x_range=[-x, x],
            y_range=[-y * h, y],
            axis_config={"color": BLUE,
                         "include_ticks": False},
        ).shift(DOWN*2)

        dummy_graph = axes.plot(lambda x: np.sin(x), color=WHITE)

        def f(x):
            return x ** 2

        # Create Graph
        graph = axes.plot(f, color=WHITE)
        label = MathTex(r'y = x^2', font_size=l_font_size).next_to(axes, UP*8)
        coord_labels = axes.get_axis_labels(x_label="x", y_label="y")

        # Gradient descent
        steps = []
        segments = []
        for ii, x_iter in enumerate(x_iters):
            xx = x_iter * x
            dummy_center = get_anchor_point(xx, f(xx), axes)
            if ii < len(x_iters) - 1:
                steps.append(get_opt_point(opt_point_size_regular, dummy_center))
                segm_arrow = get_arrow_on_graph(axes, x * x_iters[ii], f(x * x_iters[ii]), x * x_iters[ii + 1],
                                                f(x * x_iters[ii + 1]),
                                                stroke_width_segment, dummy_center)
                segments.append(segm_arrow)
        dummy_center = get_anchor_point(x_iters[-1] * x, f(x_iters[-1] * x), axes)
        steps.append(get_opt_point(opt_point_size_final, dummy_center))

        self.add(label)
        self.wait(0.7)
        self.play(Create(axes), run_time=0.9)
        self.add(coord_labels)
        self.wait(0.6)
        self.play(Create(graph), run_time=2.5)
        for step, segment in zip(steps, segments):
            self.play(Create(step), run_time=0.5)
            self.play(Create(segment), run_time=0.5)
        self.play(Create(steps[-1]), run_time=0.6)
        self.wait(12.3)


class Test(ThreeDScene):

    def construct(self):
        self.set_camera_orientation(theta=90 * DEGREES, phi=45 * DEGREES, distance=10)
        c1 = Cube(side_length=0.2).move_to(np.array([2, 0, 0])).set_color(RED)
        h = 3
        c2 = Cube(side_length=0.2).move_to(np.array([-h, 0, -h]))
        c3 = Cube(side_length=0.2).move_to(np.array([h, 0, -h]))
        c4 = Cube(side_length=0.2).move_to(np.array([-h, 0, h]))
        c5 = Cube(side_length=0.2).move_to(np.array([h, 0, h]))
        center = Sphere(radius=0.1)
        self.add(c1, c2, c3, c4, c5, center)
        self.play(Rotate(c1, 2 * PI, axis=np.array([0, 1, 0]), about_point=np.array([0, 0, 0])), run_time=3)
        self.wait(1)


class SurfaceAnimations(ThreeDScene):

    def __init__(self, func_type='quadratic'):
        super().__init__()
        self.n_segments = 5
        self.func = func_type
        self.bb = 0.1
        self.a_1 = 4
        self.a_2 = 2
        self.a_3 = 20
        self.R = 1
        self.graph_scale = 2
        if self.func == "quadratic":
            self.x = 4
            self.y = 4.5
            self.quadratic_range = 2 * PI
        elif self.func == "funky":
            self.x = 0.1
            self.y = 0.45
        else:
            raise ValueError("Wrong func")
        self.x__ = self.x
        self.y__ = self.y
        if self.func == 'quadratic':
            self.init_point_r = 0.3
        elif self.func == 'funky':
            self.init_point_r = 0.07
        else:
            raise ValueError("Wrong func")
        self.t_f_to_surf = 1
        self.t_init_segs = 3
        if self.func == 'funky':
            self.dd = 1
            self.st = 1
            self.ll = 5
        else:
            self.dd = 7.5
            self.st = 1
            self.ll = 10.5
        self.ball_R = 0.02

        self.small = True

        self.cached = {

        }

        self.three_d_surf_position = np.array([-8, -4, 0])

        self.normals = None
        self.steps = None
        self.normals2 = None
        self.steps2 = None
        self.inertia = None
        self.inertia2 = None
        self.factor = None
        self.factor2 = None
        self.axes3D_pos = None
        self.axes = None
        self.n_grad_iters = 10
        self.n_grad_iters_momentum = 10
        self.u_min = 0
        self.second_axis_stretch = 1
        self.surf_added = False
        self.axes_arrows = dict()
        self.step_size = 0.07
        self.step_adjustment = 0.9

        self.surf_exists = False
        self.do_smoothing = False

        self.f = self.get_f()
        self.df_du = self.get_df_du()
        self.df_dv = self.get_df_dv()

        self.prev_coords = None
        self.new_coords = None
        self.surf = None

    def mollifier(self, r):
        if abs(r) < 1:
            c = np.exp(1)
            return c * np.exp(-1/(1 - r**2))
        else:
            return 0

    def mollifier_diff(self, r):
        if abs(r) < 1:
            c = np.exp(1)
            return -2 * r * self.mollifier(r) / (1 - r**2)**2
        else:
            return 0

    def get_normal(self, u, v):
        dzdu = self.df_du(u, v)
        dzdv = self.df_dv(u, v)
        norm_vec = np.array([dzdu, dzdv, -1])
        if norm_vec[2] < 0:
            norm_vec = -norm_vec
        return norm_vec / np.linalg.norm(norm_vec) * self.ball_R

    def smoothen(self, vec):
        a = 0.005
        for idx in range(2, len(vec)):
            if np.linalg.norm(vec[idx] - vec[idx - 1]) > a:
                print("smoothen, index: " +str(idx))
                vec[idx] = vec[idx - 1] + (vec[idx - 1] - vec[idx - 2])
        return vec

    def setup(self):
        if self.func == 'funky':
            self.set_camera_orientation(zoom=1.3)

    def sparsen(self, vec1, vec2):
        idx_to_remove = []
        a = None
        if self.func == 'quadratic':
            a = 0.04
        elif self.func == 'funky':
            a = 0.0002
        top = len(vec1) - 1
        for _ in range(8):
            for i in range(1, top):
                v = np.array(vec1[i])
                v_prev = np.array(vec1[i - 1])
                v_next = np.array(vec1[i + 1])
                if np.linalg.norm(v - v_prev) < a or np.linalg.norm(v - v_next) < a:
                    idx_to_remove.append(i)
            remaining_idx = list(set(range(len(vec1))) - set(idx_to_remove))
            vec1 = [vec1[ii] for ii in remaining_idx]
            vec2 = [vec2[ii] for ii in remaining_idx]
            a = a * 2
            top = math.floor(top * 3 / 4)
        # Remove the long tail
        thresh = 0.08
        window = 5
        idx = len(vec1) - 1
        while abs(vec1[idx][0] - vec1[idx - window][0]) < thresh and abs(vec1[idx][1] - vec1[idx - window][1]) < thresh:
            idx = idx - 1
        vec1 = vec1[:idx]
        vec2 = vec2[:idx]
        return vec1, vec2

    def normalize_range(self, u, v):
        if self.func == 'quadratic':
            vec = np.array([u, v])
            len = np.linalg.norm(vec)
            if len > self.quadratic_range:
                vec = vec / len * self.quadratic_range
            u = vec[0]
            v = vec[1]
            return u, v
        elif self.func == 'funky':
            return u, v

    def normalize_quadratic(self, u, v):
        vec = np.array([u, v])
        len = np.linalg.norm(vec)
        if len > self.quadratic_range:
            vec = vec / len * self.quadratic_range
        u = vec[0]
        v = vec[1]
        return u, v

    def get_symbollic_atoms(self):
        u = Symbol('u')
        v = Symbol('v')
        if self.func == "quadratic":
            y = (u**2 + v**2) * self.bb
        elif self.func == "funky":
            y = 0.3 * (5 * u * v * (1 - u) * (1 - v) * cos(10 * v) * sin(10 * u * v) * exp(u)
                   + exp(-((v - 0.4) ** 2 + (u - 0.2) ** 2) / 0.03)
                   + 0.6 * exp(-((v - 0.4) ** 2) / 0.03) * sin(25*u) * (1-u)**2
                   + 1.4 * exp(-(v - 0.6)**2 / 0.02) * (exp(-(u-0.7)**2 / 0.02) - exp(-(u-0.4)**2 / 0.02))
                      )
        else:
            raise ValueError("Wrong self.func")
        return u, v, y

    def get_f(self):
        if self.func == 'quadratic':
            def my_f(u, v):
                u, v = self.normalize_quadratic(u, v)
                return (u**2 + v**2) * self.bb
            return my_f
        elif self.func == 'funky':
            u, v, y = self.get_symbollic_atoms()
            f_ = lambdify([u, v], y, 'numpy')
            return f_

    def get_df_du(self):
        u, v, y = self.get_symbollic_atoms()
        diff = y.diff(u)
        f_ = lambdify([u, v], diff, 'numpy')
        return f_

    def get_df_dv(self):
        u, v, y = self.get_symbollic_atoms()
        diff = y.diff(v)
        f_ = lambdify([u, v], diff, 'numpy')
        return f_

    def get_tangent_at_angle(self, x_, y_, angle):
        f_x = np.array([1, 0, self.df_du(x_, y_)])
        f_y = np.array([0, 1, self.df_dv(x_, y_)])
        f_x = f_x / np.linalg.norm(f_x)
        f_y = f_y / np.linalg.norm(f_y)
        a = f_x
        c = f_y
        alpha = np.arccos(np.dot(a, c))
        b = (c - np.cos(alpha) * a) / np.sin(alpha)
        tangent_vec = np.cos(angle) * a + np.sin(angle) * b
        # Because of the way manim draws axes, we have to do z = f(x, z), so swap coords.
        tangent_vec = np.array([tangent_vec[0], tangent_vec[2], tangent_vec[1]])
        return tangent_vec / np.linalg.norm(tangent_vec)

    def smoothen_at_index(self, seq, idx):
        window = 4
        idx_start = idx - window
        idx_end = idx + window
        points = []
        points.append(np.array(seq[idx_start + 1]))
        points.append(np.array(seq[idx_start]))
        points.append(np.array(seq[idx_end]))
        points.append(np.array(seq[idx_end - 1]))

        def pairwise_dist(coords):
            sum = 0
            for i in range(len(points)):
                for j in range(i):
                    sum += (np.linalg.norm(points[i] - coords) - np.linalg.norm(points[j] - coords))**2
            return sum

        bnds = ((0.2, 1), (0.2, 1))
        avg = np.zeros((2,))
        for p in points:
            avg += p
        avg = avg / 4
        res = scipy.optimize.minimize(pairwise_dist, x0=avg, bounds=bnds)
        center = res.x
        vec1 = np.array(seq[idx_start + 1]) - center
        vec0 = np.array(seq[idx_end - 1]) - center
        rotation_angle = - np.arccos(np.dot(vec1, vec0) / np.linalg.norm(vec1) / np.linalg.norm(vec0)) / (2*window - 2)

        rotation_matr = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                  [np.sin(rotation_angle), np.cos(rotation_angle)]])
        prev_point = np.array(seq[idx_start + 1])
        for iii in range(idx_start + 2, idx_end - 1):
            v = np.dot(rotation_matr, prev_point - center)
            max_dist = window - 2
            dist = abs(iii - idx)
            scale = 1
            seq[iii] = list(center + v*scale)
            prev_point = center + v

    def grad_descent(self, x, y, first_run=True, idx_to_correct=None, momentum=0):
        x_init, y_init = x, y
        print("grad descent  x: " + str(x) + ", y: " + str(y))
        n_iters = self.n_grad_iters if momentum < 0.0001 else self.n_grad_iters_momentum
        step_size = self.step_size
        step_adjustment = self.step_adjustment
        steps = []
        normals = []
        inertia = []
        idx_of_sharp_angle = 1
        max_angle = 0
        intertia_x = 0
        inertia_y = 0
        first_derivative_norm = None
        moment_step_size = 0.01
        for ii in range(n_iters):
            dx = self.df_du(x, y)
            dy = self.df_dv(x, y)
            if first_derivative_norm is None:
                first_derivative_norm = np.linalg.norm(np.array([dx, dy]))
            else:
                new_derivative_norm = np.linalg.norm(np.array([dx, dy]))
                if new_derivative_norm < first_derivative_norm / 8:
                    if momentum < 0.0001:
                        self.n_grad_iters = len(steps)
                    else:
                        self.n_grad_iters_momentum = len(steps)
                    break
            steps.append([x, y])
            normal_vec = self.get_normal(x, y)
            normals.append(normal_vec)
            if first_run and ii >= 2:
                jj = ii - 1
                prev = np.array(steps[jj]) - np.array(steps[jj-1])
                prev = prev / np.linalg.norm(prev)
                next = np.array(steps[jj+1]) - np.array(steps[jj])
                prev = prev / np.linalg.norm(prev)
                cos_ = np.dot(prev, next)
                angle = np.arccos(np.abs(cos_))
                if angle > max_angle:
                    max_angle = angle
                    idx_of_sharp_angle = jj
            grad = np.array([dx, dy])
            if momentum < 0.001:
                grad = grad / 100
            intertia_x = momentum * intertia_x + (1 - momentum) * grad[0]
            inertia_y = momentum * inertia_y + (1 - momentum) * grad[1]

            if momentum < 0.001:
                x = x - intertia_x
                y = y - inertia_y
            else:
                x = x - moment_step_size * intertia_x
                y = y - moment_step_size * inertia_y
            inertia.append(np.array([intertia_x, inertia_y]))
            step_size = step_size * step_adjustment
        steps.append([x, y])
        normal_vec = self.get_normal(x, y)
        normals.append(normal_vec)
        inertia.append(np.array([intertia_x, inertia_y]))

        if first_run and momentum < 0.001:
            self.grad_descent(x_init, y_init, first_run=False, idx_to_correct=idx_of_sharp_angle, momentum=momentum)
        else:
            if self.do_smoothing and momentum < 0.001:
                self.smoothen_at_index(steps, idx_to_correct)
            normals = []
            for step in steps:
                normals.append(self.get_normal(step[0], step[1]))
            if momentum < 0.001:
                self.normals = normals
                self.steps = steps
            else:
                self.normals2 = normals
                self.steps2 = steps

            # Interpolate points between iterations.
            def fill_sequence(seq, seq_norm, seq_iner):
                new_steps = []
                new_normals = []
                new_iner = []
                n_segments = max(1, min(7, math.floor(400 / len(seq))))
                for i in range(len(seq) - 1):
                    new_steps.append(seq[i])
                    new_normals.append(seq_norm[i])
                    new_iner.append(seq_iner[i])
                    diff = (np.array(seq[i + 1]) - np.array(seq[i])) / n_segments
                    diff_iner = (np.array(seq_iner[i + 1]) - np.array(seq_iner[i])) / n_segments
                    for j in range(1, n_segments):
                        v = list(np.array(seq[i]) + diff*j)
                        v_iner = np.array(seq_iner[i]) + diff_iner*j
                        new_steps.append(v)
                        new_iner.append(v_iner)
                        new_normals.append(self.get_normal(v[0], v[1]))
                new_steps.append(seq[-1])
                new_iner.append(seq_iner[-1])
                new_normals.append(self.get_normal(seq[-1][0], seq[-1][1]))
                return new_steps, new_normals, new_iner

            if momentum < 0.001:
                steps, normals, inert = fill_sequence(self.steps, self.normals, inertia)
                self.steps = steps
                self.normals = normals
                self.inertia = inert
            else:
                steps, normals, inert = fill_sequence(self.steps2, self.normals2, inertia)
                self.steps2 = steps
                self.normals2 = normals
                self.inertia2 = inert

    def inertia_iters(self, t, second=False):
        if second:
            return self.inertia2[max(0, min(math.floor(t), len(self.steps2) - 1))]
        else:
            return self.inertia[max(0, min(math.floor(t), len(self.steps) - 1))]

    def x_iters(self, t, second=False):
        if second:
            return self.steps2[max(0, min(math.floor(t), len(self.steps2) - 1))][0]
        else:
            return self.steps[max(0, min(math.floor(t), len(self.steps) - 1))][0]

    def y_iters(self, t, second=False):
        if second:
            return self.steps2[max(0, min(math.floor(t), len(self.steps2) - 1))][1]
        else:
            return self.steps[max(0, min(math.floor(t), len(self.steps) - 1))][1]

    def normal_x_iters(self, t, second=False):
        if second:
            return self.normals2[max(0, min(math.floor(t), len(self.steps2) - 1))][0] + self.x_iters(t, second=second)
        else:
            return self.normals[max(0, min(math.floor(t), len(self.steps) - 1))][0] + self.x_iters(t)

    def normal_y_iters(self, t, second=False):
        if second:
            return self.normals2[max(0, min(math.floor(t), len(self.steps2) - 1))][1] + self.y_iters(t, second=second)
        else:
            return self.normals[max(0, min(math.floor(t), len(self.steps) - 1))][1] + self.y_iters(t)

    def normal_z_iters(self, t, second=False):
        if second:
            return self.f(self.x_iters(t, second=second), self.y_iters(t, second=second)) + \
                   self.normals2[max(0, min(math.floor(t), len(self.steps2) - 1))][2]
        else:
            return self.f(self.x_iters(t), self.y_iters(t)) + \
                   self.normals[max(0, min(math.floor(t), len(self.steps) - 1))][2]

    def get_surf(self, slice_at=None, res=None):
        self.surf_exists = True
        opacity = 0.5
        if res is None:
            res = 20
        u_range = None
        if self.func == "quadratic":
            u_range = [-self.quadratic_range, self.quadratic_range]
        if self.func == "funky":
            u_range = [self.u_min, self.dd]
        v_range = None
        if self.func == "quadratic":
            v_range = [-self.quadratic_range, self.quadratic_range]
        if self.func == "funky":
            v_range = [0, self.dd*self.second_axis_stretch]

        surf_colors = [BLUE, WHITE]
        if hasattr(self, 'surf_colors'):
            surf_colors = self.surf_colors
        if slice_at is None:
            surf = Surface(
                lambda u, v: self.axes.
                    c2p(self.normalize_range(u, v)[0], self.f(self.normalize_range(u, v)[0],
                                                              self.normalize_range(u, v)[1]),
                        self.normalize_range(u, v)[1]),
                u_range=u_range, v_range=v_range,
                resolution=(res, res))
            surf.set_fill_by_checkerboard(surf_colors[0], surf_colors[1], opacity=0.1)
            return surf
        else:
            v_range1 = [v_range[0], slice_at]
            v_range2 = [slice_at, v_range[1]]

            return [
                Surface(
                    lambda u, v: self.axes.
                        c2p(self.normalize_range(u, v)[0], self.f(u, v), self.normalize_range(u, v)[1]),
                    u_range=u_range, v_range=v_range1,
                    resolution=(res, res)).set_fill_by_checkerboard(surf_colors[0], surf_colors[1], opacity=0.1),
                Surface(
                    lambda u, v: self.axes.
                        c2p(self.normalize_range(u, v)[0], self.f(u, v), self.normalize_range(u, v)[1]),
                    u_range=u_range, v_range=v_range2,
                    resolution=(res, res)).set_fill_by_checkerboard(surf_colors[0], surf_colors[1], opacity=0.1)
            ]

    def transform_obj(self, m):
        return m

    def get_steepest_descent_dir(self, t, second=False):
        x = self.x_iters(t, second=second)
        y = self.y_iters(t, second=second)
        z = self.f(x, y)

        print("x: " + str(x) + ", y: " + str(y) + ", z:" + str(z))

        grad = np.array([self.df_du(x, y), self.df_dv(x, y)])
        print("grad: " + str(grad))
        dxy = 2
        new_xy = np.array([x, y]) - dxy * grad / np.linalg.norm(grad)
        directional_derivative = np.linalg.norm(grad)
        new_z = self.f(x, y) - directional_derivative * dxy
        new_point = np.array([new_xy[0], new_z, new_xy[1]])
        dir = new_point - np.array([x, z, y])
        return dir / np.linalg.norm(dir)

    def rotate_to_current_position(self, m, t, radius, second=False):
        diff = self.new_coords - self.prev_coords
        dist = np.linalg.norm(diff)
        normal_start = self.axes.c2p(0, 0, 0)
        x = self.normal_x_iters(t, second)
        y = self.normal_y_iters(t, second)
        z = self.normal_z_iters(t, second)
        normal_end = self.axes.c2p(x, z, y)
        normal = normal_end - normal_start
        normal = normal / np.linalg.norm(normal)
        axis_ = np.cross(diff, normal)
        angle = - dist / (2 * PI * radius) * 2 * PI
        m.rotate(axis=axis_, angle=angle)
        return m

    def xx(self, t_):
        return self.R * np.cos(t_) + PI / 2

    def yy(self, t_):
        return self.R * np.sin(t_) + PI / 2

    def get_axes(self):
        if self.axes is not None:
            return self.axes
        else:
            x_length = self.ll
            y_length = None
            if self.func == 'quadratic':
                y_length = self.ll
            elif self.func == 'funky':
                y_length = self.ll * 0.6
            else:
                raise ValueError("Wrong func")
            z_length = self.ll
            if hasattr(self, 'axis_config'):
                print('axis_config')
                ax = ThreeDAxes(x_range=[-self.dd, self.dd, self.st], x_length=x_length,
                                y_range=[-self.dd, self.dd, self.st], y_length=y_length,
                                z_range=[-self.dd * self.second_axis_stretch, self.dd * self.second_axis_stretch,
                                         self.st],
                                z_length=z_length, axis_config=self.axis_config,
                                x_axis_config={})
            else:
                print('no axis_config')
                ax = ThreeDAxes(x_range=[-self.dd, self.dd, self.st], x_length=x_length,
                                y_range=[-self.dd, self.dd, self.st], y_length=y_length,
                                z_range=[-self.dd*self.second_axis_stretch, self.dd*self.second_axis_stretch, self.st],
                                z_length=z_length,
                                axis_config={"include_ticks": False})
            if hasattr(self, 'axes_color'):
                ax.set_color(self.axes_color)
            if hasattr(self, 'graph_scale'):
                ax.scale(self.graph_scale)
            ax.move_to(self.three_d_surf_position)
            if self.func == 'funky':
                ax.rotate(axis=UP, angle=PI/6)
                ax.rotate(axis=RIGHT, angle=PI / 4)
            if self.axes3D_pos is not None:
                ax.move_to(self.axes3D_pos)
            self.axes = ax
            return ax

    def get_paths(self):
        if not self.steps:
            self.grad_descent(self.x, self.y)
        point_iters = [np.array([step[0], step[1]]) for step in self.steps]

        paths = []

        distort_magnitude = None
        if self.func == 'quadratic':
            distort_magnitude = 0.5
        elif self.func == 'funky':
            distort_magnitude = 0.1
        else:
            raise ValueError("Wrong func")

        def distort1(points):
            start_ = points[0]
            end_ = points[-1]
            delta_x = end_[0] - start_[0]
            delta_y = end_[1] - start_[1]
            ort_dir = np.array([-delta_y / delta_x, 1])
            ort_dir = ort_dir / np.linalg.norm(ort_dir)
            new_points = []
            for idx, p in enumerate(points):
                if idx == 0 or idx == len(points) - 1:
                    new_points.append(p)
                else:
                    dx = (p[0] - start_[0]) / delta_x
                    p_ = p + (distort_magnitude * np.sin(2 * PI * dx)) * ort_dir
                    new_points.append(p_)
            return new_points

        def distort2(points):
            start_ = points[0]
            end_ = points[-1]
            delta_x = end_[0] - start_[0]
            delta_y = end_[1] - start_[1]
            ort_dir = np.array([-delta_y / delta_x, 1])
            ort_dir = ort_dir / np.linalg.norm(ort_dir)
            new_points = []
            for idx, p in enumerate(points):
                if idx == 0 or idx == len(points) - 1:
                    new_points.append(p)
                else:
                    dx = (p[0] - start_[0]) / delta_x
                    p_ = p + (distort_magnitude * np.cos(2 * PI * dx)) * ort_dir
                    new_points.append(p_)
            return new_points

        paths.append(point_iters)
        paths.append(distort1(point_iters))
        paths.append(distort2(point_iters))

        segs = []
        colors = [RED, YELLOW, GREEN]
        for point_iters, color_ in zip(paths, colors):
            segments = []
            for ii, coords in enumerate(point_iters):
                if ii < len(point_iters) - 1:
                    x1 = point_iters[ii][0]
                    y1 = point_iters[ii][1]
                    z1 = self.f(x1, y1)
                    x2 = point_iters[ii + 1][0]
                    y2 = point_iters[ii + 1][1]
                    z2 = self.f(x2, y2)
                    segments.append(
                        Line(start=self.axes.c2p(x1, z1, y1), end=self.axes.c2p(x2, z2, y2), color=color_, buff=0).add_tip(
                            tip_shape=ArrowTriangleFilledTip, tip_length=0.1).set_opacity(0))
            segs.append(segments)
        self.paths = segs
        return segs

    def get_init_point(self):
        res = 5 if self.small else 20
        init_point = Sphere(
            center=self.axes.c2p(self.x__, self.f(self.x__, self.y__), self.y__),
            radius=self.init_point_r,
            resolution=(res, res),
            u_range=[0.001, PI - 0.001],
            v_range=[0, TAU]
        ).set_color(YELLOW).set_opacity(1).set_z_index(3)
        self.init_point = init_point
        return init_point

    def get_all_objects(self, get_surf=True):
        vg = VGroup()

        # Axes
        if self.axes is None:
            axes3D = self.get_axes()
            vg.add(axes3D)
            self.axes = axes3D

        # Surface
        if get_surf and self.surf is None:
            surf = self.get_surf()
            vg.add(surf)
            self.surf = surf

        segs = self.get_paths()
        for seg in segs:
            for seg_ in seg:
                vg.add(seg_)
        self.paths = segs

        # Init point
        init_point = self.get_init_point()
        vg.add(init_point)
        vg = self.transform_obj(vg)
        return vg

    def get_label(self):
        if self.func == 'quadratic':
            return MathTex(r'f(x, y) = x^2 + y^2', font_size=three_d_f_font_size)
        elif self.func == 'funky':
            return MathTex(r'f(x, y) &= 0.3 \Biggl[5xy(1 - x)(1 - y)\cos(10y)\sin(10xy)e^x\\ &+ \exp{-\frac{(y - 0.4)^2 + (x - 0.2)^2}{0.03}} \\&+ 0.6 \exp{-\frac{(v - 0.4)^2}{0.03}} \sin(25 x) (1 - x)^2 \\&+ 1.4 \exp{-\frac{(y - 0.6)^2}{0.02}} \Bigl(\exp{-\frac{(x - 0.7)^2}{0.02}} - \exp{-\frac{(x - 0.4)^2}{0.02}}\Bigr)\Biggr]',
                           font_size=three_d_f_font_size)
        else:
            raise ValueError('Wrong func')

    def precompute_mobjects(self, momentum=0):
        config.disable_caching = True
        if not self.steps or isinstance(self, RollingBall):
            self.grad_descent(self.x, self.y, momentum=momentum)
        if self.steps is not None:
            self.factor = 0.9 * len(self.steps) / (2 * PI)
        if self.steps2 is not None:
            self.factor2 = 0.9 * len(self.steps2) / (2 * PI)
        return self.get_all_objects()

    def get_factor(self, second=False):
        if second:
            return self.factor2
        else:
            return self.factor

    def construct(self):
        pass


class MultiplePaths(SurfaceAnimations):

    def __init__(self):
        super().__init__(func_type='funky')
        self.n_segments = 12
        self.three_d_surf_position = np.array([-3, 1, 0])

    def grad_descent(self, x, y, momentum=None):
        print("grad descent  x: " + str(x) + ", y: " + str(y))
        n_iters = 10
        step_size = 0.07
        step_adjustment = 0.9
        steps = []
        normals = []
        for _ in range(n_iters):
            dx = self.df_du(x, y)
            dy = self.df_dv(x, y)
            steps.append([x, y])
            normal_vec = self.get_normal(x, y)
            normals.append(normal_vec)
            grad = np.array([dx, dy])
            grad = grad / np.linalg.norm(grad)
            x = x - step_size * grad[0]
            y = y - step_size * grad[1]
            step_size = step_size * step_adjustment
        steps.append([x, y])
        normal_vec = self.get_normal(x, y)
        normals.append(normal_vec)
        self.normals = normals
        self.steps = steps
        self.normals = self.smoothen(self.normals)

    def construct(self):
        self.setup()
        self.precompute_mobjects()

        # Formula & graph
        f_tex = self.get_label()
        self.play(Write(f_tex), run_time=2)
        self.wait(3.5)
        self.play(Uncreate(f_tex), run_time=0.5)
        self.wait(0.5)
        self.play(Create(self.surf), run_time=1.5)
        self.play(self.surf.animate.set_opacity(1), run_time=0.5)
        self.wait(2.5)

        # Initial point
        init_point = self.init_point
        init_point.set_opacity(1)
        self.add(init_point)
        self.play(Create(init_point), run_time=0.5)

        # Print paths
        self.wait(1.5)
        segments = self.paths
        for path_id, path in enumerate(segments):
            for seg in path:
                seg.set_opacity(1)
                self.add(seg)
                if path_id == 0:
                    self.play(Create(seg), run_time=0.22)
                else:
                    self.play(Create(seg), run_time=0.22)
            if path_id == 0:
                self.wait(1.5)
            else:
                self.wait(0.5)


class WhichWay(SurfaceAnimations):

    def __init__(self):
        super().__init__(func_type='funky')
        self.graph_scale = 5.5

    def construct(self):
        left_graph_scaling_factor = 1.5
        x = 4
        y = 2
        xx = -12
        yy = -5
        h = 0.3
        l_shift = 4
        graph_shift = 12
        arrow_length = 0.2
        init_point_r = 0.05

        self.set_camera_orientation(zoom=0.3)

        # 1D

        vg_1D = VGroup()

        axes = Axes(
            x_range=[-x, x],
            y_range=[-y, y],
            axis_config={"color": WHITE,
                         "include_ticks": False},
        ).set_color(WHITE)
        vg_1D.add(axes)

        def f(x):
            return np.sin(x)

        def df(x):
            return np.cos(x)

        graph = axes.plot(f, color=BLUE)
        vg_1D.add(graph)

        coord_labels = axes.get_axis_labels(x_label="x", y_label="y")
        vg_1D.add(coord_labels)

        xx = 0.6
        dummy_point = axes.plot(lambda x: np.sin(x - xx) + f(xx), x_range=[xx - 1, xx + 1], color=BLUE)
        vg_1D.add(dummy_point)

        point = Circle(color=YELLOW, fill_opacity=1, radius=3*init_point_r).move_to(dummy_point.get_center())
        vg_1D.add(point)

        dx = 1
        dir_end_dummy = axes.plot(lambda x: np.sin(x - xx - dx) + f(xx) + dx * df(xx),
                                  x_range=[xx + dx - 1, xx + dx + 1], color=WHITE)
        vg_1D.add(dir_end_dummy)

        neg_dir_end_dummy = axes.plot(lambda x: np.sin(x - xx + dx) + f(xx) - dx * df(xx),
                                      x_range=[xx - dx - 1, xx - dx + 1], color=WHITE)
        vg_1D.add(neg_dir_end_dummy)

        offset = 0*np.array([0, 0, 0.01])

        dir = Line(start=point.get_center() + offset, end=dir_end_dummy.get_center() + offset,
                   color=RED, buff=0.05).add_tip(
            tip_shape=ArrowTriangleFilledTip)
        point.set_z_index(dir.z_index + 1)
        vg_1D.add(dir)

        neg_dir = Line(start=point.get_center() + offset, end=neg_dir_end_dummy.get_center() + offset,
                       color=RED, buff=0.05).add_tip(
            tip_shape=ArrowTriangleFilledTip)
        vg_1D.add(neg_dir)

        vg_1D.scale(left_graph_scaling_factor)
        vg_1D.shift(graph_shift * LEFT)

        # 2D
        self.axes3D_pos = axes.get_center() + np.array([0, 2.5, 0])
        vg_2D = self.get_all_objects()
        vg_2D.shift(graph_shift * RIGHT)

        # 1D
        self.play(Create(axes), run_time=0.01)
        self.add(coord_labels)
        self.play(Create(graph), run_time=1.7)
        self.play(Create(point), run_time=0.8)
        self.play(Create(dir), run_time=0.8)
        self.play(Create(neg_dir), run_time=0.8)

        self.play(Create(self.surf), run_time=0.8)
        self.play(self.surf.animate.set_opacity(0.8), run_time=0.5)
        xx = 0.61
        yy = 0.57

        init_point_coords = self.axes.c2p(xx, self.f(xx, yy), yy)

        angle = ValueTracker(0)

        def get_rotated_dir(t):
            end = self.get_tangent_at_angle(xx, yy, t) * arrow_length + np.array(
                [xx, self.f(xx, yy), yy])
            return self.axes.c2p(*end)

        def get_rotated_dir_arrow(t):
            end_ = get_rotated_dir(t)
            return Line(start=init_point_coords, end=end_, color=RED, buff=0.05)\
                .add_tip(tip_shape=ArrowTriangleFilledTip)

        dir = always_redraw(lambda: get_rotated_dir_arrow(angle.get_value()))

        trajectory = always_redraw(lambda: ParametricFunction(
            lambda t: get_rotated_dir(t), t_range=np.array([0, angle.get_value(), 0.01]), color=YELLOW
        )
                                   )
        self.add(trajectory)

        # Draw the init point
        init_point = Sphere(
            center=init_point_coords,
            radius=init_point_r*4,
        ).set_color(YELLOW).set_opacity(1)
        init_point.set_z_index(dir.z_index + 6)

        self.play(Create(init_point), run_time=0.3)
        self.wait(1)

        # Animate the direction arrow
        self.add(dir)
        self.play(angle.animate.set_value(2 * PI), run_time=2, rate_func=linear)
        self.play(FadeOut(dir), Uncreate(trajectory), run_time=0.5)

        # Draw the steepest descent arrow.
        grad = np.array([self.df_du(xx, yy), self.df_dv(xx, yy)])
        directional_derivative = np.linalg.norm(grad)
        grad_normalized = grad / directional_derivative
        diff = np.array([grad_normalized[0], directional_derivative, grad_normalized[1]])
        diff = diff / np.linalg.norm(diff)
        new_point = np.array([xx, self.f(xx, yy), yy]) - arrow_length * 1.5 * diff
        grad_arrow = Line(start=init_point_coords, end=self.axes.c2p(*new_point),
                          color=RED, buff=0.05).add_tip(
            tip_shape=ArrowTriangleFilledTip)
        self.wait(4.5)
        self.play(Create(grad_arrow), run_time=1.5)
        self.wait(4.5)


class Conclusion(SurfaceAnimations):

    def __init__(self):
        super().__init__(func_type='funky')
        self.graph_scale = 5.5

        # from Execution
        self.r = 0.18

        # from StepSize
        self.x = -1
        self.y = -0.35
        self.ll = 7
        self.dist_formula_graph = 7.5
        self.plot_f_value = False
        self.f_val = None
        self.prev_f_vals = []
        self.init_point_colors = [BLUE, RED]
        self.init_point_current_color = 0
        self.e_arrow = ValueTracker(0)
        self.times_color_changed = 0

    def invert_x_y(self, vec):
        return np.array([vec[2], vec[1], vec[0]])

    def get_init_point(self):
        print("get_init_point 3")
        args = self.get_args(self.e_arrow.get_value())
        f_val = self.f(args[0], args[1])
        delta_ = 1e-4
        print("arg: " + str(self.e_arrow.get_value()))
        if len(self.prev_f_vals) >= 2 and abs(f_val - self.prev_f_vals[-1]) > delta_ \
                and (f_val - self.prev_f_vals[-1]) > delta_ \
                and (self.prev_f_vals[-1] - self.prev_f_vals[-2]) < - delta_\
                and self.e_arrow.get_value() < 0.05:
            print("1 cond satisfied")
            if self.times_color_changed == 1:
                self.init_point_current_color = 1 - self.init_point_current_color
                print("1 cond satisfied, init_point_current_color: " + str(self.init_point_current_color))
            self.times_color_changed += 1
        self.prev_f_vals.append(f_val)
        p = np.array([args[1], f_val, args[0]])
        col = 0 if self.e_arrow.get_value() < 0.5 else 1
        return Sphere(radius=self.r * 0.6).set_color(self.init_point_colors[col]).\
            move_to(self.axes.c2p(*p))


    def adjust_labels(self, label_x, label_y, label_z):
        label_x.shift(np.array([-0.5, 0, 0]))
        label_y.shift(np.array([-0.5, -0.3, 0]))

    def get_z_helper(self, str1, str2, color1, color2, font_size):
        fff = MathTex(str1, str2, font_size=font_size)
        shift = np.array([-2.05, 1, 0]) - fff[0].get_center()
        fff.shift(shift)
        fff[0].set_color(color1)
        fff[1].set_color(color2)
        return fff

    def get_z_label(self, font_size, color):
        str1 = r'f(x, y)'
        args = self.get_args(self.e_arrow.get_value())
        f_val = self.f(args[0], args[1])
        str2 = r' = ' + str("{:2.2f}".format(f_val)) if self.plot_f_value else r' = 0.00'
        color2 = RED if self.plot_f_value else BLACK
        return self.get_z_helper(str1=str1, str2=str2, color1=color, color2=color2, font_size=font_size)

    def get_args(self, t):
        init_ = np.array([self.x, self.y])
        target_ = -init_
        point = t * target_ + (1 - t) * init_
        return point

    def construct(self):
        left_graph_scaling_factor = 1.5
        x = 4
        y = 2
        xx = -12
        yy = -5
        h = 0.3
        l_shift = 4
        graph_shift = 12
        arrow_length = 0.2
        init_point_r = 0.05

        self.set_camera_orientation(zoom=0.3)

        font_size = 120
        hor_shift = 13
        vert_shift = 20

        iterative = Text(r'Iterative', font_size=font_size).move_to([-hor_shift, 9.5, 0])
        self.wait(4.5)
        self.play(Write(iterative), run_time=1.5)

        f_grad = MathTex(r'- \nabla f(x, y)',
                    r'= \left(-\frac{\partial f}{\partial x}, -\frac{\partial f}{\partial y}\right)',
                    font_size=font_size).next_to(iterative, DOWN * vert_shift)

        # 2D
        self.axes3D_pos = np.array([-13, 5, 0])
        vg_2D = self.get_all_objects()
        vg_2D.shift(graph_shift * RIGHT + UP*3).scale(0.7)

        self.wait(3)
        self.play(Write(f_grad), Create(self.surf), run_time=0.8)
        self.play(self.surf.animate.set_opacity(0.8), run_time=0.5)
        xx = 0.61
        yy = 0.57

        init_point_coords = self.axes.c2p(xx, self.f(xx, yy), yy)

        angle = ValueTracker(0)

        def get_rotated_dir(t):
            end = self.get_tangent_at_angle(xx, yy, t) * arrow_length + np.array(
                [xx, self.f(xx, yy), yy])
            return self.axes.c2p(*end)

        def get_rotated_dir_arrow(t):
            end_ = get_rotated_dir(t)
            return Line(start=init_point_coords, end=end_, color=RED, buff=0.05)\
                .add_tip(tip_shape=ArrowTriangleFilledTip)

        # Find the best angle
        grad = np.array([self.df_du(xx, yy), self.df_dv(xx, yy)])
        directional_derivative = np.linalg.norm(grad)
        grad_normalized = grad / directional_derivative
        descent = np.array([grad_normalized[0], directional_derivative, grad_normalized[1]])
        angle_best = None
        score_best = 100000
        for t in np.linspace(0, 2*PI, num=1000):
            d = self.get_tangent_at_angle(xx, yy, t)
            score = np.dot(d, descent)
            if score < score_best:
                score_best = score
                angle_best = t

        dir = always_redraw(lambda: get_rotated_dir_arrow(angle.get_value()))

        trajectory = always_redraw(lambda: ParametricFunction(
            lambda t: get_rotated_dir(t), t_range=np.array([0, angle.get_value(), 0.01]), color=YELLOW
        )
                                   )
        self.add(trajectory)

        # Draw the init point
        init_point = Sphere(
            center=init_point_coords,
            radius=init_point_r*4,
        ).set_color(YELLOW).set_opacity(1)
        init_point.set_z_index(dir.z_index + 6)

        self.play(Create(init_point), run_time=0.6)

        # Animate the direction arrow
        self.add(dir)
        self.play(angle.animate.set_value(angle_best), run_time=2.5, rate_func=linear)
        self.play(Uncreate(trajectory), run_time=0.5)

        # Steps size
        f_step = MathTex(r'x &:= x - \alpha \frac{\partial f}{\partial x} \\',
                         r'y &:= y - \alpha \frac{\partial f}{\partial y}',
                    font_size=font_size).next_to(f_grad, DOWN * vert_shift)
        self.play(Write(f_step))

        stepsize_obj = StepSize(self, f_step.get_center() + RIGHT*20)
        stepsize_obj.construct()


class RollingBall(SurfaceAnimations):

    def __init__(self):
        super().__init__(func_type='funky')
        self.ball = 0
        self.graph_scale = 13
        self.axes3D_pos = np.array([-23, 6, 0])
        self.x = 0.27
        self.y = 0.41
        self.n_grad_iters = 71
        self.n_grad_iters_momentum = 300
        self.second_axis_stretch = 1
        self.step_size = 0.007
        self.step_adjustment = 0.999
        self.trajectories = []
        self.e = []
        self.tracker = []
        self.momentum = 0

    def get_all_objects(self):
        vg = VGroup()

        # Axes
        if self.axes is None:
            axes3D = self.get_axes()
            vg.add(axes3D)
            self.axes = axes3D

        # Surface
        if not self.surf_exists:
            surf = self.get_surf()
            vg.add(surf)
            self.surf = surf

        vg = self.transform_obj(vg)
        return vg

    def roll_ball(self, momentum):
        second = False if momentum < 0.001 else True
        if self.ball is not None:
            self.remove(self.ball)
        self.precompute_mobjects(momentum)
        self.camera.exponential_projection = True
        self.set_camera_orientation(zoom=0.3)

        if not self.surf_added:
            self.surf_added = True
            self.add(self.surf)
            self.play(Create(self.surf), run_time=1)
            self.play(self.surf.animate.set_opacity(1), run_time=1)

        # Ball
        checkerboard_colors = [RED, YELLOW] if abs(momentum) < 1e-9 else [GREEN, WHITE]
        ball_res = 30

        # This is needed because of the bad software design (didn't take into account that the second element in
        # these arrays might be needed when the first one is missing).
        if momentum > 0.001 and len(self.tracker) == 0:
            self.tracker.append(None)
            self.e.append(None)

        self.tracker.append(ValueTracker(0))
        self.e.append(self.tracker[-1].get_value())

        p1_ = self.axes.c2p(0, 0, 0)
        p2_ = self.axes.c2p(0, 0, 1)
        scaling_factor = np.linalg.norm(p1_ - p2_)
        ball = Sphere(radius=self.ball_R*scaling_factor).set_color(RED).set_opacity(1)
        new_coords = self.axes.c2p(self.x_iters(self.e[-1] * self.get_factor(second), second),
                                   self.f(self.x_iters(self.e[-1] * self.get_factor(second), second),
                                          self.y_iters(self.e[-1] * self.get_factor(second), second)),
                                   self.y_iters(self.e[-1] * self.get_factor(second), second))
        self.prev_coords = new_coords
        self.new_coords = new_coords
        self.rotate_to_current_position(ball, self.e[-1] * self.get_factor(second), self.ball_R, second=second)
        ball.move_to(new_coords)
        if momentum < 0.001:
            self.wait(3)
        else:
            self.wait(1)
        self.play(Create(ball), run_time=1)
        self.wait(1)

        def get_grad_arrow(t):
            descent_dir = self.get_steepest_descent_dir(t, second)
            start_original_coords = np.array([self.x_iters(t, second=second), self.f(self.x_iters(t, second=second),
                                            self.y_iters(t, second=second)), self.y_iters(t, second=second)])
            end_original_coords = start_original_coords + descent_dir * 0.11
            # Shorten the arrow to account for object overlap not working correctly in manim.
            start_original_coords = start_original_coords + 0.2 * (end_original_coords - start_original_coords)
            start_ = self.axes.c2p(*start_original_coords)
            end_ = self.axes.c2p(*end_original_coords)
            return Line(start=start_, end=end_, color=RED, buff=0.02).add_tip(tip_shape=ArrowTriangleFilledTip)

        # Trajectory
        trajectory_color = RED
        trajectory_arg_start = 0

        def get_trajectory():
            e_idx = 1 if second else 0
            print("e: " + str(self.e[e_idx]))
            print("t_range: " + str(np.array([0, max([0, self.e[e_idx] - trajectory_arg_start]), 0.01])))
            return ParametricFunction(
                lambda t: self.axes.c2p(self.x_iters(t * self.get_factor(second), second),
                                        self.f(self.x_iters(t * self.get_factor(second), second),
                                               self.y_iters(t * self.get_factor(second), second)),
                                        self.y_iters(t * self.get_factor(second), second)),
                t_range=np.array([0, max([0, self.e[e_idx] - trajectory_arg_start]), 0.01]),
                color=trajectory_color
            )

        trajectory = always_redraw(lambda: get_trajectory())
        self.trajectories.append(trajectory)
        self.add(trajectory)

        if not second and self.momentum < 0.001:
            arrow = get_grad_arrow(0)
            self.play(Write(arrow))
            self.wait(1)
            self.play(Unwrite(arrow))

        def get_grad_arrow_current(t):
            line = get_grad_arrow(t * self.get_factor(second))
            if t > (2 - 1 / 5) * PI:
                line.set_opacity(0)
            return line

        # Animate the ball rolling.
        def update_ball(m):
            self.e[-1] = self.tracker[-1].get_value()
            self.prev_coords = self.new_coords

            new_coords = self.axes.c2p(self.x_iters(self.e[-1] * self.get_factor(second), second),
                                       self.f(self.x_iters(self.e[-1] * self.get_factor(second), second),
                                              self.y_iters(self.e[-1] * self.get_factor(second), second)),
                                       self.y_iters(self.e[-1] * self.get_factor(second), second))

            self.new_coords = new_coords
            ee = self.e[0] if not second else self.e[1]
            self.rotate_to_current_position(m, ee * self.get_factor(second), self.ball_R*scaling_factor, second=second).\
                move_to(new_coords)

        ball.add_updater(
            lambda m: update_ball(m)
        )

        # Inertia arrow
        def get_intertia_arrow():
            x = self.x_iters(self.e[1]* self.get_factor(second), second)
            y = self.y_iters(self.e[1] * self.get_factor(second), second)
            inertia = - self.inertia_iters(self.e[1] * self.get_factor(second), second)
            grad = np.array([self.df_du(x, y), self.df_dv(x, y)])
            dxy = 0.1
            new_xy = np.array([x, y]) + dxy * inertia
            directional_derivative = np.dot(grad, inertia) / np.linalg.norm(inertia)
            new_z = self.f(x, y) + directional_derivative * dxy
            new_point = np.array([new_xy[0], new_z, new_xy[1]])
            start_ = self.axes.c2p(x, self.f(x, y), y)
            end_ = self.axes.c2p(*new_point)
            # Shorten the arrow to account for object overlap not working correctly in manim.
            start_ = start_ + 0.2 * (end_ - start_)
            print("inertia start: " + str(start_))
            print("inertia end: " + str(end_))
            return Line(start=start_, end=end_, color=GREEN, buff=0.02).add_tip(tip_shape=ArrowTriangleFilledTip)

        self.play(
            self.tracker[-1].animate.set_value(2 * PI),
            run_time=6,
            rate_func=linear,
        )
        self.wait(2)

        self.ball = ball

    def construct(self):
        self.roll_ball(self.momentum)
        self.wait(5)
        self.roll_ball(0.85)


class WithMomentum(RollingBall):
    def __init__(self):
        super().__init__()
        self.momentum = 0.9


class MyTest(Scene):
    def construct(self):
        eq_1 = Tex(r"$f(x)\,$", "$=$", "$\,\,\,\,^2$", font_size=30)
        eq_2 = Tex(r"$x$", "$=$", "$\,\,\,\,\,\,-1$", font_size=30).next_to(eq_1, DOWN)
        eq_2.get_part_by_tex('$=$').align_to(eq_1.get_part_by_tex('$=$'), LEFT)
        self.play(Write(eq_1))
        self.play(Write(eq_2))


class Test2(ThreeDScene):
    dd = 5
    st = 0.1
    ll = 5

    def construct(self):
        mx = PI
        ax = ThreeDAxes(x_range=[-self.dd, self.dd, self.st], x_length=self.ll,
                        y_range=[-self.dd, self.dd, self.st], y_length=self.ll,
                        z_range=[-self.dd, self.dd, self.st], z_length=self.ll).add_coordinates()

        def f(u, v):
            vec = np.array([u, v])
            if np.linalg.norm(vec) > mx:
                vec = vec / np.linalg.norm(vec) * mx
            u = vec[0]
            v = vec[1]
            return u, (u ** 2 + v ** 2) * 0.3, v

        surface = Surface(
            lambda u, v: ax.c2p(*f(u, v)),
            u_range=[-PI, PI],
            v_range=[-PI, PI],
            resolution=8,
        ).set_opacity(0.5).shift(4 * DOWN)

        self.add(ax)
        self.play(Create(surface))


class GiganticFunction(Scene):

    def construct(self):
        shift = 3
        font_size = 40
        label_font = 120
        opt_point_size = 0.27
        N = 8

        def color_formula(f):
            f[0].set_color(BLUE)
            for i in range(1, N):
                f[i].set_color(YELLOW)
            return f

        formula = color_formula(MathTex(r'f(x) &= ',
                          r'4x^3',
                          r'- x^2',
                          r'- \cos(4x)\\',
                          r'&- \sin(4x) \cos(6x)',
                          r' - \sin(2x) x^3\\',
                          r'&- \cos(3x) \sin(5x)',
                          r'- x e^{1-x}',
                          font_size=font_size)
                                )
        center_position = formula.get_center()
        formula_shift = LEFT*shift
        formula2 = color_formula(MathTex(r'f(x) &= ',
                          r'4x^3',
                          r'- x^2',
                          r'- \cos(4x)\\',
                          r'&- \sin(4x) \cos(6x)',
                          r' - \sin(2x) x^3\\',
                          r'&- \cos(3x) \sin(5x)',
                          r'- x e^{1-x}',
                          font_size=font_size)
                                 )
        new_formulas = [formula2[i] for i in range(N)]
        orig_pos = []
        dist = 2.5
        angle = PI/10
        angle_inc = 2*PI/8
        for i in range(N):
            ff = formula[i]
            orig_pos.append(ff.get_center())
            alpha = angle + i * angle_inc
            vec = dist * (math.cos(alpha) * RIGHT + math.sin(alpha) * UP)
            print("alpha " + str(alpha) + ", vec: " + str(vec))
            ff.shift(vec)
        np.random.seed(1)

        fs = [Write(formula[i]) for i in range(N)]
        self.play(*fs, run_time=1)
        starting_points = VGroup(*[formula[i] for i in range(N)])
        finish_points = VGroup(*[formula2[i] for i in range(N)])
        self.play(
            Transform(
                starting_points,
                finish_points,
                path_func=utils.paths.spiral_path(TAU/4),
                run_time=5,
            )
        )
        anims = [formula[i].animate.move_to(orig_pos[i] + formula_shift) for i in range(N)]
        self.play(*anims, run_time=2.5)
        self.wait(1)

        # Plot graph

        def f(x):
            res = 0
            res = res + 4 * x ** 3
            res = res - x**2
            res = res - math.cos(4*x)
            res = res - math.sin(4*x) * cos(6*x)
            res = res - math.sin(2*x) * x**3
            res = res - math.cos(3*x)*math.sin(5*x)
            res = res - x * np.exp(1-x)
            return res

        scale = 0.3
        logistic_rng = 1
        ax = Axes(
            axis_config={
                "include_ticks": False,
                "color": WHITE
            },
            x_length=5 / scale,
            y_length=3.5 / scale,
            x_range=[-logistic_rng, logistic_rng, 1],
            y_range=[-3, 5, 1],
        ).shift(RIGHT*shift).shift(RIGHT*0.5)

        x_label = MathTex(r'x', font_size=label_font).shift(RIGHT*shift).shift(RIGHT*0.5).shift(RIGHT*8.9)
        y_label = MathTex(r'f(x)', font_size=label_font).shift(RIGHT*shift).shift(RIGHT*0.5).shift(UP*5 + RIGHT*1.5)

        curve = ax.plot(f, x_range=[-logistic_rng, logistic_rng])
        curve.set_stroke_color([BLUE])

        opt_point = Circle(color=YELLOW, fill_opacity=1, radius=opt_point_size).move_to(ax.c2p(0.167, -2.15))

        v_log = VGroup(ax, curve, x_label, y_label, opt_point)
        v_log.scale(scale)

        self.play(Create(ax), Write(x_label), Write(y_label), run_time=0.6)
        self.play(Create(curve), run_time=1.6)
        self.play(Create(opt_point), run_time=0.6)
        self.wait(1.1)


class CatDog(Scene):
    def construct(self):
        vert_dist = 2
        horiz_dist = 4

        # Neural network
        r = 0.1
        neuron_color = BLUE_E
        neural_dist_horizontal = 0.8
        neural_dist_vertical = 0.6
        layer_anim_time = 0.2
        n_neurons = [7, 7, 7, 4, 1]
        layers = []
        connections = []
        network_center = LEFT * horiz_dist + DOWN * vert_dist + RIGHT * 0.5
        network_start = network_center + LEFT * 2 + UP * 2

        def get_neuron_coords(layer_idx, neuron_idx, max_neurons):
            coords = network_start + layer_idx * neural_dist_horizontal * RIGHT \
                     + neuron_idx * neural_dist_vertical * DOWN
            if layer_idx >= 1 and n_neurons[layer_idx] < max_neurons:
                coords = coords + (max_neurons - n_neurons[layer_idx]) * neural_dist_vertical / 2 * DOWN
            return coords

        for layer_idx, n in enumerate(n_neurons):
            layer = []
            for neuron_idx in range(n):
                neuron = Circle(radius=r, color=neuron_color, stroke_color=BLUE_A, stroke_width=2.5, fill_opacity=1).\
                    move_to(get_neuron_coords(layer_idx, neuron_idx, max_neurons=n_neurons[0])).set_z_index(1)
                layer.append(neuron)
            layers.append(layer)
            if layer_idx >= 1:
                layer_connections = []
                for i in range(n_neurons[layer_idx - 1]):
                    for j in range(n_neurons[layer_idx]):
                        print("prev_layer=" + str(layer_idx - 1) + ", layer=" + str(layer_idx) + ", con  i=" + str(i) + ", j=" + str(j))
                        start_ = get_neuron_coords(layer_idx - 1, i, max_neurons=n_neurons[0])
                        end_ = get_neuron_coords(layer_idx, j, max_neurons=n_neurons[0])
                        print("start: " + str(start_) + ", end: " + str(end_))
                        ll = Line(start=start_, end=end_, color=YELLOW, stroke_width=1).set_z_index(-1)
                        layer_connections.append(ll)
                connections.append(layer_connections)
        animations = []
        for layer in layers:
            anims = []
            for neuron in layer:
                anims.append(Create(neuron))
            animations.append(anims)

        connect_animations = []
        for con_layer in connections:
            anims = []
            for con in con_layer:
                anims.append(Write(con))
            connect_animations.append(anims)
        connect_animations.append(None)

        for anims, con_anims in zip(animations, connect_animations):
            self.play(*anims, run_time=layer_anim_time)
            if con_anims is not None:
                self.play(*con_anims, run_time=layer_anim_time)

        # Logistic regression
        scale = 0.3
        logistic_rng = 6
        ax = Axes(
            axis_config={
                "include_ticks": False,
                "color": BLUE
            },
            x_length=5/scale,
            y_length=2.5/scale,
            x_range=[-logistic_rng, logistic_rng, 1],
            y_range=[-0.5, 1.5, 1],
        ).shift(RIGHT*horiz_dist + DOWN*vert_dist + DOWN*0.05)

        def logistic_fcn(x):
            return  1 / (1 + np.exp(-x))

        logistic_curve = ax.plot(logistic_fcn, x_range=[-logistic_rng, logistic_rng])
        logistic_curve.set_stroke_color([RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE])

        v_log = VGroup(ax, logistic_curve)
        v_log.scale(scale)

        # Cat/dog images
        img_scale = 0.4
        img_shift = 2
        cat_img = ImageMobject("images/cat.jpg").scale(img_scale).\
            shift(UP * vert_dist + img_shift * LEFT)
        dog_img = ImageMobject("images/dog.jpg").scale(img_scale).\
            shift(UP * vert_dist + img_shift * RIGHT)
        which_animal = MathTex(r'?', font_size=80, color=BLUE).shift(UP * vert_dist)

        self.play(Create(ax))
        self.play(Create(logistic_curve))
        self.wait(1.5)

        self.play(FadeIn(cat_img), run_time=1)
        self.wait(0.5)
        self.play(FadeIn(dog_img), run_time=1)
        self.wait(0.5)
        self.play(Write(which_animal), run_time=0.5)
        self.wait(0.5)


class Gradients(Scene):
    def construct(self):
        self.wait(1)
        f = MathTex(r'z &= f(x, y)\\',
                    r'- \nabla f(x, y)',
                    r'&= \left(-\frac{\partial f(x, y)}{\partial x}, -\frac{\partial f(x, y)}{\partial y}\right)',
                    font_size=font_size_big, color=YELLOW)
        self.wait(0.5)
        self.play(Write(f[0]), run_time=1.5)
        self.wait(6.5)
        self.play(Write(f[1]), run_time=1.5)
        self.wait(3.5)
        self.play(Write(f[2]), run_time=2)
        self.wait(1.5)


class GradientFormulas(SurfaceAnimations):
    def construct(self):
        left_graph_scaling_factor = 1.5
        x = 4
        y = 2
        xx = -12
        yy = -5
        h = 0.3
        l_shift = 4
        r_time = 1.7
        axes_reveal_t = 0.01
        graph_shift = 12
        arrow_length = 2
        init_point_r = 0.1

        self.set_camera_orientation(zoom=0.3)
        vg_1D = VGroup()

        label = Tex(r"$f(x, y)$ ", "$=$", "$(x^2 + y^2)/10$", font_size=l_font_size)
        coords = Tex(r"$x $", "$=$", "$-2, y = -3$", font_size=l_font_size).next_to(label, DOWN)
        coords.get_part_by_tex('$=$').align_to(label.get_part_by_tex('$=$'), LEFT)
        grad_eq = Tex(r"$-\nabla f(x, y)$", "$=$",
                      r"$\left[ -\frac{\partial f(x, y)}{\partial x}, -\frac{\partial f(x, y)}{\partial y} \right]$",
                      font_size=l_font_size).next_to(coords, DOWN)
        grad_eq.get_part_by_tex('$=$').align_to(coords.get_part_by_tex('$=$'), LEFT)
        derivatives = Tex("$=$", r"$\left[ \frac{x}{5}, \frac{y}{5} \right]$", font_size=l_font_size).next_to(grad_eq,
                                                                                                              DOWN)
        derivatives.get_part_by_tex('$=$').align_to(grad_eq.get_part_by_tex('$=$'), LEFT)
        eq_plugged = Tex("$=$", r"$\left[ \frac{-2}{5}, \frac{-3}{5} \right]$", font_size=l_font_size).next_to(
            derivatives, DOWN)
        eq_plugged.get_part_by_tex('$=$').align_to(derivatives.get_part_by_tex('$=$'), LEFT)
        eq_final = Tex("$=$", r"$\left[ -0.4, -0.6 \right]$", font_size=l_font_size).next_to(eq_plugged, DOWN)
        eq_final.get_part_by_tex('$=$').align_to(eq_plugged.get_part_by_tex('$=$'), LEFT)

        vg_1D.scale(left_graph_scaling_factor)
        vg_1D.shift(graph_shift * LEFT)

        vg_2D = self.get_all_objects()

        xx = -2
        yy = -3

        # Draw the init point
        init_point_coords = self.axes.c2p(xx, yy, self.f(xx, yy))
        init_point = Sphere(
            center=init_point_coords,
            radius=0.2,
        ).set_color(YELLOW).set_opacity(1)
        vg_2D.add(init_point)

        vg_2D.shift(graph_shift * RIGHT)

        # Function
        group1 = [Create(self.axes, run_time=0.3),
                  Create(self.surf, run_time=0.5)]
        anim_2_3 = AnimationGroup(*group1, lag_ratio=1)
        group2 = [Write(label, run_time=2), anim_2_3]
        self.play(AnimationGroup(*group2, lag_ratio=0.3))
        self.wait(1)

        # Point
        group = [Write(coords, run_time=1.4), Create(init_point, run_time=0.5)]
        self.play(AnimationGroup(*group, lag_ratio=0.8))

        # # Equations
        self.play(Write(grad_eq), run_time=1.5)
        self.wait(1)
        self.play(Write(derivatives), run_time=1.5)
        self.wait(1)
        self.play(Write(eq_plugged), run_time=1.5)
        self.wait(1)
        self.play(Write(eq_final), run_time=1.5)

        # Draw the grad vector.
        start_ = self.axes.c2p(xx, yy, self.f(xx, yy))
        grad = [self.df_du(xx, yy), self.df_dv(xx, yy)]
        grad_dir = grad / np.linalg.norm(grad)
        descent_dir = np.array([grad_dir[0], grad_dir[1], np.linalg.norm(grad)])
        descent_dir = descent_dir / np.linalg.norm(descent_dir)
        end_ = self.axes.c2p(*list(np.array([xx, yy, self.f(xx, yy)]) - 3 * descent_dir))
        arrow = Line(start=start_, end=end_,
                     color=RED, buff=0.05).add_tip(
            tip_shape=ArrowTriangleFilledTip)
        self.play(Create(arrow))


class MultipleGraphs(MovingCameraScene):
    def construct(self):
        x = 2.5
        y = 7
        d = 20
        v = 15
        h = 0.3
        l_shift = 5
        l_font_size = 144
        r_time = 1.7
        axes_reveal_t = 0.01
        t_between_graphs = 0.3
        opt_point_size = 0.3
        time_before_reveal_graph = 0.8

        self.play(
            self.camera.frame.animate.set_width(self.camera.frame.get_width() * 4), run_time=0.001
        )

        axes = Axes(
            x_range=[-x, x],
            y_range=[-y * h, y],
            axis_config={"color": BLUE},
        )

        # Create Graph
        graph = axes.plot(lambda x: (x - 1) ** 2, color=WHITE)
        dummy_graph = axes.plot(lambda x: np.sin(x), color=WHITE)
        dummy_graph_opt_point = axes.plot(lambda x: np.sin(x - 1), x_range=[0, 2, 0.01], color=WHITE)
        coord_labels = axes.get_axis_labels(x_label="x", y_label="y")
        label = Tex(r"$y = (x-1)^2$", font_size=l_font_size)
        opt_point = Circle(color=YELLOW, fill_opacity=1, radius=opt_point_size).shift(RIGHT).move_to(
            dummy_graph_opt_point)

        axes2 = Axes(
            x_range=[-x, x],
            y_range=[-y * h, y],
            axis_config={"color": BLUE},
        )

        # Create Graph
        def f2(x):
            return x ** 2 + math.sin(4 * x)

        res = minimize_scalar(f2)
        opt_f2 = res.x

        # Create Graph
        graph2 = axes2.plot(f2, color=WHITE)
        dummy_graph2 = axes2.plot(lambda x: np.sin(x), color=WHITE)
        dummy_graph_opt_point2 = axes2.plot(lambda x: np.sin(x - opt_f2) + f2(opt_f2),
                                            x_range=[opt_f2 - 1, opt_f2 + 1, 0.01], color=WHITE)
        coord_labels2 = axes2.get_axis_labels(x_label="x", y_label="y")
        label2 = Tex(r"$y = x^2 + 2\sin(4x)$", font_size=l_font_size)
        opt_point2 = Circle(color=YELLOW, fill_opacity=1, radius=opt_point_size).move_to(
            dummy_graph_opt_point2)

        axes3 = Axes(
            x_range=[-x, x],
            y_range=[-y * h, y],
            axis_config={"color": BLUE},
        )

        def f3(x):
            return (x - 0.7) ** 4 / 2 + (x - 0.7) ** 3 + x

        res = minimize_scalar(f3)
        opt_f3 = res.x

        # Create Graph
        graph3 = axes3.plot(f3, color=WHITE, x_range=[-2, 2, 0.002])
        dummy_graph3 = axes.plot(lambda x: np.sin(x), color=WHITE)
        dummy_graph_opt_point3 = axes3.plot(lambda x: np.sin(x - opt_f3) + f3(opt_f3),
                                            x_range=[opt_f3 - 1, opt_f3 + 1, 0.01], color=WHITE)
        coord_labels3 = axes.get_axis_labels(x_label="x", y_label="y")
        label3 = Tex(r"$y = (x-0.7)^4/2 + (x-0.7)^3 + x$", font_size=l_font_size)
        opt_point3 = Circle(color=YELLOW, fill_opacity=1, radius=opt_point_size).move_to(
            dummy_graph_opt_point3)

        axes4 = Axes(
            x_range=[-x, x],
            y_range=[-y * h, y],
            axis_config={"color": BLUE},
        )

        def f4(x):
            return np.exp(x) + np.exp(1 - x) - 2

        res = minimize_scalar(f4)
        opt_f4 = res.x

        # Create Graph
        graph4 = axes4.plot(f4, color=WHITE, x_range=[-1.1, 2, 0.002])
        dummy_graph4 = axes.plot(lambda x: np.sin(x), color=WHITE)
        dummy_graph_opt_point4 = axes4.plot(lambda x: np.sin(x - opt_f4) + f4(opt_f4),
                                            x_range=[opt_f4 - 1, opt_f4 + 1, 0.01], color=WHITE)
        coord_labels4 = axes.get_axis_labels(x_label="x", y_label="y")
        label4 = Tex(r"$y = e^x + e^{1-x} - 2$", font_size=l_font_size)
        opt_point4 = Circle(color=YELLOW, fill_opacity=1, radius=opt_point_size).move_to(
            dummy_graph_opt_point4)

        # Relative positioning
        vec = d * LEFT / 2 + v * UP / 2
        graph.shift(vec)
        axes.shift(vec)
        coord_labels.shift(vec)
        label.shift(vec + UP * l_shift)
        opt_point.shift(vec)

        vec = d * RIGHT / 2 + v * UP / 2
        graph2.shift(vec)
        axes2.shift(vec)
        coord_labels2.shift(vec)
        label2.shift(vec + UP * l_shift)
        opt_point2.shift(vec)

        vec = d * LEFT / 2 + v * DOWN / 2
        graph3.shift(vec)
        axes3.shift(vec)
        coord_labels3.shift(vec)
        label3.shift(vec + UP * l_shift)
        opt_point3.shift(vec)

        vec = d * RIGHT / 2 + v * DOWN / 2
        graph4.shift(vec)
        axes4.shift(vec)
        coord_labels4.shift(vec)
        label4.shift(vec + UP * l_shift)
        opt_point4.shift(vec)

        # Display graphs
        self.add(label)
        self.wait(time_before_reveal_graph)
        self.play(Create(axes), run_time=axes_reveal_t)
        self.add(coord_labels)
        self.play(Create(graph), run_time=r_time)
        self.play(Create(opt_point))
        self.wait(t_between_graphs)

        self.add(label2)
        self.wait(time_before_reveal_graph)
        self.play(Create(axes2), run_time=axes_reveal_t)
        self.add(coord_labels2)
        self.play(Create(graph2), run_time=r_time)
        self.play(Create(opt_point2))
        self.wait(t_between_graphs)

        self.add(label3)
        self.wait(time_before_reveal_graph)
        self.play(Create(axes3), run_time=axes_reveal_t)
        self.add(coord_labels3)
        self.play(Create(graph3), run_time=r_time)
        self.play(Create(opt_point3))
        self.wait(t_between_graphs)

        self.add(label4)
        self.wait(time_before_reveal_graph)
        self.play(Create(axes4), run_time=axes_reveal_t)
        self.add(coord_labels4)
        self.play(Create(graph4), run_time=r_time)
        self.play(Create(opt_point4))
        self.wait(t_between_graphs)


class PartialDerivatives(SurfaceAnimations):

    def __init__(self):
        super().__init__()
        self.axes_shift = 4
        self.surf_colors = [PURPLE, WHITE]
        self.axes_color = BLUE
        self.axis_config = {"color": BLUE,
                            "include_ticks": False}

    def construct(self):
        axes_shift = 1.1
        font_size = 120
        graph_shift = 12
        y_const = 0
        res = 9
        falling_lag_ratio = 0.2

        self.set_camera_orientation(zoom=0.25)

        # 2D
        vg_2D = self.get_all_objects(get_surf=False)
        vg_2D.shift(graph_shift * LEFT)

        self.axes.shift(DOWN * self.axes_shift + RIGHT*8).rotate(angle=PI / 4, axis=UP)

        # Axis lables
        axis_label_color = YELLOW
        x_label = MathTex(r'x', font_size=axis_label_font_size, color=axis_label_color).\
            move_to(np.array([-3.5, -4.1, 0]))
        y_label = MathTex(r'y', font_size=axis_label_font_size, color=axis_label_color).\
            move_to(np.array([-7.5, -11, 0]))
        z_label = MathTex(r'f(x, y)', font_size=axis_label_font_size, color=axis_label_color). \
            move_to(np.array([-8, 1.5, 0]))
        labels = VGroup(x_label, y_label, z_label)
        shift_down = DOWN * 2
        labels.shift(shift_down)
        labels.shift(axes_shift * UP)
        self.axes.shift(shift_down)
        self.axes.shift(axes_shift * UP)

        self.play(Create(self.axes), Write(x_label), Write(y_label), Write(z_label), run_time=0.3)
        self.wait(0.1)

        dist = 12
        vert_dist = 2
        f_formula = MathTex(r'f(x, y)', r'&=', r'\frac{x^2 + y^2}{10} \\',
                            r'\frac{\partial f}{\partial x}', r'&=', r'\frac{\Delta f}{\Delta x}',
                            font_size=font_size_partial_der).shift(10 * UP)
        f_y_0 = MathTex(r'f(x)', r'&=', r'\frac{x^2}{10} \\',
                            r'\frac{\partial f}{\partial x}', r'&=', r'\frac{x}{5}',
                            font_size=font_size_partial_der)

        diff = f_formula[1].get_center() - f_y_0[1].get_center()
        f_y_0.shift(diff)
        old_center = f_formula[1].get_center()

        # Surface
        self.surf = self.get_surf(slice_at=y_const, res=res)
        opacity_ = 1
        create_surf_suntime = 1.5
        if isinstance(self.surf, list):
            self.play(Create(self.surf[0]), Create(self.surf[1]), Create(f_formula[0]), Create(f_formula[1]),
                      Create(f_formula[2]), run_time=create_surf_suntime)
            self.play(self.surf[0].animate.set_opacity(opacity_), self.surf[1].animate.set_opacity(opacity_))
        else:
            self.play(Create(self.surf), Create(f_formula[0]), Create(f_formula[1]),
                      Create(f_formula[2]), run_time=create_surf_suntime)
            self.play(self.surf.animate.set_opacity(opacity_))

        self.wait(0.25)
        y_1 = f_formula[0][4:5]
        y_2 = f_formula[2][3:4]
        r_height = 1.4
        r_width = 1.1
        s_1 = Rectangle(height=r_height, width=r_width, stroke_color=YELLOW, stroke_width=3, stroke_opacity=1,
                        fill_opacity=0). \
            move_to(y_1.get_center())
        s_2 = Rectangle(height=r_height, width=r_width, stroke_color=YELLOW, stroke_width=3, stroke_opacity=1,
                        fill_opacity=0). \
            move_to(y_2.get_center())
        self.play(Create(s_1), Create(s_2), run_time=1)
        zero_1 = MathTex(r'0', font_size=font_size_partial_der).move_to(y_1.get_center())
        zero_2 = MathTex(r'0', font_size=font_size_partial_der).move_to(y_2.get_center())
        self.play(FadeIn(zero_1), FadeOut(y_1), FadeIn(zero_2), FadeOut(y_2), run_time=0.6)
        self.play(Uncreate(s_1), Uncreate(s_2), run_time=0.25)
        self.play(AnimationGroup(AnimationGroup(Write(f_y_0[0]), Write(f_y_0[1]), Write(f_y_0[2]), run_time=0.7),
                                 AnimationGroup(Unwrite(f_formula[0]), Unwrite(f_formula[1]), Unwrite(f_formula[2]),
                                 FadeOut(zero_1), FadeOut(zero_2), run_time=0.05)), lag_ratio=0)
        self.wait(0.05)

        # Slice with a plane const y
        def func_const_y(u, v):
            return np.array([u, v, y_const])

        plane_slide_range = 6.5 * PI
        plane_movement_factor = 4
        e_max = plane_movement_factor * plane_slide_range
        e_min = -plane_slide_range * 0.6
        e = ValueTracker(e_min)
        range = 2 * PI

        def trunc_arg(a):
            return min([a, self.quadratic_range * 1.3])

        def get_plane_and_half_surf():
            vg_ = VGroup()
            u_range = [-range + e.get_value(), range + e.get_value()]
            opacity_current = max(0, 1 - max(0, e.get_value() / 30))
            plane_ = Surface(
                lambda u, v: self.axes.c2p(*func_const_y(u, v)),
                u_range=u_range,
                v_range=[-range, range],
                resolution=res).set_opacity(opacity_current)
            vg_.add(plane_)
            self.cached["plane"] = plane_
            if e_max * falling_lag_ratio * 0.95 < e.get_value():
                try:
                    self.remove(self.cached["surf_to_remove"])
                except:
                    pass
            else:
                half_surf = Surface(
                    lambda u, v: self.axes.
                        c2p(self.normalize_range(u, v)[0], self.f(u, v), self.normalize_range(u, v)[1]),
                    u_range=[-self.quadratic_range, self.quadratic_range], v_range=[0, self.quadratic_range],
                    checkerboard_colors=[self.surf_colors[0], self.surf_colors[1]],
                    resolution=(res, res)).set_opacity(opacity_)
                vg_.add(half_surf)
                self.cached["surf_to_remove"] = half_surf
            return vg_

        plane_const_y = always_redraw(
            lambda: get_plane_and_half_surf()
        )
        self.add(plane_const_y)

        # Intersection curve
        def foo(t):
            return self.axes.c2p(t, self.f(t, y_const), y_const + 0.1)

        curve_arg_range = math.sqrt(self.quadratic_range ** 2 - y_const ** 2)

        def opacity_tracker():
            h = 0.1
            alpha_ = 0.5/h/e_max
            beta_ = - alpha_ * e_max * falling_lag_ratio
            val_ = alpha_ * e.get_value() + beta_
            ret = min(0.5, max(0, val_))
            print("opacity: " + str(ret))
            return ret

        def get_curve3D():
            curve_ = ParametricFunction(foo, t_range=np.array(
                [-curve_arg_range, max([-curve_arg_range, min([curve_arg_range, e.get_value() + range])])]),
                               fill_opacity=0, stroke_opacity=0.5 + opacity_tracker(),
                               stroke_width=3).set_color(RED)
            return curve_

        curve = always_redraw(lambda: get_curve3D())
        self.add(curve)

        # Falling slice
        e_fall = ValueTracker(0)
        surf = self.surf[1]
        init_point = surf.get_center()
        g = 0.1

        def falling_surface_updater(m):
            m.move_to(init_point + np.array([0, - g * e_fall.get_value() ** 2, 0]))
            if e.get_value() > e_max * falling_lag_ratio:
                ax_start = self.axes.c2p(0, 0, 0)
                ax_end = self.axes.c2p(1, 0, 0)
                ax_ = ax_end - ax_start
                ax_ = ax_ / np.linalg.norm(ax_)
                m.rotate(angle=0.05, axis=ax_, about_point=ax_start)
            return m

        surf.add_updater(lambda m: falling_surface_updater(m))

        self.play(AnimationGroup(e.animate.set_value(e_max), e_fall.animate.set_value(10 * PI),
                                 lag_ratio=falling_lag_ratio, rate_func=linear),
                  run_time=3.5, rate_func=linear)

        # 2D curve
        axes2D = Axes(
            x_range=[-self.quadratic_range, self.quadratic_range, 1],
            y_range=[-self.quadratic_range, self.quadratic_range, 1],
            tips=True,
            axis_config={"color": BLUE,
                         "include_ticks": False}
        ).shift(graph_shift * RIGHT).shift(DOWN * self.axes_shift).scale(1.5)
        axes2D.shift(axes_shift * UP)

        # Axis labels
        axis_label_color = YELLOW
        font_2D = math.floor(axis_label_font_size * 0.8)
        x_label_2D = MathTex(r'x', font_size=font_2D, color=axis_label_color).\
            move_to(np.array([21.5, -1.2, 0]))
        y_label_2D = MathTex(r'f(x)', font_size=font_2D, color=axis_label_color). \
            move_to(np.array([14, 2, 0]))
        labels_2D = VGroup(x_label_2D, y_label_2D)
        shift_down = DOWN * 2
        labels_2D.shift(shift_down)
        labels_2D.shift(axes_shift * UP)
        self.play(Create(axes2D), Write(x_label_2D), Write(y_label_2D), run_time=0.3)

        def f(x):
            return self.f(x, y_const)

        curve_2D = axes2D.plot(f, x_range=[-curve_arg_range, curve_arg_range]).set_color(RED)  # .scale(graph_scale)
        curve_2D.shift(IN * 0.1)
        curve_2D.set_shade_in_3d(True)
        self.play(Create(curve_2D), run_time=1)

        # Changing argument
        r = 0.3
        e_point = ValueTracker(-curve_arg_range)
        point1 = Sphere(radius=r).set_color(YELLOW). \
            move_to(self.axes.c2p(-curve_arg_range, self.f(-curve_arg_range, y_const), y_const))
        point1.add_updater(lambda m:
                           m.move_to(self.axes.c2p(e_point.get_value(),
                                                             self.f(e_point.get_value(), y_const), y_const)))
        point2 = Sphere(radius=r).set_color(YELLOW). \
            move_to(axes2D.c2p(-curve_arg_range, self.f(-curve_arg_range, y_const)))
        point2.add_updater(lambda m:
                           m.move_to(axes2D.c2p(e_point.get_value(),
                                                self.f(e_point.get_value(), y_const))))

        def get_val():
            N = 40
            return math.floor(e_point.get_value() * N) / N

        def get_label():
            font_size = 125
            formula = MathTex(r'f(' + str('{:2.2f}'.format(get_val())) + r', ' + str(y_const) + r') = ' +
                          str('{:2.3f}'.format(self.f(get_val(), y_const))), font_size=font_size)
            return formula.next_to(self.axes, 3 * RIGHT)

        formula = always_redraw(lambda: get_label())
        self.add(formula)
        curve.set_shade_in_3d(True)
        # self.wait(1)
        self.play(Create(point1), Create(point2), run_time=0.7)

        self.play(
            e_point.animate.set_value(curve_arg_range),
            run_time=3.5,
            rate_func=linear,
        )

        self.play(Uncreate(point1), Uncreate(point2), Unwrite(formula), run_time=0.5)

        # Differentials
        x = -5
        font_size_small = 85
        slope_arrow_length = 3.5
        mid_point_3D = self.axes.c2p(x, self.f(x, y_const), y_const)
        mid_point_2D = axes2D.c2p(x, self.f(x, y_const), y_const)
        vec = np.array([1, self.df_du(x, y_const), 0])
        vec = vec / np.linalg.norm(vec)
        end_3D = self.axes.c2p(*(np.array([x, self.f(x, y_const), y_const]) + vec * slope_arrow_length))
        end_2D = axes2D.c2p(*(np.array([x, self.f(x, y_const), y_const]) + vec * slope_arrow_length))
        start_3D = mid_point_3D - (end_3D - mid_point_3D)
        start_2D = mid_point_2D - (end_2D - mid_point_2D)

        point_3D = Sphere(radius=r).set_color(YELLOW). \
            move_to(mid_point_3D)
        point_2D = Sphere(radius=r).set_color(YELLOW). \
            move_to(mid_point_2D)

        slope_arrow_3D = Line(start=start_3D, end=end_3D, color=YELLOW, buff=0)
        slope_arrow_2D = Line(start=start_2D, end=end_2D, color=YELLOW, buff=0)
        self.wait(1)
        self.play(Create(point_3D), Create(point_2D), run_time=0.9)
        self.wait(0.4)
        self.play(Create(slope_arrow_3D), Create(slope_arrow_2D), run_time=1.2)
        self.wait(0.4)

        # Slope angle
        a = mid_point_3D
        b = self.axes.c2p(-10, 0, 0)
        o = end_3D
        v_a = (a - o) / np.linalg.norm(a - o)
        v_b = (b - o) / np.linalg.norm(b - o)
        t_min = 0
        t_max = np.arccos(np.abs(np.dot(v_a, v_b))) * 1.25
        k = np.cross(v_a, v_b)

        t__ = ValueTracker(t_min)

        def point_on_curve(theta):
            v = v_a
            part_1 = v * math.cos(theta)
            part_2 = np.cross(k, v) * math.sin(theta)
            part_3 = k * np.dot(k, v) * (1 - math.cos(theta))
            result = (part_1 + part_2 + part_3) * np.linalg.norm(a - o) / 1.8 + o
            return result

        slope_curve = always_redraw(lambda: ParametricFunction(point_on_curve,
                                                               t_range=np.array([0, t__.get_value()]),
                                                               fill_opacity=0).set_color(GREEN)
                                    )
        self.add(slope_curve)

        a_2D = np.array([x, self.f(x, y_const)])
        b_2D = np.array([-10, 0])
        o_2D = np.array([-2.5, 0])
        v_a_2D = (a_2D - o_2D) / np.linalg.norm(a_2D - o_2D)
        v_b_2D = (b_2D - o_2D) / np.linalg.norm(b_2D - o_2D)
        t_min_2D = 0
        t_max_2D = np.arccos(np.abs(np.dot(v_a_2D, v_b_2D)))
        t___2D = ValueTracker(t_min_2D)

        def point_on_curve_2D(theta):
            print("theta: " + str(theta))
            M = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            v_a_rotated = np.dot(M, v_a_2D)
            result = v_a_rotated * np.linalg.norm(a_2D - o_2D) + o_2D
            print("point: " + str(result))
            offset = axes2D.c2p(*o_2D)
            return np.array([result[0], result[1], 0]) * 0.55 + offset

        slope_curve_2D = always_redraw(lambda: ParametricFunction(point_on_curve_2D,
                                                               t_range=np.array([0, t___2D.get_value()]),
                                                               fill_opacity=0).set_color(GREEN)
                                    )
        self.add(slope_curve_2D)

        self.wait(1.5)
        self.play(t__.animate.set_value(t_max), t___2D.animate.set_value(t_max_2D), run_time=1.2,
                  rate_func=linear)

        alpha_label = MathTex(r'\alpha', font_size=130).set_color(GREEN).\
            move_to(end_3D + np.array([-4.1, 0.15, 0]))
        alpha_label_2D = MathTex(r'\alpha', font_size=130).set_color(GREEN).\
            move_to(end_2D + np.array([-4.1, 0.78, 0]))
        self.play(Write(alpha_label), Write(alpha_label_2D))

        self.wait(1)
        self.play(Uncreate(slope_arrow_3D), Uncreate(slope_arrow_2D), Uncreate(point_2D), Uncreate(point_3D),
                  Uncreate(slope_curve_2D), Uncreate(slope_curve), Uncreate(slope_arrow_3D), Uncreate(alpha_label),
                  Uncreate(alpha_label_2D))

        x = -6
        d_x = 3.7
        diff_color = YELLOW

        def get_delta_x(ax, x_, dx_):
            print("get_delta_x: " + "x_tracker: " + str(x_tracker.get_value()) + "x_: " + str(x_))
            return Line(start=ax.c2p(x_, self.f(x_, y_const), y_const),
                        end=ax.c2p(x_ + dx_, self.f(x_, y_const), y_const),
                        buff=0).set_color(diff_color)

        dx_tracker = ValueTracker(d_x)
        x_tracker = ValueTracker(x)
        delta_x_3D = always_redraw(lambda: get_delta_x(self.axes, x_tracker.get_value(), dx_tracker.get_value()))
        delta_x_2D = always_redraw(lambda: get_delta_x(axes2D, x_tracker.get_value(), dx_tracker.get_value()))
        h = 0.5
        font_scaler = 1.15

        def get_font(x_val, dx_val, dim):
            return math.floor(font_scaler * font_size_small)

        def get_x_label_shift():
            xxx = x_tracker.get_value()
            if xxx > 0:
                print("-1")
                return -1
            else:
                print("0")
                return 0

        delta_label_color = ORANGE
        label_stroke = 1.5
        delta_x_label_3D = always_redraw(lambda: Tex(r"$\Delta x$", font_size=get_font(x_tracker.get_value(),
                                        dx_tracker.get_value(), 3), stroke_width=label_stroke).next_to(delta_x_3D,
                                        UP * (h + get_x_label_shift())).set_color(delta_label_color))
        delta_x_label_2D = always_redraw(lambda: Tex(r"$\Delta x$", font_size=get_font(x_tracker.get_value(),
                                        dx_tracker.get_value(), 2), stroke_width=label_stroke).next_to(delta_x_2D,
                                        UP * (h + get_x_label_shift())).set_color(delta_label_color))
        self.play(Create(delta_x_3D), Create(delta_x_2D), Write(delta_x_label_3D), Write(delta_x_label_2D))
        self.wait(2)

        def get_delta_y(ax, x_, dx_):
            return Line(start=ax.c2p(x_ + dx_, self.f(x_, y_const), y_const),
                        end=ax.c2p(x_ + dx_, self.f(x_ + dx_, y_const), y_const),
                        buff=0).set_color(diff_color)

        delta_y_3D = always_redraw(lambda: get_delta_y(self.axes, x_tracker.get_value(), dx_tracker.get_value()))
        delta_y_2D = always_redraw(lambda: get_delta_y(axes2D, x_tracker.get_value(), dx_tracker.get_value()))
        l = 0.7
        delta_y_label_3D = always_redraw(lambda: Tex(r"$\Delta f$", font_size=get_font(x_tracker.get_value(),
                                        dx_tracker.get_value(), 3), stroke_width=label_stroke).next_to(delta_y_3D, RIGHT * l)
                                         .set_color(delta_label_color))
        delta_y_label_2D = always_redraw(lambda: Tex(r"$\Delta f$", font_size=get_font(x_tracker.get_value(),
                                        dx_tracker.get_value(), 3), stroke_width=label_stroke).next_to(delta_y_2D, RIGHT * l)
                                         .set_color(delta_label_color))
        self.wait(1.5)
        self.play(Create(delta_y_3D), Create(delta_y_2D), Write(delta_y_label_3D), Write(delta_y_label_2D), run_time=1.5)
        self.wait(4)

        self.play(dx_tracker.animate.set_value(d_x * 0.38), run_time=3.5, rate_func=linear)
        self.wait(5)
        self.play(x_tracker.animate.set_value(curve_arg_range * 0.76), run_time=5)
        self.play(Uncreate(delta_x_3D), Uncreate(delta_x_2D), Uncreate(delta_y_3D), Uncreate(delta_y_2D),
                  Uncreate(delta_x_label_3D), Uncreate(delta_x_label_2D), Uncreate(delta_y_label_3D),
                  Uncreate(delta_y_label_2D), run_time=0.6)

        self.wait(2.4)
        self.play(Write(f_formula[3]), Write(f_formula[4]), Write(f_y_0[5]), run_time=1)
        self.wait(4.5)
        self.play(Unwrite(f_formula[3]), Unwrite(f_formula[4]), Unwrite(f_y_0[5]),
                  Unwrite(f_y_0[0]), Unwrite(f_y_0[1]), Unwrite(f_y_0[2]), run_time=0.001)
        self.wait(1)
        # Partial derivatives for y
        f_formula = MathTex(r'f(x, y)', r'&=', r'\frac{x^2 + y^2}{10} \\',
                            r'\frac{\partial f}{\partial y}', r'&=', r'\frac{\Delta f}{\Delta y}',
                            font_size=font_size_partial_der).shift(10 * UP)
        f_y_0 = MathTex(r'f(y)', r'&=', r'\frac{y^2}{10} \\',
                        r'\frac{\partial f}{\partial y}', r'&=', r'\frac{y}{5}',
                        font_size=font_size_partial_der)

        diff = old_center - f_formula[1].get_center()
        f_formula.shift(diff)

        self.play(Write(f_formula[0]), Write(f_formula[1]), Write(f_formula[2]), run_time=0.7)

        diff = f_formula[1].get_center() - f_y_0[1].get_center()
        f_y_0.shift(diff)

        y_1 = f_formula[0][2:3]
        y_2 = f_formula[2][0:1]

        s_1 = Rectangle(height=r_height, width=r_width, stroke_color=YELLOW, stroke_width=3, stroke_opacity=1,
                        fill_opacity=0). \
            move_to(y_1.get_center())
        s_2 = Rectangle(height=r_height, width=r_width, stroke_color=YELLOW, stroke_width=3, stroke_opacity=1,
                        fill_opacity=0). \
            move_to(y_2.get_center())
        self.play(Create(s_1), Create(s_2), run_time=1)
        zero_1 = MathTex(r'0', font_size=font_size_partial_der).move_to(y_1.get_center())
        zero_2 = MathTex(r'0', font_size=font_size_partial_der).move_to(y_2.get_center())
        self.play(FadeIn(zero_1), FadeOut(y_1), FadeIn(zero_2), FadeOut(y_2))
        self.play(Uncreate(s_1), Uncreate(s_2), run_time=0.5)
        self.play(AnimationGroup(AnimationGroup(Write(f_y_0[0]), Write(f_y_0[2]), run_time=1),
                                 AnimationGroup(Unwrite(f_formula[0]), Unwrite(f_formula[2]),
                                                FadeOut(zero_1), FadeOut(zero_2), run_time=0.05)), lag_ratio=0)
        self.wait(1)
        self.play(Write(f_y_0[3]), Write(f_y_0[4]), Write(f_y_0[5]))
        self.wait(1)


class BasicDerivatives(Scene):
    def construct(self):
        formula_color = YELLOW
        const_color = WHITE
        f = MathTex(r'f(', r'x', r', ', r'y', r') &=',  r'x', r'^2', r'y', r' + 2', r'y', r'^2 \\',
                            r'\frac{\partial f}{\partial x} &=', r'2xy \\',
                            r'\frac{\partial f}{\partial y} &=', r'x^2 + 4y',
                            font_size=font_size_big, color=formula_color)
        n_eq_f = 11
        eqs = [Write(f[i]) for i in range(n_eq_f)]
        self.wait(5)
        self.play(*eqs, run_time=2)
        self.wait(0.5)

        # Diff x
        self.play(Write(f[n_eq_f]), run_time=1)
        self.wait(0.6)
        y_idx = [3, 7, 9]
        y_eq = [f[i].animate.set_color(const_color) for i in y_idx]
        self.play(*y_eq, run_time=2)
        self.wait(2)
        y_eq_reverse = [f[i].animate.set_color(formula_color) for i in y_idx]
        self.play(Write(f[n_eq_f + 1]), run_time=2)
        self.wait(3)
        self.play(*y_eq_reverse, run_time=0.3)
        self.wait(1)

        x_idx = [1, 5]
        x_eq = [f[i].animate.set_color(const_color) for i in x_idx]
        x_eq += [Write(f[n_eq_f + 2])]
        self.wait(0.7)
        self.play(*x_eq, run_time=1)
        self.wait(3)
        x_eq_reverse = [f[i].animate.set_color(formula_color) for i in x_idx]
        self.play(Write(f[n_eq_f + 3]), run_time=1)
        self.wait(2.2)
        self.play(*x_eq_reverse, run_time=0.5)
        self.wait(0.1)


class Execution(ThreeDScene):
    def __init__(self):
        super().__init__()
        self.st = 1
        self.ll = 8
        self.axes_shift = 2.2
        self.axes_counter_shift = 1
        self.x = 0.2
        self.y = 0.3
        self.r = 0.18
        self.c = 1.1
        self.dist_formula_graph = 7
        self.range = 1.2
        self.coord_range = 2
        self.height = 1.1
        self.horizontal_shift = 3
        self.font_size = 55
        self.step_size = 1
        self.formulas_color = YELLOW
        self.numbers_color = RED
        self.highlight_color = GREEN
        self.axes = None
        self.surf = None
        self.steps = None
        self.grads = None
        self.points = None
        self.x_y_dist = None
        self.truncate_number_count = 0
        self.f_0_position = None
        self.n_steps_to_play = 5

    def extra_rotation(self, m):
        return m

    def adjust_labels(self, label_x, label_y, label_z):
        pass

    def f(self, x, y):
        return self.c * (x ** 2 + y ** 2)

    def df_du(self, x, y):
        return 2 * self.c * x

    def df_dv(self, x, y):
        return 2 * self.c * y

    def get_axes(self):
        ax = ThreeDAxes(x_range=[-self.coord_range, self.coord_range, self.st], x_length=self.ll,
                        y_range=[-self.coord_range * self.height, self.coord_range * self.height, self.st],
                        y_length=self.ll*self.height,
                        z_range=[-self.coord_range, self.coord_range, self.st], z_length=self.ll)
        if hasattr(self, 'graph_scale'):
            ax.scale(self.graph_scale)
        return ax

    def normalize_quadratic(self, u, v):
        vec = np.array([u, v])
        len = np.linalg.norm(vec)
        if len > self.range:
            vec = vec / len * self.range
        u = vec[0]
        v = vec[1]
        return u, v

    def get_surf(self, ax, res=None):
        opacity = 1
        if res is None:
            res = 20
        u_range = [-self.range, self.range]
        v_range = [-self.range, self.range]
        return Surface(
            lambda u, v: ax.
                c2p(self.normalize_quadratic(u, v)[0], self.f(self.normalize_quadratic(u, v)[0],
                                                              self.normalize_quadratic(u, v)[1]),
                    self.normalize_quadratic(u, v)[1]),
            u_range=u_range, v_range=v_range, checkerboard_colors=[GREEN_D, GREEN_E],
            resolution=(res, res)).set_opacity(opacity)

    def grad_descent(self, x, y):
        n_iters = 12
        step_size = self.step_size
        self.steps = []
        self.grads = []
        for iter in range(n_iters):
            dx = self.df_du(x, y)
            dy = self.df_dv(x, y)
            self.steps.append([x, y])
            grad = np.array([dx, dy])
            self.grads.append(grad)
            x = x - step_size * grad[0]
            y = y - step_size * grad[1]
        self.steps.append([x, y])

    def xy_to_3d(self, two_d_vec):
        x = two_d_vec[0]
        y = two_d_vec[1]
        return x, self.f(x, y), y

    def axes_graph_setup(self, angle_y_axis=0, other_scene=None, other_coords=None):
        x = self.x
        y = self.y
        self.grad_descent(x, y)
        self.axes = self.get_axes()
        alpha = 1
        secondary_rotation_axis = alpha * RIGHT + (1 - alpha) * OUT
        if other_coords is None:
            self.axes.shift(DOWN * self.axes_shift + LEFT * self.horizontal_shift).rotate(angle=PI / 3, axis=UP) \
                .rotate(angle=angle_y_axis, axis=UP).rotate(angle=PI/36, axis=secondary_rotation_axis) \
                .shift(0.7 * UP)
            self.axes = self.extra_rotation(self.axes)
            self.axes.shift(1.1 * DOWN)
        else:
            self.axes.rotate(axis=IN, angle=-PI/150)
            self.axes.move_to(other_coords)

        axis_label_color = BLUE
        lab_font_size = 46
        x_label = MathTex(r'x', font_size=lab_font_size, color=axis_label_color).\
            move_to(np.array([0.3, -2.6, 0]))
        y_label = MathTex(r'y', font_size=lab_font_size, color=axis_label_color).\
            move_to(np.array([-1, -1.5, 0]))
        z_label = always_redraw(lambda: self.get_z_label(font_size=lab_font_size, color=axis_label_color))
        print("z_label initial: " + str(z_label.get_center()))
        self.adjust_labels(x_label, y_label, z_label)
        labels = VGroup(x_label, y_label, z_label)

        self.surf = self.get_surf(self.axes)

        formula_f = MathTex(r'f(x, y) &= ' + str(self.c) + r'(x^2 + y^2)', font_size=self.font_size)\
            .set_color(self.formulas_color). \
            next_to(self.axes, UP * self.dist_formula_graph).shift(self.axes_counter_shift * DOWN)
        if other_scene is None:
            self.play(Create(self.axes), Create(self.surf), Write(formula_f), Write(x_label), Write(y_label),
                      Write(z_label))
        else:
            other_scene.play(Create(self.axes), Create(self.surf))
        return formula_f

    def get_init_point_helper(self):
        i = 0
        vec = self.steps[i]
        u = vec[0]
        v = vec[1]
        return self.axes.c2p(*self.invert_x_y(np.array([u, self.f(u, v), v])))

    def get_init_point(self):
        print("get_init_point 2")
        prev_point = self.get_init_point_helper()
        return Sphere(radius=self.r).set_color(YELLOW).move_to(prev_point)

    def get_updates(self):
        return MathTex(r'x &:= x - \frac{\partial f}{\partial x}\\',
                          r'y &:= y - \frac{\partial f}{\partial y}',
                          font_size=self.font_size).set_color(self.formulas_color)

    def invert_x_y(self, vec):
        return np.array([vec[2], vec[1], vec[0]])

    def get_steps_for_plotting(self, n_steps):
        steps = VGroup()
        for idx in range(n_steps):
            prev_point = self.steps[idx]
            print("x: " + str(prev_point[0]) + ", y: " + str(prev_point[1]))
            currect_point = self.steps[idx+1]
            color = RED if (np.array(currect_point) - np.array(prev_point))[0] > 0 else BLUE
            new_step = Line(start=self.axes.c2p(*self.invert_x_y(self.xy_to_3d(prev_point))),
                            end=self.axes.c2p(*self.invert_x_y(self.xy_to_3d(currect_point))), color=color, buff=0.05).\
                add_tip(tip_shape=ArrowTriangleFilledTip, tip_length=0.2)
            steps.add(new_step)
        return steps

    def set_color(self, m):
        m[0].set_color(self.formulas_color)
        m[1].set_color(self.numbers_color)
        m[2].set_color(self.formulas_color)
        m[3].set_color(self.numbers_color)
        m[4].set_color(self.formulas_color)
        m[5].set_color(self.formulas_color)
        m[6].set_color(self.numbers_color)
        m[7].set_color(self.formulas_color)
        m[8].set_color(self.formulas_color)
        m[9].set_color(self.numbers_color)
        return m

    def position_xy(self, m):
        current_dist = m[2].get_left()[0] - m[0].get_left()[0]
        coords = m[2].get_center()
        m[2].move_to(np.array([coords[0] + self.x_y_dist - current_dist, coords[1], coords[2]]))

        dist = m[1].get_left()[0] - m[0].get_right()[0]
        dist_current = m[3].get_left()[0] - m[2].get_right()[0]
        m[3].shift(RIGHT * (dist - dist_current))
        return m

    def setup_static_stuff(self):
        formula_f = self.axes_graph_setup()

        formulae_init = MathTex(r'x &=',
                           str('{:2.2f}'.format(self.x)),
                           r'\,\,\,\,' + r'y =',
                           str('{:2.2f}'.format(self.y)) + r'\\',
                           r'\frac{\partial f}{\partial x} &= ' + str(2*self.c) + r'x',
                           r'=',
                           str('{:2.2f}'.format(2 * self.c * self.x)) + r'\\',
                           r'\frac{\partial f}{\partial y} &= ' + str(2*self.c) + r'y',
                           r'=',
                           str('{:2.2f}'.format(2 * self.c * self.y)),
                           font_size=self.font_size).shift(RIGHT * self.horizontal_shift)
        align_top(formulae_init, formula_f)
        formulae_init.shift(RIGHT * 0.7)

        self.set_color(formulae_init)
        self.x_y_dist = (formulae_init[2].get_left()[0] - formulae_init[0].get_left()[0]) * 1.1

        self.position_xy(formulae_init)
        return formulae_init, formula_f

    def truncate_number(self, num, precision):
        self.truncate_number_count += 1
        scale = 10**precision
        res = round(num * scale) / scale
        print("trunc count: " + str(self.truncate_number_count) + ", before trunc: " + str(num) + str(", after: ") + str(res))
        return res

    def play_dynamic_stuff(self, updates_initial, formula_f, formulae_init=None):
        e_grad_tracker = ValueTracker(0)

        def get_step(e):
            e_val = e + 0.1
            return self.steps[math.floor(e_val)]

        if formulae_init is not None:
            self.play(FadeOut(formulae_init), run_time=0.01)

        def get_formulae():
            formulae = align_left(self.set_color(align_top(self.position_xy(MathTex(r'x &=',
                                                                                    str('{:2.2f}'.format(
                                                                                        self.truncate_number(get_step(
                                                                                            e_grad_tracker.get_value())[
                                                                                                                 0],
                                                                                                             2))),
                                                                                    r'\,\,\,\,' + r'y =',
                                                                                    str('{:2.2f}'.format(
                                                                                        self.truncate_number(get_step(
                                                                                            e_grad_tracker.get_value())[
                                                                                                                 1],
                                                                                                             2))) + r'\\',
                                                                                    r'\frac{\partial f}{\partial x} &= ' + str(
                                                                                        2 * self.c) + r'x',
                                                                                    r'=',
                                                                                    str('{:2.2f}'.format(
                                                                                        2 * self.c *
                                                                                        self.truncate_number(get_step(
                                                                                            e_grad_tracker.get_value())[
                                                                                                                 0],
                                                                                                             2))) + r'\\',
                                                                                    r'\frac{\partial f}{\partial y} &= ' + str(
                                                                                        2 * self.c) + r'y',
                                                                                    r'=',
                                                                                    str('{:2.2f}'.format(
                                                                                        2 * self.c *
                                                                                        self.truncate_number(get_step(
                                                                                            e_grad_tracker.get_value())[
                                                                                                                 1],
                                                                                                             2))),
                                                                                    font_size=self.font_size)
                                                                            )
                                                           .shift(RIGHT * self.horizontal_shift),
                                                           formula_f)
                                                 ),
                                  updates_initial
                                  )
            if self.f_0_position is not None:
                diff = self.f_0_position - formulae[0].get_center()
                formulae.shift(diff)
            else:
                self.f_0_position = formulae[0].get_center()
            return formulae

        formulae = always_redraw(lambda: get_formulae())
        self.add(formulae)

        # Add arrows.
        n_steps = self.n_steps_to_play
        steps = self.get_steps_for_plotting(n_steps)

        ttt = 8.4 if isinstance(self, ConvergingExecution) else 15
        self.play(AnimationGroup(*[Create(step) for step in steps], lag_ratio=1),
                  e_grad_tracker.animate.set_value(n_steps), run_time=ttt, rate_func=linear)

    def get_z_label(self, font_size, color):
        fff = MathTex(r'f(x, y)', r' = 0.00', font_size=font_size, color=color)
        shift = np.array([-2.05, 1.45, 0]) - fff[0].get_center()
        fff.shift(shift)
        fff[0].set_color(color)
        fff[1].set_color(BLACK)
        return fff

    def construct(self):
        formulae_init, formula_f = self.setup_static_stuff()
        self.wait(4.2)
        self.play(Write(formulae_init[4]), Write(formulae_init[7]), run_time=2)
        interval_t = 0.1
        self.wait(4.5)
        self.play(Write(formulae_init[0]), Write(formulae_init[1]), Write(formulae_init[2]), Write(formulae_init[3]),
                  run_time=1.5)
        self.wait(5.5)
        self.play(Write(formulae_init[5]), Write(formulae_init[6]))
        self.wait(0.3)
        self.play(Write(formulae_init[8]), Write(formulae_init[9]))
        self.wait(5.5)

        updates_initial = self.get_updates().next_to(formulae_init, DOWN * 2)
        align_left(updates_initial, formulae_init)

        self.play(Write(updates_initial[0]))
        self.play(Write(updates_initial[1]))

        self.play_dynamic_stuff(updates_initial, formula_f, formulae_init=formulae_init)


class StepSize(Execution):
    def __init__(self, other_scene=None, other_coords=None):
        super().__init__()
        self.ll = 7
        self.dist_formula_graph = 7.5
        self.plot_f_value = False
        self.f_val = None
        self.prev_f_vals = []
        self.init_point_colors = [BLUE, RED]
        self.init_point_current_color = 0
        self.e_arrow = ValueTracker(0)
        self.times_color_changed = 0
        self.other_scene = other_scene
        self.other_coords = other_coords
        self.x = -1
        self.y = -0.35
        if other_scene is not None:
            factor = 0.9
            self.x = self.x * factor
            self.y = self.y * factor

    def get_init_point(self):
        print("get_init_point 3")
        args = self.get_args(self.e_arrow.get_value())
        f_val = self.f(args[0], args[1])
        delta_ = 1e-4
        print("arg: " + str(self.e_arrow.get_value()))
        if len(self.prev_f_vals) >= 2 and abs(f_val - self.prev_f_vals[-1]) > delta_ \
                and (f_val - self.prev_f_vals[-1]) > delta_ \
                and (self.prev_f_vals[-1] - self.prev_f_vals[-2]) < - delta_\
                and self.e_arrow.get_value() < 0.05:
            print("1 cond satisfied")
            if self.times_color_changed == 1:
                self.init_point_current_color = 1 - self.init_point_current_color
                print("1 cond satisfied, init_point_current_color: " + str(self.init_point_current_color))
            self.times_color_changed += 1
        self.prev_f_vals.append(f_val)
        p = np.array([args[1], f_val, args[0]])
        col = 0 if self.e_arrow.get_value() < 0.5 else 1
        return Sphere(radius=self.r * 0.6).set_color(self.init_point_colors[col]).\
            move_to(self.axes.c2p(*p))

    def adjust_labels(self, label_x, label_y, label_z):
        label_x.shift(np.array([-0.5, 0, 0]))
        label_y.shift(np.array([-0.5, -0.3, 0]))

    def get_z_helper(self, str1, str2, color1, color2, font_size):
        fff = MathTex(str1, str2, font_size=font_size)
        shift = np.array([-2.05, 1, 0]) - fff[0].get_center()
        fff.shift(shift)
        fff[0].set_color(color1)
        fff[1].set_color(color2)
        return fff

    def get_z_label(self, font_size, color):
        str1 = r'f(x, y)'
        args = self.get_args(self.e_arrow.get_value())
        f_val = self.f(args[0], args[1])
        str2 = r' = ' + str("{:2.2f}".format(f_val)) if self.plot_f_value else r' = 0.00'
        color2 = RED if self.plot_f_value else BLACK
        return self.get_z_helper(str1=str1, str2=str2, color1=color, color2=color2, font_size=font_size)

    def get_args(self, t):
        init_ = np.array([self.x, self.y])
        target_ = -init_
        point = t * target_ + (1 - t) * init_
        print("get_args: " + "t = " + str(t) + ", point = " + str(point))
        return point

    def construct(self):
        f_formula = self.axes_graph_setup(other_scene=self.other_scene, other_coords=self.other_coords)
        init_point = always_redraw(lambda: self.get_init_point())
        self.wait(3.5)
        if self.other_scene is None:
            self.play(Create(init_point))
        else:
            self.other_scene.play(Create(init_point))
        target_step = self.steps[1]
        target_point = self.axes.c2p(*self.invert_x_y(np.array([target_step[0], self.f(target_step[0], target_step[1]),
                                                               target_step[1]])))

        def get_arg_dynamic(t):
            init_point_ = init_point if isinstance(init_point, np.ndarray) else init_point.get_center()
            target_point_ = target_point if isinstance(target_point, np.ndarray) else target_point.get_center()
            return t*target_point_ + (1-t)*init_point_

        def get_arrow_dynamic(t, add_pointer=True, buff=0.05):
            init_point_ = init_point if isinstance(init_point, np.ndarray) else init_point.get_center()
            target_point_ = target_point if isinstance(target_point, np.ndarray) else target_point.get_center()
            new_end = t*target_point_ + (1-t)*init_point_
            print("t: " + str(t))
            if t <= 0.5:
                return get_arrow_new(init_point_, new_end, add_pointer=add_pointer, color=BLUE, buff=buff)
            else:
                return get_arrow_new(init_point_, new_end, add_pointer=add_pointer, color=RED, buff=buff)

        def get_point_for_curve(t):
            point = self.get_args(t)
            self.f_val = self.f(point[0], point[1])
            self.plot_f_value = True
            return self.axes.c2p(*self.invert_x_y(np.array([point[0], self.f(point[0], point[1]), point[1]])))

        def get_curve_new(t, color, opacity=1):
            return ParametricFunction(get_point_for_curve, t_range=np.array([0, t]),
                               stroke_opacity=opacity, fill_opacity=0, stroke_width=4.5).set_color(color)

        def draw_full_curve(t):
            if t <= 0.5:
                return get_curve_new(t, BLUE)
            else:
                return get_curve_new(t, RED, opacity=0.7)

        def draw_half_curve(t):
            t = min(t, 0.5)
            return get_curve_new(t, BLUE)

        def draw_opt_point():
            res = 5
            r = 0.1
            opt_point = np.array([0, 0])
            coords = self.axes.c2p(*self.invert_x_y(np.array([opt_point[0], self.f(opt_point[0], opt_point[1]),
                                                             opt_point[1]])))
            return Sphere(
                center=coords,
                radius=r,
                resolution=(res, res),
                u_range=[0.001, PI - 0.001],
                v_range=[0, TAU]
            ).set_color(YELLOW).set_opacity(1)

        half_curve = always_redraw(lambda: draw_half_curve(self.e_arrow.get_value()))
        full_curve = always_redraw(lambda: draw_full_curve(self.e_arrow.get_value()))
        if self.other_scene is None:
            self.add(full_curve, half_curve)
        else:
            self.other_scene.add(full_curve, half_curve)

        if self.other_scene is None:
            self.play(self.e_arrow.animate.set_value(0.999), run_time=10, rate_func=linear)
            self.wait(8.1)
        else:
            self.other_scene.play(self.e_arrow.animate.set_value(0.999), run_time=4.1, rate_func=linear)

        updates = align_top(MathTex(r'x &:= x - ',
                          r'\alpha',
                          r'\frac{\partial f}{\partial x}\\',
                          r'y &:= y - ',
                          r'\alpha',
                          r'\frac{\partial f}{\partial y}',
                          font_size=self.font_size)
                          .next_to(f_formula, RIGHT*11)
                            , f_formula
                            )
        updates[1].set_color(BLUE)
        updates[4].set_color(BLUE)

        true_pos_1 = updates[2].get_center()
        true_pos_2 = updates[5].get_center()
        align_left(updates[2], updates[1])
        align_left(updates[5], updates[4])

        if self.other_scene is None:
            self.play(Write(updates[0]), Write(updates[2]))
            self.play(Write(updates[3]), Write(updates[5]))
            self.wait(0.1)
            t_play = 0.5
            self.play(updates[2].animate.move_to(true_pos_1), run_time=t_play)
            self.play(Write(updates[1]), run_time=t_play)
            self.play(updates[5].animate.move_to(true_pos_2), run_time=t_play)
            self.play(Write(updates[4]), run_time=t_play)

        alpha = 0.4
        alpha_eq = MathTex(r'\alpha = ' + str(alpha), font_size=self.font_size).next_to(self.axes, RIGHT * 1.5) \
            .shift(UP * 2).set_color(BLUE)
        if self.other_scene is None:
            self.play(Write(alpha_eq), run_time=0.8)
        tt = 3.9
        if self.other_scene is None:
            self.play(self.e_arrow.animate.set_value(alpha), run_time=tt)
            self.wait(0.5)


class ConvergingExecution(Execution):
    def __init__(self):
        super().__init__()
        self.step_size = 0.85
        self.x = 0.7
        self.y = 0.8
        self.n_steps_to_play = 10

    def construct(self):
        formulae_init, formula_f = self.setup_static_stuff()
        updates_initial = self.get_updates().next_to(formulae_init, DOWN * 2)
        align_left(updates_initial, formulae_init)

        self.play(Write(updates_initial[0]), Write(updates_initial[1]))

        self.play_dynamic_stuff(updates_initial, formula_f, formulae_init=formulae_init)


class OneDimensional(Scene):
    def func(self, u, v):
        return np.array([np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), u])

    def construct(self):
        axes = ThreeDAxes(x_range=[-4, 4], x_length=8)
        surface = Surface(
            lambda u, v: axes.c2p(*self.func(u, v)),
            u_range=[-PI, PI],
            v_range=[0, TAU],
            resolution=8,
        )
        self.set_camera_orientation(theta=70 * DEGREES, phi=75 * DEGREES)
        self.add(axes, surface)


class FinishingTouch(MultiplePaths):

    def construct(self):
        self.setup()
        self.precompute_mobjects()
        self.play(Create(self.surf), run_time=1.5)
        self.play(self.surf.animate.set_opacity(0.8), run_time=0.5)
        self.wait(0.1)

        # Initial point
        init_point = self.init_point
        init_point.set_opacity(1)
        self.add(init_point)
        self.play(Create(init_point), run_time=0.5)

        # Print paths
        segments = self.paths
        for path_id, path in enumerate(segments):
            for seg in path:
                seg.set_opacity(1)
                self.add(seg)
                if path_id == 0:
                    self.play(Create(seg), run_time=0.28)
                else:
                    self.play(Create(seg), run_time=0.28)
            if path_id == 0:
                self.wait(0.5)
            else:
                self.wait(0.3)
            break
        self.wait(26.5)
