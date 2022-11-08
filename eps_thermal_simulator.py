#!/usr/bin/python3
#
###
# Copyright 2022 University of Luxembourg
#
#  Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###
# Author: André Stemper (andre.stemper@uni.lu)
###

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from contextlib import closing
from datetime import datetime
import os
import math
import glob
import time
import signal
import gc

# enable features
enable_surface_radiation = True  # enable surface radiation
enable_surface_convection = True  # enable surface convection
enable_radiative_panel_loss = True  # enable radiative panel loss
enable_plotting_material_map = True  # enable plotting material map
enable_dry_run = False  # prepare everything but do not actually solve

# multiprocessing
enable_multiprocessing = True
multiprocessing_set_priority = True  # set process priority
multiprocessing_number_of_cpus = 16  # number of cpus to use.
multiprocessing_niceness = 0  # negative numbers require root rights
multiprocessing_use_lock = False  # use lock to synchronize shared memory access
multiprocessing_adaptation = (0.1, 0.5)
multiprocessing_enable_adaptation = True
multiprocessing_enable_signal_handler = True  # handle subprocess signals

if enable_multiprocessing:
    import multiprocessing as mp
    import psutil
    import ctypes
    multiprocessing_number_of_cpus = min(
        multiprocessing_number_of_cpus, mp.cpu_count())

multiprocessing_number_of_processes = multiprocessing_number_of_cpus
multiprocessing_number_of_slices = multiprocessing_number_of_cpus
multiprocessing_min_number_of_slices = 4
multiprocessing_max_number_of_slices = multiprocessing_number_of_slices

# enable exporting pngs in separate processes
enable_multiprocessing_heatmap = True
# number of processes for drawing heatmaps
multiprocessing_heatmap_number_of_processes = 8
if not enable_multiprocessing:
    # enable exporting pngs in separate processes
    enable_multiprocessing_heatmap = False
    multiprocessing_set_priority = False           # set process priority
else:
    multiprocessing_heatmap_number_of_processes = min(
        multiprocessing_heatmap_number_of_processes, mp.cpu_count())

cooling_cycle_every = 1000  # do some pause every <nth> iterations
cooling_cycle_duration = 1000  # [ms] duration of cooling cycle

# board dimensions
dimension_x = 0.106                             # [m]
dimension_y = 0.106                             # [m]
dimension_z = 0.0015                            # [m] board thickness

# smallest dimensions (for stability)
# [m] smallest resolution in x direction
dimension_smallest_x = 0.0015
# [m] smallest resolution in y direction
dimension_smallest_y = 0.0015

# number of cells inside the field
number_of_cells_x = 41                          # [1]
number_of_cells_y = 41                          # [1]
number_of_cells_z = 1                           # [1] 2D model
border_cells = 1                                # [1]

delta_x = dimension_x/number_of_cells_x
delta_y = dimension_y/number_of_cells_y
delta_z = dimension_z/number_of_cells_z
delta_a = delta_x*delta_y
delta_v = delta_x*delta_y*delta_z
delta_xx = delta_x**2
delta_yy = delta_y**2

dimension_border_x = border_cells * delta_x
dimension_border_y = border_cells * delta_y

# solar panel
panel_height = 0.025                            # [m] partial height per PCB
panel_width = 0.1                              # [m] full width of satellite
panel_aera = panel_height * panel_width         # [m^2]
panel_absorptivity = 0.6                        # [1] absorptivity 0.1-0.4
panel_emissivity = 0.75                         # [1]

# temperatures
# all calculations are in Kelvin, results are stored in °C
celsius_to_kelvin = 273.15                    # offset between kelvin and celsius
room_temperature = celsius_to_kelvin + 20.0  # [K] room temperture
initial_temperature = celsius_to_kelvin + \
    26.0  # [K] initial temperature of the board

# constants
sigma = 5.670374e-8                             # [W/m^2.K^4] Stefan-Boltzman
# [ms]# conversion factor [s] to [ms]
time_ms_per_second = 1000.0

# timing
# [s] set to None to use simulation_number_of_iterations instead
simulation_end_time = 20000.0
# [1] used if simulation_end_time is None
simulation_number_of_iterations = 1000
# [1] do not execute if number of iterations larger that this number
simulation_max_iterations = 1500000
simulation_max_delta_t = 0.25          # [s] limit the size of delta_t
# [1] stability factor: amount to reduce critical delta_t
simulation_stability_factor = 0.75
simulation_report_every = 100           # [1] iterations
simulation_save_every = 1000          # record sensor data every <nth> iterations
simulation_abort = False         # this global will be set on signal

# heatmap plot control
heatmap_plot_every = 50            # plot every <n>th frame only
# [s|None] if None use heatmap_plot_every else draw heatmap every given amount of seconds
heatmap_plot_every_s = 10.0
heatmap_resampled = True          # enable plotting resample heatmap
heatmap_resolution = (200, 200)    # resolution of resampled heatmap
heatmap_calculated = True          # enable plotting calculated heatmap
heatmap_temperature_min = celsius_to_kelvin + 15.0  # heatmap min temperature
heatmap_temperature_max = celsius_to_kelvin + 50.0  # heatmap max temperature
heatmap_frames_calculated_directory = './calculated/'
heatmap_frames_resampled_directory = './resampled/'

heatmap_animation_output_file = 'animated.gif'
# None: automatic <nr> duration between frames in [ms]
heatmap_animation_duration = None
# time lapse factor. active if heatmap_animation_duration=None
heatmap_animation_time_lapse_factor = 20.0
heatmap_dpi = 300

#
sensor_plot_dpi = 300

# buildup
buildup_output_directory = './buildup/'

# record sensors every
record_every = 100     # record every <n>th frame only
# [s|None] if None use record_every else take record every given amount of seconds
record_every_s = 1.0

# material data


class MaterialFR4(object):
    kappa = 9.11       # [W/m.K]   conductivity
    rho = 2530         # [kg/m^3]  density
    c_p = 1080         # [J/kg.K]  specific heat capacity
    epsilon = 0.8      # [1]       emissivity
    h = 5              # [W/m^2.K] 2.5-25 convective heat transfer coefficient


class MaterialBat(object):
    kappa = 100        # [W/m.K]   conductivity
    rho = 2000         # [kg/m^3]  density
    c_p = 920          # [J/kg.K]  specific heat capacity
    epsilon = 0.8      # [1]       emissivity
    h = 5              # [W/m^2.K] 2.5-25 convective heat transfer coefficient


class MaterialPlastic(object):
    kappa = 0.35       # [W/m.K]   conductivity
    rho = 1000         # [kg/m^3]  density
    c_p = 1600         # [J/kg.K]  specific heat capacity
    epsilon = 0.9      # [1]       emissivity
    h = 5              # [W/m^2.K] 2.5-25 convective heat transfer coefficient


class MaterialAl(object):
    kappa = 160        # [W/m.K]   conductivity
    rho = 2700         # [kg/m^3]  density
    c_p = 980          # [J/kg.K]  specific heat capacity
    epsilon = 0.1      # [1]       emissivity
    h = 5              # [W/m^2.K] 2.5-25 convective heat transfer coefficient


class MaterialCu(object):
    kappa = 397        # [W/m.K]   conductivity
    rho = 8960         # [kg/m^3]  density
    c_p = 389          # [J/kg.K]  specific heat capacity
    epsilon = 0.05     # [1]       emissivity
    h = 5              # [W/m^2.K] 2.5-25 convective heat transfer coefficient


class MaterialAir(object):
    # Note air is a combination of gasses:
    # values are temperature and pressure dependant and SHOULD be recalculated
    # at every time step!
    kappa = 0.026      # [W/m.K]   conductivity
    rho = 1.2          # [kg/m^3]  density
    c_p = 1005         # [J/kg.K]  specific heat capacity
    epsilon = 0.05     # [1]       emissivity
    h = 5              # [W/m^2.K] 2.5-25 convective heat transfer coefficient


class DummyPool():
    def close(self):
        pass

    def terminate(self):
        pass

    def join(self):
        pass


def position2cell(position=(0.05, 0.05)):
    x, y = position
    cell_x, cell_y = ((border_cells+int((x/dimension_x)*number_of_cells_x),
                       border_cells+int((y/dimension_y)*number_of_cells_y)))
    return(cell_x, cell_y)


def cell2position(cell=(0.05, 0.05)):
    """ position of cell center """
    return (delta_x/2 + (cell[0]-border_cells)*delta_x, delta_y/2 + (cell[1]-border_cells)*delta_y)


class Component(object):
    def __init__(self, name="", position=(0, 0), width=dimension_x, height=dimension_y,
                 depth=0.000035, material=MaterialFR4(), first_component=False, color=(0.5, 0.5, 1.0, 0.8)):
        self.name = name
        self.position = position
        self.width = width
        self.height = height
        self.depth = depth
        self.material = material
        self.first_component = first_component
        self.color = color
        self.rectangle = ((self.position[0]-self.width/2, self.position[1]-self.height/2),
                          (self.position[0]+self.width/2, self.position[1]+self.height/2))

    def plot_heatmap(self, figure, resolution=(number_of_cells_x, number_of_cells_y)):
        # position is in borderless coords, cell in heatmap coords
        cell_x, cell_y = ((((self.position[0]-self.width / 2)/dimension_x)*resolution[0]),
                          (((self.position[1]-self.height/2) /
                            dimension_y)*resolution[1])
                          )
        ax = figure.gca()
        rect = patches.Rectangle((cell_x, cell_y),
                                 self.width/dimension_x*resolution[0],
                                 self.height/dimension_y*resolution[1],
                                 linewidth=3, linestyle='dashed', edgecolor=self.color, facecolor='none')
        ax.add_patch(rect)

    def get_coverage_at(self, cell=(0, 0)):
        # returns material contribution at cell
        cell_x = delta_x/2+(cell[0]-border_cells)*delta_x
        cell_y = delta_x/2+(cell[1]-border_cells)*delta_y
        cell_rect = ((cell_x-delta_x/2, cell_y-delta_y/2),
                     (cell_x+delta_x/2, cell_y+delta_y/2))
        # area of intersection
        intersection_a = max(0, min(cell_rect[1][0], self.rectangle[1][0]) - max(cell_rect[0][0], self.rectangle[0][0])) \
            * max(0, min(cell_rect[1][1], self.rectangle[1][1]) - max(cell_rect[0][1], self.rectangle[0][1]))
        if intersection_a >= 0:
            return intersection_a/delta_a
        return 0.0


class BuildUp(object):
    KAPPA = 0
    RHO = 1
    C_P = 2
    EPSILON = 3
    H = 4

    def __init__(self, components=[], background_material=MaterialAir()):
        self.components = components
        self.number_of_material_constants = 5
        self.rasterized_buildup = None
        self.background_material = background_material

    def rasterize(self, dimension=(number_of_cells_x, number_of_cells_y)):
        self.rasterized_buildup = np.empty(
            (self.number_of_material_constants, dimension[1]+2*border_cells, dimension[0]+2*border_cells))

        # background PCB
        # kappa   [W/m.K]   conductivity
        self.rasterized_buildup[0, :, :] = 9.11
        self.rasterized_buildup[1, :, :] = 2530  # rho     [kg/m^3]  density
        self.rasterized_buildup[2, :, :] = 1080  # c_p     [J/kg.K]  specific heat capacity
        self.rasterized_buildup[3, :, :] = 0.8  # epsilon [1]       emissivity
        self.rasterized_buildup[4, :, :] = 5.0 # h       [W/m^2.K] 2.5-25 convective heat transfer coefficient

        for j in range(0, 2*border_cells+dimension[1]):
            for i in range(0, 2*border_cells+dimension[0]):
                sum_of_weights = 0.0
                kappa = 0.0       # [W/m.K]   conductivity
                rho = 0.0         # [kg/m^3]  density
                c_p = 0.0         # [J/kg.K]  specific heat capacity
                epsilon = 0.0     # [1]       emissivity
                h = 0.0 # [W/m^2.K] 2.5-25 convective heat transfer coefficient

                for component in self.components:
                    weight = component.get_coverage_at(
                        cell=(i, j)) * component.depth/dimension_z
                    if (weight > 0) and (component.first_component == True):
                        sum_of_weights = 0.0
                        kappa = 0.0       # [W/m.K]   conductivity
                        rho = 0.0         # [kg/m^3]  density
                        c_p = 0.0         # [J/kg.K]  specific heat capacity
                        epsilon = 0.0     # [1]       emissivity
                        h = 0.0 # [W/m^2.K] 2.5-25 convective heat transfer coefficient
                    sum_of_weights += weight
                    kappa += weight * component.material.kappa
                    rho += weight * component.material.rho
                    c_p += weight * component.material.c_p
                    epsilon += weight * component.material.epsilon
                    h += weight * component.material.h
                if sum_of_weights != 0:
                    kappa /= sum_of_weights
                    rho /= sum_of_weights
                    c_p /= sum_of_weights
                    epsilon /= sum_of_weights
                    h /= sum_of_weights
                else:
                    # [W/m.K]   conductivity
                    kappa = self.background_material.kappa
                    rho = self.background_material.rho      # [kg/m^3]  density
                    # [J/kg.K]  specific heat capacity
                    c_p = self.background_material.c_p
                    # [1]       emissivity
                    epsilon = self.background_material.epsilon
                    # [W/m^2.K] 2.5-25 convective heat transfer coefficient
                    h = self.background_material.h

                self.rasterized_buildup[0, j, i] = kappa
                self.rasterized_buildup[1, j, i] = rho
                self.rasterized_buildup[2, j, i] = c_p
                self.rasterized_buildup[3, j, i] = epsilon
                self.rasterized_buildup[4, j, i] = h

        if enable_plotting_material_map:
            plt.clf()
            plt.pcolormesh(
                self.rasterized_buildup[0, border_cells:-border_cells, border_cells:-border_cells], cmap=plt.cm.jet)
            self.plot_heatmap(plt, resolution=(dimension[0], dimension[1]))
            plt.colorbar()
            plt.savefig(buildup_output_directory+'kappa.png')
            plt.clf()
            plt.pcolormesh(
                self.rasterized_buildup[1, border_cells:-border_cells, border_cells:-border_cells], cmap=plt.cm.jet)
            self.plot_heatmap(plt, resolution=(dimension[0], dimension[1]))
            plt.colorbar()
            plt.savefig(buildup_output_directory+'rho.png')
            plt.clf()
            plt.pcolormesh(
                self.rasterized_buildup[2, border_cells:-border_cells, border_cells:-border_cells], cmap=plt.cm.jet)
            self.plot_heatmap(plt, resolution=(dimension[0], dimension[1]))
            plt.colorbar()
            plt.savefig(buildup_output_directory+'c_p.png')
            plt.clf()
            plt.pcolormesh(
                self.rasterized_buildup[3, border_cells:-border_cells, border_cells:-border_cells], cmap=plt.cm.jet)
            self.plot_heatmap(plt, resolution=(dimension[0], dimension[1]))
            plt.colorbar()
            plt.savefig(buildup_output_directory+'epsilon.png')
            plt.clf()
            plt.pcolormesh(
                self.rasterized_buildup[4, border_cells:-border_cells, border_cells:-border_cells], cmap=plt.cm.jet)
            self.plot_heatmap(plt, resolution=(dimension[0], dimension[1]))
            plt.colorbar()
            plt.savefig(buildup_output_directory+'h.png')

    def plot_heatmap(self, figure, resolution=(number_of_cells_x, number_of_cells_y)):
        # plot components
        for component in self.components:
            component.plot_heatmap(figure, resolution=resolution)

    def coefficients(self, cell=(0, 0)):
        # return all
        return tuple(self.rasterized_buildup[:, cell[1], cell[0]])

    def kappa(self, cell=(0, 0)):
        # [W/m.K] conductivity
        return self.rasterized_buildup[0, cell[1], cell[0]]

    def rho(self, cell=(0, 0)):
        # [kg/m^3] density
        return self.rasterized_buildup[1, cell[1], cell[0]]

    def c_p(self, cell=(0, 0)):
        # # [J/kg.K] specific heat capacity
        return self.rasterized_buildup[2, cell[1], cell[0]]

    def epsilon(self, cell=(0, 0)):
        # [1] emissivity
        return self.rasterized_buildup[3, cell[1], cell[0]]

    def h(self, cell=(0, 0)):
        # [W/m^2.K] 2.5-25 convective heat transfer coefficient
        return self.rasterized_buildup[4, cell[1], cell[0]]


class HeatModulation(object):
    pass


class ExternalHeatModulation(HeatModulation):
    def __init__(self, name='H0', file='heatsource_sequence.csv', time='t'):
        """ assumes ordered values """
        self.name = name
        import pandas
        data = pandas.read_csv(file, sep=';', header=0)
        values = list(zip(data[time].tolist(), data[name].tolist()))
        self.__iterator = iter(values)
        self.current_value = next(self.__iterator)

    def get(self, t=0.0):
        tt, vv = self.current_value
        if t <= tt:
            return vv
        self.current_value = next(self.__iterator)
        return self.get(t=t)


class HeatSource(object):
    def __init__(self, name, position=(0, 0), width=0.006, height=0.006, heat=0.1, color=(0.8, 0.2, 0.2, 0.5)):
        """
            Point heat source
            heat: power at position [W]
        """
        self.name = name
        self.position = position
        self.heat = heat        # [W]
        self.color = color
        self.width = width
        self.height = height
        # rectangle
        self.rectangle = ((self.position[0]-self.width/2, self.position[1]-self.height/2),
                          (self.position[0]+self.width/2, self.position[1]+self.height/2))
        # aera of component
        self.rectangle_a = abs((self.rectangle[0][0]-self.rectangle[1][0]) *
                               (self.rectangle[0][1]-self.rectangle[1][1]))

    def plot_heatmap(self, figure, resolution=(number_of_cells_x, number_of_cells_y)):
        cell_x, cell_y = ((((self.position[0]-self.width / 2)/dimension_x)*resolution[0]),
                          (((self.position[1]-self.height/2) /
                            dimension_y)*resolution[1])
                          )
        ax = figure.gca()
        rect = patches.Rectangle((cell_x, cell_y),
                                 self.width/dimension_x*resolution[0],
                                 self.height/dimension_y*resolution[1],
                                 linewidth=1, edgecolor=self.color, facecolor='none')
        ax.add_patch(rect)

    def get_specific_heat_at(self, t, cell=(0, 0)):
        # returns specific heat at cell
        cell_x = (cell[0]-border_cells)*delta_x
        cell_y = (cell[1]-border_cells)*delta_y
        cell_rect = ((cell_x-delta_x/2, cell_y-delta_y/2),
                     (cell_x+delta_x/2, cell_y+delta_y/2))
        # area of intersection
        intersection_a = max(0, min(cell_rect[1][0], self.rectangle[1][0]) - max(cell_rect[0][0], self.rectangle[0][0])) \
            * max(0, min(cell_rect[1][1], self.rectangle[1][1]) - max(cell_rect[0][1], self.rectangle[0][1]))
        if intersection_a > 0:
            heat = self.heat
            # if callable(heat):
            if isinstance(heat, HeatModulation):
                heat = heat.get(t=t)
            # [W/m^3]
            return heat/(self.rectangle_a*delta_z) * (intersection_a/delta_a)
        return 0.0  # [W/m^3]


class BilinearInterpolator(object):

    def __interpolate(self, u, position=(0, 0), A=(0, 0), B=(0, 0), C=(0, 0), D=(0, 0), debug_mark_point=False):
        # DC
        # AB
        T_A = self.get_cell_value(u, position=A)
        T_B = self.get_cell_value(u, position=B)
        T_C = self.get_cell_value(u, position=C)
        T_D = self.get_cell_value(u, position=D)
        T_alpha = (((B[0]-A[0])-(position[0]-A[0]))/(B[0]-A[0])) * \
            T_A + (((position[0]-A[0]))/(B[0]-A[0]))*T_B
        T_beta = (((C[0]-D[0])-(position[0]-A[0]))/(C[0]-D[0])) * \
            T_D + (((position[0]-D[0]))/(C[0]-D[0]))*T_C
        T = (((D[1]-A[1])-(position[1]-A[1]))/(D[1]-A[1])) * \
            T_alpha + (((position[1]-A[1]))/(D[1]-A[1]))*T_beta
        return T

    def get_value(self, u, position=(0, 0), do_not_interpolate=False, debug_mark_point=False):
        if do_not_interpolate:
            return self.get_cell_value(u, position=position, debug_mark_point=debug_mark_point)
        center_x, center_y = cell2position(cell=position2cell(position))
        position = (position[0] + delta_x/2, position[1] + delta_y/2)
        if position[0] >= center_x:
            # right side
            if position[1] >= center_y:
                # top
                return self.__interpolate(u,
                                          position=position,
                                          A=(center_x,         center_y),
                                          B=(center_x+delta_x, center_y),
                                          C=(center_x+delta_x, center_y+delta_y),
                                          D=(center_x,         center_y+delta_y),
                                          debug_mark_point=debug_mark_point
                                          )
            else:
                # bottom
                return self.__interpolate(u,
                                          position=position,
                                          A=(center_x,         center_y),
                                          B=(center_x+delta_x, center_y),
                                          C=(center_x+delta_x, center_y-delta_y),
                                          D=(center_x,         center_y-delta_y),
                                          debug_mark_point=debug_mark_point
                                          )
        else:
            # left side
            if position[1] >= center_y:
                # top
                return self.__interpolate(u,
                                          position=position,
                                          A=(center_x,         center_y),
                                          B=(center_x-delta_x, center_y),
                                          C=(center_x-delta_x, center_y+delta_y),
                                          D=(center_x,         center_y+delta_y),
                                          debug_mark_point=debug_mark_point
                                          )
            else:
                # bottom
                return self.__interpolate(u,
                                          position=position,
                                          A=(center_x,         center_y),
                                          B=(center_x-delta_x, center_y),
                                          C=(center_x-delta_x, center_y-delta_y),
                                          D=(center_x,         center_y-delta_y),
                                          debug_mark_point=debug_mark_point
                                          )
        # return 0.0

    def get_cell_value(self, u, position=(0, 0), debug_mark_point=False):
        x, y = position2cell(position)
        # if debug_mark_point:
        #     u[y][x] = 1000.0 # mark point
        return u[y][x]


class TemperatureSensor(object):
    def __init__(self, name="---", position=(0.05, 0.05), resolution=1.0, color=(1.0, 0.5, 0.5, 0.5)):
        self.name = name
        self.position = position
        self.resolution = resolution
        self.color = color
        self.bilinear_interpolator = BilinearInterpolator()
        self.sensor_data_latent = []
        self.sensor_data_discretized = []
        self.record_times = []
        self.keyframe_id = []

    def plot_heatmap(self, figure, resolution=(number_of_cells_x, number_of_cells_y)):
        cell_x, cell_y = ((border_cells + ((self.position[0]/dimension_x)*resolution[0]),
                           border_cells + ((self.position[1]/dimension_y)*resolution[1])))
        figure.plot(cell_x-border_cells, cell_y -
                    border_cells, 'x', color=self.color)

    def plot_latent(self, figure):
        if len(self.record_times) > 0:
            figure.plot(self.record_times, self.sensor_data_latent,
                        label=self.name+"_latent")

    def plot_discretized(self, figure):
        if len(self.record_times) > 0:
            figure.plot(self.record_times,
                        self.sensor_data_discretized, label=self.name)

    def discretize(self, x):
        # Mid-point discretization
        try:
            return float(int((float(x+0.5*self.resolution)*self.resolution)))/self.resolution
        except ValueError:
            return 0

    def sample(self, t, u, id, do_not_interpolate=True):
        self.keyframe_id.append(id)
        self.record_times.append(t)
        value = self.bilinear_interpolator.get_value(
            u, self.position, do_not_interpolate=do_not_interpolate)
        # convert to degree celsius
        value = value - celsius_to_kelvin
        self.sensor_data_latent.append(value)
        # discretize value
        dvalue = self.discretize(value)
        # store discretized value
        self.sensor_data_discretized.append(dvalue)


class KeyFrames(object):
    def __init__(self, keyframes=[]):
        self.__keyframes = []
        self.__keyframes.extend(keyframes)
        self.current_keyframe = None
        self.__reset_iterator()

    def __reset_iterator(self):
        self.__iterator = iter(self.__keyframes)
        if len(self.__keyframes) > 0:
            self.current_keyframe = next(self.__iterator)

    def append(self, keyframe):
        self.__keyframes.append(keyframe)
        self.__reset_iterator()

    def get(self, t=0.0):
        if self.current_keyframe == None:
            raise StopIteration
        try:
            return self.current_keyframe.get(t=t)
        except StopIteration:
            self.current_keyframe = next(self.__iterator)
        return self.get(t=t)


class KeyFrame(object):

    def __init__(self, id=0, t_start=0, t_end=0, illumination_enabled=False,
                 illumination_theta=0.0, heat_sources=[],
                 enable_radiative_panel_loss=True,
                 I_illumination=1360):
        self.id = int(id)
        self.t_start = t_start
        self.t_end = t_end
        self.illumination_enabled = illumination_enabled
        self.illumination_theta = illumination_theta
        self.enable_radiative_panel_loss = enable_radiative_panel_loss
        self.heat_sources = heat_sources
        self.I_illumination = I_illumination

    def get(self, t=0.0):
        if (t >= self.t_start) and (t < self.t_end):
            return self
        raise StopIteration


class Problem(object):
    stability_factor = 0.8  # to ensure numerical stability

    def __init__(self,  buildup=None, keyframes=None, sensors=[], stability_factor=0.8, dimension=(number_of_cells_x, number_of_cells_y), simulation_max_delta_t=0.5):
        self.stability_factor = stability_factor  # [1]
        self.buildup = buildup  # [1]
        self.keyframes = keyframes  # [1]
        self.sensors = sensors  # [1]
        self.delta_t = 0.0001  # [s]
        self.number_of_iterations = 10  # [1]
        self.dimension = dimension  # [1]
        self.simulation_max_delta_t = simulation_max_delta_t  # [s]
        self.multiprocessing_number_of_slices = multiprocessing_number_of_slices
        self.outter_execution_time = 0.1
        self.avg_inner_execution_time = 0.1

    def cleanup_solution(self):
        files = glob.glob(heatmap_frames_calculated_directory+'*')
        for f in files:
            os.remove(f)
        files = glob.glob(heatmap_frames_resampled_directory+'*')
        for f in files:
            os.remove(f)
        files = glob.glob(buildup_output_directory+'*')
        for f in files:
            os.remove(f)

    def generate_animation(self, frames_directory=heatmap_frames_calculated_directory, field_frame_name='field_*.png'):
        png_path = frames_directory+field_frame_name
        # Create the frames
        frames = []
        images = glob.glob(png_path)
        images.sort()
        if len(images) == 0:
            return
        for i in images:
            try:
                new_frame = Image.open(i)
                frames.append(new_frame)
            except:
                return
        # save to gif
        if heatmap_animation_duration is None:
            duration = self.delta_t*time_ms_per_second/heatmap_animation_time_lapse_factor
        else:
            duration = heatmap_animation_duration
        frames[0].save(heatmap_animation_output_file, format='GIF',
                       append_images=frames[1:], save_all=True, duration=duration, loop=0)

    # animate

    def plotheatmap(self, u_k, k, t,
                    frame_output=heatmap_frames_calculated_directory +
                    'field_{:08d}',
                    T_min=celsius_to_kelvin-5, T_max=celsius_to_kelvin+5,
                    sensors=[], heat_sources=[], resolution=(1000, 1000), autorange=False, resampled=True):

        plt.clf()
        if resampled:
            # copy border from inside to make the outer points look correct
            u_k = u_k.copy()
            u_k[:, 0] = u_k[:, border_cells]
            u_k[0, :] = u_k[border_cells, :]
            u_k[:, number_of_cells_x+border_cells] = u_k[:, number_of_cells_x]
            u_k[number_of_cells_y+border_cells, :] = u_k[number_of_cells_y, :]

            bi = BilinearInterpolator()
            hires = np.empty((resolution[1], resolution[0]))
            p0 = 1.0/resolution[0]*dimension_x
            p1 = 1.0/resolution[1]*dimension_y
            for j in range(0, resolution[1]):
                for i in range(0, resolution[0]):
                    hires[j, i] = bi.get_value(u_k, position=(i*p0, j*p1))
            u = hires
        else:
            u = u_k[border_cells:-border_cells, border_cells:-border_cells]
            resolution = (number_of_cells_x, number_of_cells_y)

        # U field is borderless
        plt.title("Temperature [K] at t={:03.03f} [s]".format(t))
        plt.xlabel("x")
        plt.ylabel("y")
        if autorange:
            plt.pcolormesh(u, cmap=plt.cm.jet)
        else:
            plt.pcolormesh(u, cmap=plt.cm.jet, vmin=T_min, vmax=T_max)
        plt.colorbar()
        ax = plt.gca()
        ax.set_aspect('equal')

        # plot sensors
        for sensor in sensors:
            sensor.plot_heatmap(plt, resolution=resolution)

        # plot heat sources
        for heat_source in heat_sources:
            heat_source.plot_heatmap(plt, resolution=resolution)

        # buildup
        self.buildup.plot_heatmap(plt, resolution=resolution)

        # store
        plt.savefig(frame_output.format(k), dpi=heatmap_dpi)
        return plt

    def plotheatmap_mp(self, pool, u_k, k, t,
                       frame_output=heatmap_frames_calculated_directory +
                       'field_{:08d}',
                       T_min=celsius_to_kelvin-5, T_max=celsius_to_kelvin+5,
                       sensors=[], heat_sources=[], resolution=(1000, 1000), autorange=False, resampled=True):
        args = (u_k, k, t,
                frame_output,
                T_min, T_max,
                sensors, heat_sources, resolution, autorange, resampled)
        pool.map_async(self.plotheatmap_mp_worker, [args])

    def plotheatmap_mp_worker(self, args):
        (u_k, k, t,
         frame_output,
         T_min, T_max,
         sensors, heat_sources, resolution, autorange, resampled) = args
        self.plotheatmap(u_k, k, t,
                         frame_output=frame_output,
                         T_min=T_min, T_max=T_max,
                         sensors=sensors, heat_sources=heat_sources,
                         resolution=resolution, resampled=resampled, autorange=autorange)

    def multiprocessing_heatmap_initialize(self, args):
        if multiprocessing_set_priority:
            p = psutil.Process(os.getpid())
            try:
                p.nice(multiprocessing_niceness)
            except:
                print("Negative niceness requires root (sudo) rights!")
        if multiprocessing_enable_signal_handler:
            def sig_int(signal_num, frame):
                global simulation_abort
                simulation_abort = True
            signal.signal(signal.SIGINT, sig_int)

    def calculate_timestep(self):
        # delta_t deduced from stability constraint: gamma = alpha * delta_t * (1/delta_x**2 + 1/delta_y**2) with gamma <= 1/2
        delta_t = 10000000000
        smallest_xx = min(delta_x, dimension_smallest_x)**2
        smallest_yy = min(delta_y, dimension_smallest_y)**2
        for j in range(border_cells, border_cells+self.dimension[1]):
            for i in range(border_cells, border_cells+self.dimension[0]):
                cell = (i, j)
                this_delta_t = self.stability_factor * 1 / (2.0 * self.buildup.kappa(cell=cell)
                                                            / (self.buildup.rho(cell=cell) * self.buildup.c_p(cell=cell)) * (1/(smallest_xx) + 1/(smallest_yy)))
                delta_t = min(delta_t, this_delta_t)
        delta_t = min(delta_t, self.simulation_max_delta_t)
        self.delta_t = delta_t
        return delta_t

    def calculate_number_of_iterations(self, t=5.0):
        self.number_of_iterations = math.ceil(t/self.delta_t)
        return self.number_of_iterations

    def apply_neumann_boundary_conditions(self, u, illumination_theta=0,
                                          I_illumination=1360, illumination_enabled=False):
        global delta_x, delta_y, delta_a, panel_absorptivity, sigma, kappa
        global room_temperature

        u0, shared_u0, idx0 = u[0]  # [K]
        u1, shared_u1, idx0 = u[1]  # [K]

        dI_top = 0.0  # [W/m^2] heat flux density
        dI_bottom = 0.0  # [W/m^2] heat flux density
        dI_left = 0.0  # [W/m^2] heat flux density
        dI_right = 0.0  # [W/m^2] heat flux density

        # irradiance due to illumination
        if illumination_enabled:
            illumination_theta = illumination_theta % (2*math.pi)

            def cell_irradiance(angle_factor, number_of_cells, delta_direction):
                # power_on_panel [W] = absorptivity [1] * I_illumination [W/m^2] * panel_aera [m^2]
                P_panel = panel_absorptivity * I_illumination * angle_factor * panel_aera
                # power_per_cell [W] = power_on_panel [W] / number_of_cells_in_contact_with_panel [1]
                P_cell = P_panel / number_of_cells
                # irradiance_per_cell = power_per_cell [W] / (delta_x [m] * delta_z [m]) = [W/m^2]
                I_cell = P_cell / (delta_direction * delta_z)
                return I_cell

            if (illumination_theta >= 0) and (illumination_theta <= math.pi/2):
                # 0 - 90 top, left
                I_cell = cell_irradiance(
                    math.cos(illumination_theta), number_of_cells_x, delta_x)
                dI_top += I_cell  # [W/m^2]
                I_cell = cell_irradiance(
                    math.sin(illumination_theta), number_of_cells_y, delta_y)
                dI_left += I_cell  # [W/m^2]

            elif (illumination_theta >= math.pi/2) and (illumination_theta <= math.pi):
                # 90 - 180
                theta2 = illumination_theta - math.pi/2
                I_cell = cell_irradiance(
                    math.cos(theta2), number_of_cells_y, delta_y)
                dI_left += I_cell  # [W/m^2]
                I_cell = cell_irradiance(
                    math.sin(theta2), number_of_cells_x, delta_x)
                dI_bottom += I_cell  # [W/m^2]

            elif (illumination_theta >= math.pi) and (illumination_theta <= 3/2*math.pi):
                # 180 - 270
                theta2 = illumination_theta - math.pi
                I_cell = cell_irradiance(
                    math.cos(theta2), number_of_cells_x, delta_x)
                dI_bottom += I_cell  # [W/m^2]
                I_cell = cell_irradiance(
                    math.sin(theta2), number_of_cells_y, delta_y)
                dI_right += I_cell  # [W/m^2]

            else:
                # 270 - 360
                theta2 = illumination_theta - 3/2*math.pi
                I_cell = cell_irradiance(
                    math.cos(theta2), number_of_cells_y, delta_y)
                dI_right += I_cell  # [W/m^2]
                I_cell = cell_irradiance(
                    math.sin(theta2), number_of_cells_x, delta_x)
                dI_top += I_cell  # [W/m^2]

        # top
        # -kappa  * (u1[2*border_cells+number_of_cells_y-1, :] - u0[2*border_cells+number_of_cells_y-2, :] ) / delta_y = q_dot
        # -kappa  = [W/m.K]
        # [W/m.K] * [K]/[m] = [W/m^2]
        # [W/m.K] * [K/m]   = [W/m^2]
        # [W.K/m^2.K] = [W/m^2]
        # [W/m^2] = [W/m^2]
        #
        # kappa  * (u1[2*border_cells+number_of_cells_y-1, :] - u0[2*border_cells+number_of_cells_y-2, :] ) / delta_y =  q_dot
        #          (u1[2*border_cells+number_of_cells_y-1, :] - u0[2*border_cells+number_of_cells_y-2, :] ) =  delta_y/kappa * q_dot
        #           u1[2*border_cells+number_of_cells_y-1, :] = u0[2*border_cells+number_of_cells_y-2, :]  + delta_y/kappa * q_dot
        # [K] = [m]/[W/m.K] * {}
        # [K] = [m/W/m.K] * {}
        # [K] = [m^2.K/W] * {W/m^2}
        # top
        kappa = self.buildup.rasterized_buildup[0,
                                                2*border_cells+number_of_cells_y-1, :]
        u1[2*border_cells+number_of_cells_y-1, :] = u0[2*border_cells + number_of_cells_y-2, :] \
            + delta_y/kappa * dI_top
        # kappa = self.buildup.rasterized_buildup[0, 2*border_cells+number_of_cells_y-1, border_cells:-border_cells]
        # u1[2*border_cells+number_of_cells_y-1, :] = u0[2*border_cells + number_of_cells_y-2, :]
        # u1[2*border_cells+number_of_cells_y-1, border_cells:-border_cells] += delta_y/kappa * dI_top

        if enable_radiative_panel_loss:
            u1[2*border_cells+number_of_cells_y-1, :] -= 1/(delta_y*kappa) * panel_height \
                * (panel_width/number_of_cells_y) * sigma * panel_emissivity \
                * (u0[2*border_cells + number_of_cells_y-2, :]**4 - room_temperature**4)
            # u1[2*border_cells+number_of_cells_y-1, border_cells:-border_cells] -= 1/(delta_y*kappa) * panel_height \
            #     * (panel_width/number_of_cells_y) * sigma * panel_emissivity \
            #     * (u0[2*border_cells + number_of_cells_y-2, border_cells:-border_cells]**4 - room_temperature**4)
        # bottom
        kappa = self.buildup.rasterized_buildup[0, 0, :]
        u1[0, :] = u0[1, :] + delta_y/kappa * dI_bottom
        # kappa = self.buildup.rasterized_buildup[0, 0, border_cells:-border_cells]
        # u1[0, :] = u0[1, :]
        # u1[0, border_cells:-border_cells] += delta_y/kappa * dI_bottom
        if enable_radiative_panel_loss:
            u1[0, :] -= 1/(delta_y*kappa) * panel_height \
                * (panel_width/number_of_cells_y) * sigma * panel_emissivity \
                * (u0[1, :]**4 - room_temperature**4)
            # u1[0, border_cells:-border_cells] -= 1/(delta_y*kappa) * panel_height \
            #     * (panel_width/number_of_cells_y) * sigma * panel_emissivity \
            #     * (u0[1, border_cells:-border_cells]**4 - room_temperature**4)
        # left
        kappa = self.buildup.rasterized_buildup[0, :, 0]
        u1[:, 0] = u0[:, 1] + delta_x/kappa * dI_left
        # kappa = self.buildup.rasterized_buildup[0, border_cells:-border_cells, 0]
        # u1[:, 0] = u0[:, 1]
        # u1[border_cells:-border_cells, 0] += delta_x/kappa * dI_left
        if enable_radiative_panel_loss:
            u1[:, 0] -= 1/(delta_x*kappa) * panel_height \
                * (panel_width/number_of_cells_y) * sigma * panel_emissivity \
                * (u0[:, 1]**4 - room_temperature**4)
            # u1[border_cells:-border_cells, 0] -= 1/(delta_x*kappa) * panel_height \
            #     * (panel_width/number_of_cells_y) * sigma * panel_emissivity \
            #     * (u0[border_cells:-border_cells, 1]**4 - room_temperature**4)
        # right
        kappa = self.buildup.rasterized_buildup[0,
                                                :, 2*border_cells+number_of_cells_x-1]
        u1[:, 2*border_cells+number_of_cells_x-1] = u0[:, 2 * border_cells+number_of_cells_x-2] \
            + delta_x/kappa * dI_right
        # kappa = self.buildup.rasterized_buildup[0, border_cells:-border_cells, 2*border_cells+number_of_cells_x-1]
        # u1[:, 2*border_cells+number_of_cells_x-1] = u0[:, 2 * border_cells+number_of_cells_x-2]
        # u1[border_cells:-border_cells, 2*border_cells+number_of_cells_x-1] += delta_x/kappa * dI_right
        if enable_radiative_panel_loss:
            # [K] =  [1/m] * 1/[W/m.K] * [m^2] * [W/m^2]
            # [K] =  [1/m] * [m.K/W] * [m^2] * [W/m^2]
            u1[:, 2*border_cells+number_of_cells_x-1] -= 1/(delta_x*kappa) * panel_height \
                * (panel_width/number_of_cells_y) * sigma * panel_emissivity \
                * (u0[:, 2 * border_cells+number_of_cells_x-2]**4 - room_temperature**4)
            # u1[border_cells:-border_cells, 2*border_cells+number_of_cells_x-1] -= 1/(delta_x*kappa) * panel_height \
            #     * (panel_width/number_of_cells_y) * sigma * panel_emissivity \
            #     * (u0[border_cells:-border_cells, 2 * border_cells+number_of_cells_x-2]**4 - room_temperature**4)

    def diffusion_equation_iteration(self, u0, u1, range_i, range_j, keyframe, t, delta_t):
        global simulation_abort
        heat_sources = keyframe.heat_sources
        room_temperature_4 = room_temperature**4
        sigma_delta_a = sigma * delta_a
        delta_a_delta_v = delta_a/delta_v
        range_j = range(range_j[0], range_j[1])
        range_i = range(range_i[0], range_i[1])
        for j in range_j:
            for i in range_i:
                if simulation_abort:
                    return
                cell = (i, j)

                # speed up: get all material values at cell at once
                kappa, rho, c_p, epsilon, h_conv = self.buildup.coefficients(
                    cell=cell)

                # sources
                g = 0.0  # [W/m^3]
                # apply heat sources
                for source in heat_sources:
                    # [W/m^3]
                    g = g + source.get_specific_heat_at(t, cell=cell)

                # radiative gain / loss
                if enable_surface_radiation:
                    # radiative component
                    # surface radiation (might not be correct as the PCB is sandwitched between other PCBs of same temperature)
                    # the surounding room is assument to be huge compared with one delta_a
                    # W_rad = sigma_delta_a * self.buildup.epsilon(cell=cell) * (room_temperature_4 - u0[j][i]**4)
                    W_rad = sigma_delta_a * epsilon * \
                        (room_temperature_4 - u0[j][i]**4)

                    g = g + W_rad/delta_v

                if enable_surface_convection:
                    # convective component (lab environment)
                    # [W/m^3] = ([W/m^2.K] * [m^2] * [K]) / [m^3]
                    # W_conv = (self.buildup.h(cell=cell) * (u0[j][i] - room_temperature))*delta_a_delta_v # [W/m^3]
                    # [W/m^3]
                    W_conv = (
                        h_conv * (u0[j][i] - room_temperature))*delta_a_delta_v
                    g = g - W_conv

                # kappa         = [W/m.K]     : thermal conductivity
                # rho           = [kg/m^3]    : volumetric density
                # c_p           = [J/kg.K]    : specific heat capacity
                # sigma         = [W/m^2.K^4] : stefan-boltzman
                # alpha_factor2 = 1 / (rho      * c_p)
                #               = 1 / ([kg/m^3] * [J/kg.K])
                #               = 1 / [kg.J/m^3.kg.K]
                #               = 1 / [J/m^3.K]
                #               = [1/W.s/m^3.K]
                #               = [m^3.K/W.s]
                # alpha_factor1 = kappa   / (rho      * c_p)
                #               = [W/m.K] / ([kg/m^3] * [J/kg.K])
                #               = [W/m.K] / [kg.J/m^3.kg.K]
                #               = [W/m.K] / [J/m^3.K]
                #               = [W.m^3.K/m.K.J]
                #               = [W.m^3/m.J]
                #               = [W.m^3/m.W.s]
                #               = [m^3/m.s]
                #               = [m^2/s]
                # d^2T/dx^2     = [K/m^2]
                # [K]           = [s] * [K/s]
                #               = [s] * ( {W/m^3} * [m^3.K/W.s])
                # g             = [W/m^3]
                # W_rad         = sigma * delta_a * epsilon * (u0[j][i] ** 4)
                #               = [W/m^2.K^4] * [m^2]* [1] * [K^4]
                #               = [W]
                #
                # I_rad         = sigma * epsilon * (u0[j][i] ** 4)
                #               = [W/m^2.K^4] * [1] * [K^4]
                #               = [W/m^2]
                #
                # Heat equation (dT/dt -> dT1 = dT0 + dt)
                # dT1 = dT0 + dt  * (alpha_factor1 * (d^2T/dx^2 + d^2T/dy^2) + g/(rho * c_p))
                # [K] = [K] + [s] * (
                #   [m^2/s] * [K/m^2]
                # + [m^2/s] * [K/m^2]
                # + [W/m^3] * [m^3.K/W.s]
                # )

                #
                #alpha_factor2 = 1.0 / (self.buildup.rho(cell=cell) * self.buildup.c_p(cell=cell))
                #alpha_factor1 = self.buildup.kappa(cell=cell) * alpha_factor2

                alpha_factor2 = 1.0 / (rho * c_p)
                alpha_factor1 = kappa * alpha_factor2

                # heat equation 2d
                # dT/dt = alpha * d^2T/dx^2 + alpha * d^2T/dy^2 + g/(rho * c_p)
                # alpha = kappa / (rho * c_c)
                u1[j][i] = u0[j][i] + delta_t * (
                    alpha_factor1 * (
                        ((u0[j][i+1]+u0[j][i-1]-2*u0[j][i])/(delta_xx))
                        + ((u0[j+1][i]+u0[j-1][i]-2*u0[j][i])/(delta_yy))
                    )
                    + alpha_factor2 * g
                )

    def multiprocessing_solver_initialize(self, *deimp_parameters_):
        global deimp_parameters, simulation_abort
        shared_u0, shared_u1, delta_t = deimp_parameters_
        shared_u0_raw_np = self.get_npbuffer(shared_u0)
        shared_u1_raw_np = self.get_npbuffer(shared_u1)
        shared_u0_np = shared_u0_raw_np.reshape(
            ((number_of_cells_y+2*border_cells), (number_of_cells_x+2*border_cells)))
        shared_u1_np = shared_u1_raw_np.reshape(
            ((number_of_cells_y+2*border_cells), (number_of_cells_x+2*border_cells)))
        deimp_parameters = (shared_u0_np, shared_u1_np, delta_t)

        if multiprocessing_enable_signal_handler:
            def sig_int(signal_num, frame):
                global simulation_abort
                simulation_abort = True
            signal.signal(signal.SIGINT, sig_int)

        if multiprocessing_set_priority:
            p = psutil.Process(os.getpid())
            try:
                p.nice(multiprocessing_niceness)
            except:
                print("Negative niceness requires root (sudo) rights!")

    def __split_into_slices(self, range_j=(1, 20), multiprocessing_number_of_slices=16):
        """ split into slices using error accumulation """
        slices = []
        step = (range_j[1] - range_j[0]) / (multiprocessing_number_of_slices-1)
        step_short = math.floor(step)
        step_long = step_short + 1
        error = 0.0
        idx = range_j[0]
        while idx < range_j[1]:
            this_step = step_short
            if error >= 1.0:
                this_step = step_long
            this_step = min(this_step, (range_j[1] - idx))
            if this_step > 0:
                slices.append((idx, idx+this_step))
            error += (step-this_step)
            idx += this_step
        return slices

    def diffusion_equation_iteration_mp(self, pool, u0_index, range_i, range_j, keyframe, t, delta_t):
        global multiprocessing_number_of_slices

        slices = self.__split_into_slices(
            range_j=range_j, multiprocessing_number_of_slices=self.multiprocessing_number_of_slices)
        working_slice = [(u0_index,
                          slice,
                          range_i, range_j, keyframe, t
                          ) for slice in slices]

        # this is a try to decrease the main process load from 100% down (70-80% during test)
        time_adapt_start = time.time()
        if True:
            inner_execution_times = pool.map(
                self.diffusion_equation_iteration_mp_calc_slice, working_slice)
        else:
            iter = pool.imap_unordered(
                self.diffusion_equation_iteration_mp_calc_slice, working_slice)
            time.sleep(0)
            inner_execution_times = [t for t in iter]
        time_adapt_end = time.time()

        self.outter_execution_time = time_adapt_end - time_adapt_start
        self.avg_inner_execution_time = sum(
            inner_execution_times)/len(inner_execution_times)

        self.multiprocessing_execution_time_ratio = self.avg_inner_execution_time / \
            self.outter_execution_time
        if multiprocessing_enable_adaptation:
            # adapt the number of processes used to solve
            if self.multiprocessing_execution_time_ratio < multiprocessing_adaptation[0]:
                self.multiprocessing_number_of_slices -= 1
            if self.multiprocessing_execution_time_ratio >= multiprocessing_adaptation[1]:
                self.multiprocessing_number_of_slices += 1
            # limit number of slices to 0..multiprocessing_number_of_slices
            if self.multiprocessing_number_of_slices < multiprocessing_min_number_of_slices:
                self.multiprocessing_number_of_slices = multiprocessing_min_number_of_slices
            if self.multiprocessing_number_of_slices > multiprocessing_max_number_of_slices:
                self.multiprocessing_number_of_slices = multiprocessing_max_number_of_slices

    def diffusion_equation_iteration_mp_calc_slice(self, workpack):
        global deimp_parameters
        time_adapt_start = time.time()
        try:
            u0_index, working_slice, range_i, range_j, keyframe, t = workpack
            if u0_index == 0:
                shared_u0_np, shared_u1_np, delta_t = deimp_parameters
            else:
                shared_u1_np, shared_u0_np, delta_t = deimp_parameters
            self.diffusion_equation_iteration(
                shared_u0_np, shared_u1_np, range_i, working_slice, keyframe, t, delta_t)
        except KeyboardInterrupt:
            pass
        finally:
            return float(time.time()-time_adapt_start)

    def get_npbuffer(self, buffer):
        if multiprocessing_use_lock:
            return np.frombuffer(buffer.get_obj())
        return np.frombuffer(buffer)

    def solve(self, t_end=None, number_of_iterations=simulation_number_of_iterations, enable_dry_run=False):
        global delta_xx, delta_yy
        global heatmap_plot_every, record_every

        cooling_cycle_counter = 0
        record_counter = 0
        record_last_time = 0       # [s]
        heatmap_plot_counter = 0
        heatmap_plot_last_time = 0  # [s]
        simulation_report_counter = 0
        simulation_save_counter = 0
        execution_start_time = datetime.now()

        if self.keyframes is None:
            self.keyframes = KeyFrame()

        if self.buildup is None:
            return

        if multiprocessing_set_priority:
            p = psutil.Process(os.getpid())
            try:
                p.nice(multiprocessing_niceness)
            except:
                print("Negative niceness requires root (sudo) rights!")

        # rasterize buildup
        self.buildup.rasterize(dimension=self.dimension)

        # calculate timestep
        delta_t = self.calculate_timestep()

        # calculate number of iterations
        if not t_end is None:
            self.number_of_iterations = self.calculate_number_of_iterations(
                t_end)
        else:
            self.number_of_iterations = number_of_iterations

        if self.number_of_iterations > simulation_max_iterations:
            print("Number of iterations exceeds maximum number of iteratons. If want to wait, please increase simulation_max_iterations.")
            print("simulation_max_iterations={}".format(
                simulation_max_iterations))
            print("number_of_iterations={}".format(self.number_of_iterations))
            return

        print("number of iterations: {}".format(self.number_of_iterations))
        print("delta_x={}[m], delta_y={}[m], delta_t={}[s], simulated time={}[s]={}[min]".format(delta_x,
                                                                                                 delta_y, delta_t, delta_t*self.number_of_iterations,  delta_t*self.number_of_iterations/60))
        print(
            "accuracy: t+/- {}[s], x+/- {}[m], y+/- {}[m]".format(delta_t, delta_xx, delta_yy))

        if enable_dry_run:
            return

        if enable_multiprocessing:
            if multiprocessing_use_lock:
                shared_u0 = mp.Array(ctypes.c_double, (number_of_cells_x+2 *
                                     border_cells)*(number_of_cells_y+2*border_cells), lock=True)
                shared_u1 = mp.Array(ctypes.c_double, (number_of_cells_x+2 *
                                     border_cells)*(number_of_cells_y+2*border_cells), lock=True)
            else:
                shared_u0 = mp.Array(ctypes.c_double, (number_of_cells_x+2 *
                                     border_cells)*(number_of_cells_y+2*border_cells), lock=False)
                shared_u1 = mp.Array(ctypes.c_double, (number_of_cells_x+2 *
                                     border_cells)*(number_of_cells_y+2*border_cells), lock=False)
            shared_u0_raw_np = self.get_npbuffer(shared_u0)
            shared_u1_raw_np = self.get_npbuffer(shared_u1)
            u_0 = shared_u0_raw_np.reshape(
                ((number_of_cells_y+2*border_cells), (number_of_cells_x+2*border_cells)))
            u_1 = shared_u1_raw_np.reshape(
                ((number_of_cells_y+2*border_cells), (number_of_cells_x+2*border_cells)))
            process_pool = mp.Pool(processes=multiprocessing_number_of_processes,
                                   initializer=self.multiprocessing_solver_initialize, initargs=((shared_u0, shared_u1, delta_t)))
            print("calculating using {} processes.".format(
                multiprocessing_number_of_processes))
        else:
            u_0 = np.empty((number_of_cells_y+2*border_cells,
                            number_of_cells_x+2*border_cells))
            u_1 = np.empty((number_of_cells_y+2*border_cells,
                            number_of_cells_x+2*border_cells))
            shared_u0 = u_0
            shared_u1 = u_1
            process_pool = DummyPool()

        if enable_multiprocessing_heatmap:
            print("using {} processes to store .PNGs".format(
                multiprocessing_heatmap_number_of_processes))
            process_pool_heatmap = mp.Pool(processes=multiprocessing_heatmap_number_of_processes,
                                           initializer=self.multiprocessing_heatmap_initialize, initargs=(None,))
        else:
            process_pool_heatmap = DummyPool()

        # Initial state
        u_0.fill(initial_temperature)
        u_1.fill(initial_temperature)

        # np_buffer, shared_buffer, buffer_index
        u = [(u_0, shared_u0, 0), (u_1, shared_u1, 1)]

        with closing(process_pool_heatmap) as pool_heatmap:
            with closing(process_pool) as pool:
                # with closing(process_pool) as pool:
                for k in range(0, self.number_of_iterations-1, 1):
                    if simulation_abort:
                        return

                    t = k*delta_t

                    # get active keyframe
                    try:
                        keyframe = self.keyframes.get(t=t)
                    except StopIteration:
                        print(
                            "Stop calculation with the last keyframe at t={}.".format(t))
                        break

                    # past and current temperature arrays
                    u0 = u[0][0]
                    u1 = u[1][0]

                    # apply boundary conditions
                    self.apply_neumann_boundary_conditions(u,
                                                           illumination_theta=keyframe.illumination_theta,
                                                           I_illumination=keyframe.I_illumination,
                                                           illumination_enabled=keyframe.illumination_enabled)
                    # apply_dirchlet_boundary_conditions(u, k, theta)

                    if enable_multiprocessing:
                        self.diffusion_equation_iteration_mp(pool, u[0][2],
                                                             (border_cells, border_cells +
                                                              number_of_cells_x),
                                                             (border_cells, border_cells +
                                                              number_of_cells_y),
                                                             keyframe, t, delta_t)

                    else:
                        self.diffusion_equation_iteration(u0, u1,
                                                          (border_cells, border_cells +
                                                           number_of_cells_x),
                                                          (border_cells, border_cells +
                                                           number_of_cells_y),
                                                          keyframe, t, delta_t)

                    # make all sensors sample the signal
                    if record_every_s is None:
                        # iso-iterations
                        if record_counter >= record_every:
                            record_counter = 0
                            for sensor in self.sensors:
                                sensor.sample(t, u1, keyframe.id)
                        record_counter = record_counter + 1
                    else:
                        # iso-time
                        if (t-record_last_time) >= record_every_s:
                            record_last_time = t
                            for sensor in self.sensors:
                                sensor.sample(t, u1, keyframe.id)

                    # plot frame for animation
                    do_plot_heatmap = False
                    if heatmap_plot_every_s is None:
                        # iso-iterations
                        if heatmap_plot_counter >= heatmap_plot_every:
                            heatmap_plot_counter = 0
                            do_plot_heatmap = True
                        heatmap_plot_counter = heatmap_plot_counter + 1
                    else:
                        # iso-time
                        if (t-heatmap_plot_last_time) >= heatmap_plot_every_s:
                            heatmap_plot_last_time = t
                            do_plot_heatmap = True

                    if do_plot_heatmap:
                        if enable_multiprocessing_heatmap:
                            if heatmap_resampled:
                                self.plotheatmap_mp(pool_heatmap, u1, k, t,
                                                    frame_output=heatmap_frames_resampled_directory +
                                                    'field_{:08d}',
                                                    T_min=heatmap_temperature_min, T_max=heatmap_temperature_max,
                                                    sensors=self.sensors, heat_sources=keyframe.heat_sources,
                                                    resolution=heatmap_resolution, resampled=True)
                            if heatmap_calculated:
                                self.plotheatmap_mp(pool_heatmap, u1, k, t,
                                                    frame_output=heatmap_frames_calculated_directory +
                                                    'field_{:08d}',
                                                    T_min=heatmap_temperature_min, T_max=heatmap_temperature_max,
                                                    sensors=self.sensors, heat_sources=keyframe.heat_sources,
                                                    resolution=heatmap_resolution, resampled=False)
                        else:
                            if heatmap_resampled:
                                self.plotheatmap(u1, k, t,
                                                 frame_output=heatmap_frames_resampled_directory +
                                                 'field_{:08d}',
                                                 T_min=heatmap_temperature_min, T_max=heatmap_temperature_max,
                                                 sensors=self.sensors, heat_sources=keyframe.heat_sources,
                                                 resolution=heatmap_resolution, resampled=True)
                            if heatmap_calculated:
                                self.plotheatmap(u1, k, t,
                                                 frame_output=heatmap_frames_calculated_directory +
                                                 'field_{:08d}',
                                                 T_min=heatmap_temperature_min, T_max=heatmap_temperature_max,
                                                 sensors=self.sensors, heat_sources=keyframe.heat_sources,
                                                 resolution=heatmap_resolution, resampled=False)

                    if simulation_report_counter >= simulation_report_every:
                        simulation_report_counter = 0
                        execution_passed_time = datetime.now() - execution_start_time
                        remaining_time = (
                            execution_passed_time/k) * self.number_of_iterations - execution_passed_time
                        print("[working: {} remaining: {} {}/{}={:.2f}%] t={}[s] t_ix={:.4f} t_ox={:.4f} r={:.2f} #cores={}    \r".format(execution_passed_time, remaining_time, k, self.number_of_iterations, k*100 /
                              self.number_of_iterations, t, self.avg_inner_execution_time, self.outter_execution_time, self.avg_inner_execution_time/self.outter_execution_time, self.multiprocessing_number_of_slices), end='')
                       
                    simulation_report_counter = simulation_report_counter + 1

                    if simulation_save_counter >= simulation_save_every:
                        simulation_save_counter = 0
                        # periodic saving
                        # save sensor data
                        header_cols = []
                        s = self.sensors[0]
                        if len(s.record_times) > 0:
                            data = np.empty(
                                (len(s.record_times), 2+2*len(self.sensors)))
                            header_cols.append("Time [s]")
                            data[:, 0] = np.array(s.record_times)
                            header_cols.append("Keyframe")
                            data[:, 1] = np.array(s.keyframe_id)
                            for i, sensor in enumerate(self.sensors):
                                header_cols.append(
                                    "{}_latent [°C]".format(sensor.name))
                                header_cols.append(
                                    "{} [°C]".format(sensor.name))
                                data[:, 2+i *
                                     2] = np.array(sensor.sensor_data_latent)
                                data[:, 2+i*2 +
                                     1] = np.array(sensor.sensor_data_discretized)
                            np.savetxt("data.csv", tuple(data), delimiter=';', header=';'.join(
                                header_cols), comments='')

                            # plot sensors data
                            if True:
                                plt.clf()
                                plt.title("Temperature [°C] vs time")
                                plt.xlabel("t[s]")
                                plt.ylabel("T[°C]")
                                for sensor in self.sensors:
                                    sensor.plot_discretized(plt)
                                plt.legend()
                                plt.savefig("sensors.png", dpi=sensor_plot_dpi)

                            if True:
                                plt.clf()
                                plt.title("Temperature [°C] vs time")
                                plt.xlabel("t[s]")
                                plt.ylabel("T[°C]")
                                for sensor in self.sensors:
                                    sensor.plot_latent(plt)
                                plt.legend()
                                plt.savefig("sensors_latent.png",
                                            dpi=sensor_plot_dpi)
                    simulation_save_counter = simulation_save_counter + 1

                    # flip buffers
                    u = [u[1], u[0]]

                    if cooling_cycle_counter >= cooling_cycle_every:
                        cooling_cycle_counter = 0
                        time.sleep(cooling_cycle_duration)
                        gc.collect()


# --------------------------------------

def main():
    # Board buildup
    pcb_dimension_x = 0.1
    pcb_dimension_y = 0.1
    bat_width = 0.008
    bat_height = 0.050
    bat_depth = 0.008
    bat_of_center = bat_width + 0.001

    # Illumination rotating around the PCB
    # With heat pulse
    # component_power=0.015  # [W] 15mW (experimentally: heating about 4K/60s]
    component_power = 0.015  # [W] 0mW (experimentally: heating about 4K/60s]
    # [W] 0.8 calculated from battery_current_out*battery_voltage
    discipated_power = 0.4
    heatsource_color = (1, 1, 0)
    surface_heat = HeatSource("HS", (dimension_x/2, dimension_y/2),
                              width=pcb_dimension_x, height=pcb_dimension_y, heat=discipated_power)

    abnormal_heat_sources = [
        surface_heat,
        HeatSource("Ha", (dimension_x/4+0.01, dimension_y*((0.5/3)+(1/4)*(2/3))+0.01),
                   width=0.006, height=0.006, heat=component_power),

        # mppt (left row)
        HeatSource("H0",  (dimension_x/4,   dimension_y*((0.5/3)+(1/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H0"), color=heatsource_color),
        HeatSource("H1",  (dimension_x/4,   dimension_y*((0.5/3)+(2/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H1"), color=heatsource_color),
        HeatSource("H2",  (dimension_x/4,   dimension_y*((0.5/3)+(3/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H2"), color=heatsource_color),
        HeatSource("H3",  (dimension_x/4,   dimension_y*((0.5/3)+(4/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H3"), color=heatsource_color),
        # output converter (right row)
        HeatSource("H4",  (dimension_x*3/4, dimension_y*((0.5/3)+(1/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H4"), color=heatsource_color),
        HeatSource("H5",  (dimension_x*3/4, dimension_y*((0.5/3)+(2/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H5"), color=heatsource_color),
        HeatSource("H6",  (dimension_x*3/4, dimension_y*((0.5/3)+(3/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H6"), color=heatsource_color),
        HeatSource("H7",  (dimension_x*3/4, dimension_y*((0.5/3)+(4/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H7"), color=heatsource_color)

    ]

    normal_heat_sources = [
        surface_heat,
        # mppt (left row)
        HeatSource("H0",  (dimension_x/4,   dimension_y*((0.5/3)+(1/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H0"), color=heatsource_color),
        HeatSource("H1",  (dimension_x/4,   dimension_y*((0.5/3)+(2/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H1"), color=heatsource_color),
        HeatSource("H2",  (dimension_x/4,   dimension_y*((0.5/3)+(3/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H2"), color=heatsource_color),
        HeatSource("H3",  (dimension_x/4,   dimension_y*((0.5/3)+(4/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H3"), color=heatsource_color),
        # output converter (right row)
        HeatSource("H4",  (dimension_x*3/4, dimension_y*((0.5/3)+(1/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H4"), color=heatsource_color),
        HeatSource("H5",  (dimension_x*3/4, dimension_y*((0.5/3)+(2/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H5"), color=heatsource_color),
        HeatSource("H6",  (dimension_x*3/4, dimension_y*((0.5/3)+(3/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H6"), color=heatsource_color),
        HeatSource("H7",  (dimension_x*3/4, dimension_y*((0.5/3)+(4/4)*(2/3))), width=0.01,
                   height=0.01, heat=ExternalHeatModulation(name="H7"), color=heatsource_color)

    ]

    # 360 degree rotation
    rounds = 5
    nr_steps = 90
    t_start = 0.0

    keyframes = KeyFrames()
    time_step = (simulation_end_time/(nr_steps*rounds))
    for i in range(int(nr_steps*rounds)):
        hs = normal_heat_sources
        t_end = t_start + time_step

        # activate additional heat source
        if t_start > 500.0 and t_start < 800.0:
            hs = abnormal_heat_sources

        keyframe = KeyFrame(id=i,
                            t_start=t_start, t_end=t_end,
                            illumination_enabled=True,
                            illumination_theta=i*((2.0*math.pi)/(nr_steps)),
                            I_illumination=4 / (0.2*0.2),
                            heat_sources=hs
                            )

        keyframes.append(keyframe)
        t_start = t_end

    buildup = BuildUp([
        Component(name="PCB", position=(dimension_x/2,  dimension_y/2), width=pcb_dimension_x,
                  height=pcb_dimension_y, depth=0.0015, material=MaterialFR4(), first_component=False),
    ], background_material=MaterialAir()
    )

    # EPS like (position coarsly estimated)
    # T0-3 - MPPT converter    (left 1/4 upper 2/3 )
    # T4-7 - Output converter  (right 3/4 upper 2/3)
    # T8   - center
    sensor_color = (1, 0, 0)
    sensors = [
        # mppt converter (left row)
        TemperatureSensor("T0",  (dimension_x/4,   dimension_y * \
                          ((0.5/3)+(1/4)*(2/3))), 1.0, color=sensor_color),
        TemperatureSensor("T1",  (dimension_x/4,   dimension_y * \
                          ((0.5/3)+(2/4)*(2/3))), 1.0, color=sensor_color),
        TemperatureSensor("T2",  (dimension_x/4,   dimension_y * \
                          ((0.5/3)+(3/4)*(2/3))), 1.0, color=sensor_color),
        TemperatureSensor("T3",  (dimension_x/4,   dimension_y * \
                          ((0.5/3)+(4/4)*(2/3))), 1.0, color=sensor_color),
        # output converter (right row)
        TemperatureSensor("T4",  (dimension_x*3/4, dimension_y * \
                          ((0.5/3)+(1/4)*(2/3))), 1.0, color=sensor_color),
        TemperatureSensor("T5",  (dimension_x*3/4, dimension_y * \
                          ((0.5/3)+(2/4)*(2/3))), 1.0, color=sensor_color),
        TemperatureSensor("T6",  (dimension_x*3/4, dimension_y * \
                          ((0.5/3)+(3/4)*(2/3))), 1.0, color=sensor_color),
        TemperatureSensor("T7",  (dimension_x*3/4, dimension_y * \
                          ((0.5/3)+(4/4)*(2/3))), 1.0, color=sensor_color),
        # battery (center)
        TemperatureSensor("T8", (dimension_x/2, dimension_y/2),
                          1.0, color=sensor_color)
    ]

    try:
        problem = Problem(buildup=buildup, keyframes=keyframes, sensors=sensors,
                          simulation_max_delta_t=simulation_max_delta_t, stability_factor=simulation_stability_factor)
        # cleanup workspace
        problem.cleanup_solution()
        # calculate results
        problem.solve(t_end=simulation_end_time,
                      number_of_iterations=simulation_number_of_iterations, enable_dry_run=enable_dry_run)
        # generate animation
        problem.generate_animation()
    except KeyboardInterrupt:
        pass

    # save sensor data
    header_cols = []
    s = sensors[0]
    if len(s.record_times) > 0:
        data = np.empty((len(s.record_times), 2+2*len(sensors)))
        header_cols.append("Time [s]")
        data[:, 0] = np.array(s.record_times)
        header_cols.append("Keyframe")
        data[:, 1] = np.array(s.keyframe_id)
        for i, sensor in enumerate(sensors):
            header_cols.append("{}_latent [°C]".format(sensor.name))
            header_cols.append("{} [°C]".format(sensor.name))
            data[:, 2+i*2] = np.array(sensor.sensor_data_latent)
            data[:, 2+i*2+1] = np.array(sensor.sensor_data_discretized)
        np.savetxt("data.csv", tuple(data), delimiter=';',
                   header=';'.join(header_cols), comments='')

        # plot sensors data
        if True:
            plt.clf()
            plt.title("Temperature [°C] vs time")
            plt.xlabel("t[s]")
            plt.ylabel("T[°C]")
            for sensor in sensors:
                sensor.plot_discretized(plt)
            plt.legend()
            plt.savefig("sensors.png", dpi=sensor_plot_dpi)

        if True:
            plt.clf()
            plt.title("Temperature [°C] vs time")
            plt.xlabel("t[s]")
            plt.ylabel("T[°C]")
            for sensor in sensors:
                sensor.plot_latent(plt)
            plt.legend()
            plt.savefig("sensors_latent.png", dpi=sensor_plot_dpi)


if __name__ == '__main__':
    if enable_multiprocessing:
        mp.freeze_support()

    def sig_int(signal_num, frame):
        global simulation_abort
        simulation_abort = True
    signal.signal(signal.SIGINT, sig_int)

    main()
    print("")
    print("Done.")
    print("")
