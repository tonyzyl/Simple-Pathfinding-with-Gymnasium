"""
uav_sim.py

This module defines the UAV simulation environment (a Gym environment)
using Box2D and pygame. All simulation constants are organized in a configuration
dataclass so that they can be adjusted without relying on global variables.
"""

# uav_sim.py
import math
import random
import warnings
import io
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

import pygame

import Box2D
from Box2D.b2 import (
    contactListener,
    edgeShape,
)

import torch
from torch import nn
from collections import deque
import pickle

# ------------------------------------------------------------------------------
# Configuration as a dataclass
# ------------------------------------------------------------------------------
from dataclasses import dataclass, field

@dataclass
class UAVSimConfig:
    # Simulation / visualization parameters
    FPS: int = 50
    SCALE: float = 20.0
    OBSTACLE_SCALE: float = 5.0
    VIEWPORT_W: int = 600
    VIEWPORT_H: int = 400

    # These fields will be computed in __post_init__
    UAV_RADIUS: float = field(init=False)
    GOAL_RADIUS: float = field(init=False)
    WALL_THICKNESS: float = field(init=False)
    NUM_OBSTACLES: int = 10
    OBS_MAX_RAD: float = field(init=False)
    OBS_MIN_RAD: float = field(init=False)
    MIN_CLEARANCE: float = field(init=False)

    # UAV specifications
    UAV_INI_ANGLE: float = field(init=False)
    UAV_DENSITY: float = 1.0
    UAV_FRICTION: float = 0.3
    UAV_FOV: float = field(init=False)        # in radians
    UAV_NUM_RAYS: int = 20
    UAV_FOV_DISTANCE: float = field(init=False)
    UAV_ANG_POW: float = field(init=False)
    UAV_THRUST_POW: float = field(init=False)

    # Action / state sizes
    ACTION_SPACE: int = 4
    STATE_SPACE: int = field(init=False)  # computed as 5 + UAV_NUM_RAYS

    # Reward/penalty coefficients
    PEN_THRUST: float = field(init=False)
    PEN_ANG: float = field(init=False)
    PEN_OBSTACLE: float = -1
    OBS_OFFSET: float = field(init=False)
    PEN_COLLISION: float = -100
    PEN_NOT_FINISHED: float = -100
    REW_VEL: float = field(init=False)
    REW_ANGLE: float = field(init=False)
    REW_GOAL: float = 100
    REW_DIST2GOAL: float = 5

    def __post_init__(self):
        # Compute dependent values
        self.UAV_RADIUS = 10.0 / self.SCALE
        self.GOAL_RADIUS = self.UAV_RADIUS
        self.WALL_THICKNESS = 10.0 / self.SCALE
        self.OBS_MAX_RAD = min(self.VIEWPORT_W, self.VIEWPORT_H) / self.SCALE / self.OBSTACLE_SCALE
        self.OBS_MIN_RAD = self.OBS_MAX_RAD / 3
        self.MIN_CLEARANCE = self.UAV_RADIUS * 3

        self.UAV_INI_ANGLE = math.radians(0)
        self.UAV_FOV = math.radians(360)
        self.UAV_FOV_DISTANCE = 200 / self.SCALE
        self.UAV_ANG_POW = 100 / self.SCALE
        self.UAV_THRUST_POW = 100 / self.SCALE

        self.STATE_SPACE = 5 + self.UAV_NUM_RAYS

        self.PEN_THRUST = -0.1 / self.FPS
        self.PEN_ANG = -0.05 / self.FPS
        self.OBS_OFFSET = 3 * self.UAV_RADIUS
        self.REW_VEL = 1 / self.FPS
        self.REW_ANGLE = 1 / self.FPS

# Create a default configuration instance.
DEFAULT_CONFIG = UAVSimConfig()

# ------------------------------------------------------------------------------
# UAV simulation environment code
# ------------------------------------------------------------------------------
class ContactDetector(Box2D.b2ContactListener):
    def __init__(self, env):
        super(ContactDetector, self).__init__()
        self.env = env

    def BeginContact(self, contact):
        # Check if one of the bodies is the UAV and the other is an obstacle or wall
        if contact.fixtureA.body == self.env.uav or contact.fixtureB.body == self.env.uav:
            self.env.game_over = True

class SimpleUAVEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, config: UAVSimConfig = DEFAULT_CONFIG):
        self.config = config  # store the configuration
        # Define the Box2D world
        self.world = Box2D.b2World(gravity=(0, 0))
        self.uav = None
        self.goal = None
        self.dist2goal = None
        self.walls = []
        self.obstacles = []
        self.obstacles_properties = []
        self.uav_pos_history = []
        self.continuous = False

        if self.continuous:
            raise NotImplementedError
        else:
            self.action_space = spaces.Discrete(self.config.ACTION_SPACE)

        self.screen = None
        self.isopen = True

    def _create_walls(self):
        cfg = self.config
        wall_shapes = [
            edgeShape(vertices=[(0, 0), (cfg.VIEWPORT_W/cfg.SCALE, 0)]),
            edgeShape(vertices=[(0, 0), (0, cfg.VIEWPORT_H/cfg.SCALE)]),
            edgeShape(vertices=[(0, cfg.VIEWPORT_H/cfg.SCALE), (cfg.VIEWPORT_W/cfg.SCALE, cfg.VIEWPORT_H/cfg.SCALE)]),
            edgeShape(vertices=[(cfg.VIEWPORT_W/cfg.SCALE, 0), (cfg.VIEWPORT_W/cfg.SCALE, cfg.VIEWPORT_H/cfg.SCALE)])
        ]
        for shape in wall_shapes:
            wall_body = self.world.CreateStaticBody(position=(0, 0), shapes=shape)
            self.walls.append(wall_body)

    def _is_position_valid(self, new_properties):
        cfg = self.config
        for prop in self.obstacles_properties:
            distance = math.sqrt((new_properties['centroid_x'] - prop['centroid_x'])**2 + 
                                 (new_properties['centroid_y'] - prop['centroid_y'])**2)
            if distance < (new_properties['max_span'] + prop['max_span'] + cfg.MIN_CLEARANCE):
                return False

        goal_distance = math.sqrt((new_properties['centroid_x'] - self.goal[0])**2 +
                                  (new_properties['centroid_y'] - self.goal[1])**2)
        if goal_distance < (new_properties['max_span'] + 4 * cfg.UAV_RADIUS):
            return False

        uav_distance = math.sqrt((new_properties['centroid_x'] - self.uav_start_pos[0])**2 +
                                 (new_properties['centroid_y'] - self.uav_start_pos[1])**2)
        if uav_distance < (new_properties['max_span'] + 4 * cfg.UAV_RADIUS):
            return False

        return True

    def _generate_triangle_properties(self):
        cfg = self.config
        centroid_x = random.uniform(cfg.UAV_RADIUS, cfg.VIEWPORT_W / cfg.SCALE - cfg.UAV_RADIUS)
        centroid_y = random.uniform(cfg.UAV_RADIUS, cfg.VIEWPORT_H / cfg.SCALE - cfg.UAV_RADIUS)
        length = random.uniform(cfg.OBS_MIN_RAD, cfg.OBS_MAX_RAD)
        angle_offset = 2 * math.pi / 3
        vertices = []
        for i in range(3):
            angle = angle_offset * i
            vertex_x = centroid_x + length * math.cos(angle)
            vertex_y = centroid_y + length * math.sin(angle)
            vertices.append((vertex_x, vertex_y))
        rotation_angle = random.uniform(0, math.pi)
        return {'type': 'triangle', 'vertices': vertices, 'centroid_x': centroid_x, 'centroid_y': centroid_y, 'angle': rotation_angle, 'max_span': length}

    def _generate_rectangle_properties(self):
        cfg = self.config
        width = random.uniform(cfg.OBS_MIN_RAD, cfg.OBS_MAX_RAD)
        height = random.uniform(cfg.OBS_MIN_RAD, cfg.OBS_MAX_RAD)
        angle = random.uniform(0, math.pi)
        centroid_x = random.uniform(width / 2, cfg.VIEWPORT_W / cfg.SCALE - width / 2)
        centroid_y = random.uniform(height / 2, cfg.VIEWPORT_H / cfg.SCALE - height / 2)
        return {'type': 'rectangle', 'centroid_x': centroid_x, 'centroid_y': centroid_y, 'width': width, 'height': height, 'angle': angle, 'max_span': max(width, height)}

    def _generate_circle_properties(self):
        cfg = self.config
        radius = random.uniform(cfg.OBS_MIN_RAD, cfg.OBS_MAX_RAD)
        centroid_x = random.uniform(radius, cfg.VIEWPORT_W / cfg.SCALE - radius)
        centroid_y = random.uniform(radius, cfg.VIEWPORT_H / cfg.SCALE - radius)
        return {'type': 'circle', 'centroid_x': centroid_x, 'centroid_y': centroid_y, 'max_span': radius}

    def _create_obstacle_from_properties(self, properties):
        if properties['type'] == 'circle':
            body = self.world.CreateStaticBody(position=(properties['centroid_x'], properties['centroid_y']))
            circle = body.CreateCircleFixture(radius=properties['max_span'], density=1, friction=0.3)
            self.obstacles.append(circle)
        elif properties['type'] == 'rectangle':
            body = self.world.CreateStaticBody(position=(properties['centroid_x'], properties['centroid_y']))
            rectangle = body.CreatePolygonFixture(box=(properties['width'] / 2, properties['height'] / 2), density=1, friction=0.3)
            body.angle = properties['angle']
            self.obstacles.append(rectangle)
        elif properties['type'] == 'triangle':
            vertices = [(v[0] - properties['centroid_x'], v[1] - properties['centroid_y']) for v in properties['vertices']]
            body = self.world.CreateStaticBody(position=(properties['centroid_x'], properties['centroid_y']))
            triangle = body.CreatePolygonFixture(vertices=vertices, density=1, friction=0.3)
            body.angle = properties['angle']
            self.obstacles.append(triangle)

    def _create_obstacles(self, num_obstacles=None):
        cfg = self.config
        num_obstacles = num_obstacles or cfg.NUM_OBSTACLES
        obstacle_types = ['triangle', 'rectangle', 'circle']
        max_iter = 1000
        for _ in range(num_obstacles):
            obstacle_type = random.choice(obstacle_types)
            for iter in range(max_iter):
                if obstacle_type == 'triangle':
                    properties = self._generate_triangle_properties()
                elif obstacle_type == 'rectangle':
                    properties = self._generate_rectangle_properties()
                elif obstacle_type == 'circle':
                    properties = self._generate_circle_properties()
                if self._is_position_valid(properties):
                    self.obstacles_properties.append(properties)
                    self._create_obstacle_from_properties(properties)
                    break

    def _create_uav(self):
        cfg = self.config
        uav_start_pos = (cfg.UAV_RADIUS + 2 * cfg.WALL_THICKNESS, cfg.VIEWPORT_H / cfg.SCALE / 2)
        self.uav_start_pos = uav_start_pos
        self.uav = self.world.CreateDynamicBody(position=uav_start_pos, angle=cfg.UAV_INI_ANGLE, linearVelocity=(0,0), angularVelocity=0.0)
        self.uav.CreateCircleFixture(radius=cfg.UAV_RADIUS, density=cfg.UAV_DENSITY, friction=cfg.UAV_FRICTION)

    def _create_goal(self):
        cfg = self.config
        goal_pos_x = cfg.VIEWPORT_W / cfg.SCALE - cfg.GOAL_RADIUS - 2 * cfg.WALL_THICKNESS
        goal_pos_y = random.uniform(cfg.WALL_THICKNESS + cfg.UAV_RADIUS, cfg.VIEWPORT_H / cfg.SCALE - cfg.UAV_RADIUS)
        self.goal = (goal_pos_x, goal_pos_y)
        self.ini_to_goal_dist = math.sqrt((self.uav_start_pos[0] - self.goal[0])**2 + (self.uav_start_pos[1] - self.goal[1])**2)

    def _relu_penalty(self, distance, offset=None, max_penalty=None):
        cfg = self.config
        offset = offset if offset is not None else cfg.OBS_OFFSET
        max_penalty = max_penalty if max_penalty is not None else cfg.PEN_COLLISION
        adjusted_distance = (offset - distance) / (offset - cfg.UAV_RADIUS)
        relu_output = adjusted_distance / (1 + np.exp(-adjusted_distance))
        return max(relu_output * -max_penalty, 0)

    def _leakyrelu_penalty(self, distance, leaky_slope=0.001, offset=None, max_penalty=None):
        cfg = self.config
        offset = offset if offset is not None else cfg.OBS_OFFSET
        max_penalty = max_penalty if max_penalty is not None else cfg.PEN_COLLISION
        adjusted_distance = (offset - distance) / (offset - cfg.UAV_RADIUS)
        if distance < offset:
            return adjusted_distance * -max_penalty
        else:
            return leaky_slope * adjusted_distance

    def _calculate_reward(self, obs, action):
        cfg = self.config
        dist2goal = math.sqrt((self.uav.position.x - self.goal[0])**2 + (self.uav.position.y - self.goal[1])**2)
        self.dist2goal = dist2goal
        distance_reward = (1 - dist2goal / self.ini_to_goal_dist) * cfg.REW_DIST2GOAL
        velocity_reward = self.uav.linearVelocity.length * cfg.REW_VEL
        self.ang2goal = (math.atan2(self.goal[1] - self.uav.position.y, self.goal[0] - self.uav.position.x) - self.uav.angle + math.pi) % (2 * math.pi) - math.pi
        angle_reward = (1 - abs(self.uav.angle - self.ang2goal) / math.pi) * cfg.REW_ANGLE

        fov_reward = 0
        for distance in obs:
            fov_reward += self._leakyrelu_penalty(distance, offset=cfg.OBS_OFFSET, max_penalty=cfg.PEN_COLLISION) * cfg.PEN_OBSTACLE
        fov_reward /= cfg.UAV_NUM_RAYS

        if self.continuous:
            raise NotImplementedError
        else:
            act_reward = cfg.PEN_THRUST if action else 0

        return distance_reward, velocity_reward, angle_reward, fov_reward, act_reward

    def step(self, action: int):
        """
        action: int
        0: right
        1: left
        2: up
        3: down
        """
        cfg = self.config
        assert self.uav is not None, "You forgot to call reset()"
        if self.continuous:
            raise NotImplementedError
        else:
            assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid "

        if self.continuous:
            pass
        else:
            thrust = np.array([0, 0])
            if action == 0:
                thrust[0] = 1
            elif action == 1:
                thrust[0] = -1
            elif action == 2:
                thrust[1] = 1
            elif action == 3:
                thrust[1] = -1

        self.uav_pos_history.append((self.uav.position.x, self.uav.position.y))
        Front_direction = Box2D.b2Vec2(math.cos(self.uav.angle), math.sin(self.uav.angle))
        Right_direction = Box2D.b2Vec2(math.cos(self.uav.angle + math.pi / 2), math.sin(self.uav.angle + math.pi / 2))
        thrust_force = (Front_direction * thrust[0] + Right_direction * thrust[1]) * cfg.UAV_THRUST_POW

        self.uav.ApplyLinearImpulse(thrust_force, self.uav.worldCenter, True)
        self.world.Step(1.0 / cfg.FPS, 6 * 30, 2 * 30)
        new_obs = self._get_obs()
        distance_reward, velocity_reward, angle_reward, fov_reward, act_reward = self._calculate_reward(new_obs, action)
        reward = distance_reward + velocity_reward + angle_reward + fov_reward + act_reward

        done = False
        if self.game_over:
            done = True
            reward += cfg.PEN_COLLISION
        if self.dist2goal <= 2 * cfg.GOAL_RADIUS:
            done = True
            reward += cfg.REW_GOAL
            print('Goal reached! Vel: ', self.uav.linearVelocity.length)
            frame = self.render()
            plt.imshow(frame)
            plt.show()
            self.close()

        raw_reward = np.array((distance_reward/cfg.REW_DIST2GOAL, velocity_reward/cfg.REW_VEL, angle_reward/cfg.REW_ANGLE, fov_reward, act_reward))
        pos = self.uav.position
        vel = self.uav.linearVelocity

        state = np.array([
            vel.x * (cfg.VIEWPORT_W/cfg.SCALE/2) / cfg.FPS,
            vel.y * (cfg.VIEWPORT_H/cfg.SCALE/2) / cfg.FPS,
            self.uav.angle,
            distance_reward/cfg.REW_DIST2GOAL,
            self.ang2goal,
            *(1 - new_obs/cfg.UAV_FOV_DISTANCE)
        ])

        return state, reward, done, raw_reward

    def _destroy(self):
        if not self.uav:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.uav)
        self.uav = None
        for obstacle in self.obstacles:
            self.world.DestroyBody(obstacle.body)
        self.obstacles = []
        self.obstacles_properties = []
        self.uav_pos_history = []

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        self._create_walls()
        self._create_uav()
        self._create_goal()
        self._create_obstacles()
        self.game_over = False

        return self.step(0)[0]

    def _raycast_distance(self, start_pos, angle, max_distance):
        end_pos = (start_pos.x + max_distance * math.cos(angle),
                   start_pos.y + max_distance * math.sin(angle))
        class RayCastCallback(Box2D.b2RayCastCallback):
            def __init__(self):
                super(RayCastCallback, self).__init__()
                self.fixture = None
                self.point = None
                self.normal = None

            def ReportFixture(self, fixture, point, normal, fraction):
                self.fixture = fixture
                self.point = Box2D.b2Vec2(point)
                self.normal = Box2D.b2Vec2(normal)
                return fraction

        callback = RayCastCallback()
        self.world.RayCast(callback, start_pos, end_pos)
        if callback.fixture:
            hit_position = callback.point
            distance = math.sqrt((hit_position.x - start_pos.x)**2 + (hit_position.y - start_pos.y)**2)
            return distance
        else:
            return max_distance

    def _get_obs(self):
        cfg = self.config
        fov_array = np.zeros(cfg.UAV_NUM_RAYS)
        start_angle = self.uav.angle - cfg.UAV_FOV / 2
        angle_increment = cfg.UAV_FOV / cfg.UAV_NUM_RAYS
        for i in range(cfg.UAV_NUM_RAYS):
            ray_angle = start_angle + i * angle_increment
            distance = self._raycast_distance(self.uav.position, ray_angle, cfg.UAV_FOV_DISTANCE)
            fov_array[i] = distance
        return fov_array

    def render(self, mode='human'):
        cfg = self.config
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((cfg.VIEWPORT_W, cfg.VIEWPORT_H))
        self.screen.fill((255, 255, 255))
        # Draw the UAV
        for body in [self.uav]:
            for fixture in body.fixtures:
                shape = fixture.shape
                position = body.transform * shape.pos * cfg.SCALE
                position = (position[0], cfg.VIEWPORT_H - position[1])
                pygame.draw.circle(self.screen, (255, 0, 0), [int(x) for x in position], int(shape.radius * cfg.SCALE))
        # Draw the goal
        goal_position = (self.goal[0] * cfg.SCALE, cfg.VIEWPORT_H - self.goal[1] * cfg.SCALE)
        pygame.draw.circle(self.screen, (0, 255, 0), [int(x) for x in goal_position], int(cfg.GOAL_RADIUS * cfg.SCALE))
        # Draw obstacles
        for fixture in self.obstacles:
            shape = fixture.shape
            body = fixture.body
            if isinstance(shape, Box2D.b2CircleShape):
                position = (body.position.x * cfg.SCALE, cfg.VIEWPORT_H - body.position.y * cfg.SCALE)
                pygame.draw.circle(self.screen, (0, 0, 255), [int(x) for x in position], int(shape.radius * cfg.SCALE))
            elif isinstance(shape, Box2D.b2PolygonShape):
                vertices = [(body.transform * v) * cfg.SCALE for v in shape.vertices]
                vertices = [(v[0], cfg.VIEWPORT_H - v[1]) for v in vertices]
                pygame.draw.polygon(self.screen, (0, 0, 255), vertices)
        wall_color = (0, 0, 0)
        for wall in self.walls:
            for fixture in wall.fixtures:
                shape = fixture.shape
                vertex1, vertex2 = shape.vertices
                vertex1 = (wall.transform * vertex1) * cfg.SCALE
                vertex2 = (wall.transform * vertex2) * cfg.SCALE
                vertex1 = (vertex1[0], cfg.VIEWPORT_H - vertex1[1])
                vertex2 = (vertex2[0], cfg.VIEWPORT_H - vertex2[1])
                pygame.draw.line(self.screen, wall_color, vertex1, vertex2, int(cfg.WALL_THICKNESS * cfg.SCALE))
        start_angle = self.uav.angle - cfg.UAV_FOV / 2
        angle_increment = cfg.UAV_FOV / cfg.UAV_NUM_RAYS
        for i in range(cfg.UAV_NUM_RAYS):
            ray_angle = start_angle + i * angle_increment
            distance = self._raycast_distance(self.uav.position, ray_angle, cfg.UAV_FOV_DISTANCE)
            end_x = self.uav.position.x + distance * math.cos(ray_angle)
            end_y = self.uav.position.y + distance * math.sin(ray_angle)
            pygame.draw.line(self.screen, (0, 255, 0),
                             (self.uav.position.x * cfg.SCALE, cfg.VIEWPORT_H - self.uav.position.y * cfg.SCALE),
                             (end_x * cfg.SCALE, cfg.VIEWPORT_H - end_y * cfg.SCALE))
        if self.uav:
            velocity_vector = self.uav.linearVelocity
            uav_center = (self.uav.position.x * cfg.SCALE, cfg.VIEWPORT_H - self.uav.position.y * cfg.SCALE)
            velocity_end = (uav_center[0] + velocity_vector.x, uav_center[1] - velocity_vector.y)
            pygame.draw.line(self.screen, (0, 0, 0), uav_center, velocity_end, 2)
        if len(self.uav_pos_history) > 1:
            dash_length = 5
            draw = True
            segment_length = 0
            for i in range(len(self.uav_pos_history) - 1):
                start_pos = self.uav_pos_history[i]
                end_pos = self.uav_pos_history[i + 1]
                start_pos_pygame = (start_pos[0] * cfg.SCALE, cfg.VIEWPORT_H - start_pos[1] * cfg.SCALE)
                end_pos_pygame = (end_pos[0] * cfg.SCALE, cfg.VIEWPORT_H - end_pos[1] * cfg.SCALE)
                segment_length += math.sqrt((start_pos_pygame[0] - end_pos_pygame[0])**2 + (start_pos_pygame[1] - end_pos_pygame[1])**2)
                if segment_length > dash_length:
                    segment_length = 0
                    draw = not draw
                if draw:
                    pygame.draw.line(self.screen, (255, 0, 0), start_pos_pygame, end_pos_pygame, 2)
        pygame.display.flip()
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
            self.isopen = False

def capture_pygame_surface(surface):
    image = pygame.surfarray.array3d(surface)
    image = np.transpose(image, (1, 0, 2))
    return image
