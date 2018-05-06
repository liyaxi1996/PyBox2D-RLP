#!/usr/bin/env python
# -*- coding: utf-8 -*-

from examples.framework import (Framework, Keys, main)
import numpy as np
from Box2D import (b2FixtureDef, b2PolygonShape,b2_pi)
from random import randint, random

class InvertedPendulum(Framework):
    name = "InvertedPendulum"
    description = "Multi Inverted Pendulum Control Framework."

    def __init__(self):
        super(InvertedPendulum, self).__init__()
        # basic info of the InvertedPendulum world
        self.world.gravity = (0.0, -9.8)
        self.num_of_pendulum = 1
        self.action_space = 2  # left and right
        self.force_mag = 30.0
        self.volcart = (3, 2)
        self.length = 8.3
        self.volpole = (0.3, self.length)
        self.masscart = 1.0 / (self.volcart[0] * self.volcart[1] * 4)
        self.masspole = 0.1 / (self.volpole[0] * self.volpole[1] * 4)
        self.poscart = (0, 2.01)
        self.observation_space = ((self.num_of_pendulum + 1) * 2,)
        self.force_pos = (3,0)
        self.friction = 0
        self.theta_threshold_radians = 12 * 2 * b2_pi / 360
        self.x_threshold = 40
        self._elapsed_steps = None
        self.step_beyond_done = None
        self.max_steps = 200

        # The boundaries
        self.world.CreateStaticBody(position=(0, 0), shapes=b2PolygonShape(box=(50, 0.01)), shapeFixture = b2FixtureDef(friction = self.friction))
        self.car = None
        self.pendulum = []
        self.CreateInvertedPendulum()

    def CreateInvertedPendulum(self):
        self._elapsed_steps = 0
        self.car = self.world.CreateDynamicBody(
            position= self.poscart,
            shapes=b2PolygonShape(box=self.volcart),
            shapeFixture=b2FixtureDef(density=self.masscart,friction = self.friction),
        )
        self.pendulum = []
        for i in range(self.num_of_pendulum):
            self.CreatePendulum(i)

    def CreatePendulum(self,i):
        fixtures = b2FixtureDef(shape=b2PolygonShape(box=self.volpole),density=self.masspole,friction = self.friction)
        self.pendulum.append(self.world.CreateDynamicBody(fixtures=fixtures,angle = 0,position =  (0,12+8 * i )))  
        # create Joint. i == 0 for the joint beween pendulum and car. others for joints beween pendulums
        self.world.CreateRevoluteJoint(
            bodyA=self.car if i == 0 else self.pendulum[i-1],
            bodyB=self.pendulum[i],
            localAnchorA=(0, 1) if i == 0 else (0,self.length),
            localAnchorB=(0,-self.length),
            collideConnected=False,
        )

    def DestoryInvertedPendulum(self):
        self.world.DestroyBody(self.car)
        for pendulum in self.pendulum:
            self.world.DestroyBody(pendulum)

    def Reset(self):
        self.DestoryInvertedPendulum()
        self.CreateInvertedPendulum()
        self._elapsed_steps = 0
        self.step_beyond_done = None
        self.Step(action=randint(0,1), rand=random())
        state = self.State()
        return np.array(state)
    
    def Force(self, action, rand):
        if rand:
            force = (self.force_mag * rand if action == 1 else -self.force_mag * rand, 0)
        else:
            force = (self.force_mag if action == 1 else -self.force_mag, 0 )
        f = self.car.GetWorldVector(localVector = force)
        p = self.car.GetWorldPoint(localPoint=self.force_pos)
        self.car.ApplyForce(f, p, True)

    def State(self):
        state = [self.car.position[0], self.car.linearVelocity[0]]
        state.extend([f(x) for x in self.pendulum for f in (lambda x: x.angle, lambda x: x.angularVelocity)])
        return state

    def Step(self, action=None, rand=None):
        self._elapsed_steps += 1
        if (self.settings.istrain or self.settings.isinference):
            self.Force(action, rand)
        Framework.Step(self, self.settings)
        done = self.car.position[0] < -self.x_threshold \
               or self.car.position[0] > self.x_threshold
        for angle in [x.angle for x in self.pendulum]:
            done = done or angle < -self.theta_threshold_radians \
                   or angle > self.theta_threshold_radians
        state = self.State()
        done = bool(done)
        if not done:
            reward = 1.0
        elif self.step_beyond_done is None:
            self.step_beyond_done = 0
            reward = 1.0
        else:
            if self.step_beyond_done == 0:
                print("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.step_beyond_done += 1
            reward = 0
        if self._elapsed_steps >= self.max_steps:
            _ = self.Reset()
            done = True
        return np.array(state), reward, done, {}

if __name__ == "__main__":
    main(InvertedPendulum)
