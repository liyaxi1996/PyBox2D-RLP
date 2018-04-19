#!/usr/bin/env python
# -*- coding: utf-8 -*-

from examples.framework import (Framework, Keys, main)
from math import sqrt
from math import pi 
import numpy as np 
import tensorflow as tf
from Box2D import (b2FixtureDef, b2PolygonShape,
                   b2Transform, b2Mul,
                   b2_pi)


class InvertedPendulum(Framework):
    name = "InvertedPendulum"
    description = "Multi Inverted Pendulum Control Framework."

    def __init__(self):
        super(InvertedPendulum, self).__init__()
        
        # basic info of the InvertedPendulum world
        self.world.gravity = (0.0, -9.8)
        self.num_of_pendulum = 2
        self.action_space = 2 # left and right 
        self.force_mag = 10.0
        self.masscart = 1.0
        self.masspole = 0.1
        self.volcart = (3, 2)
        self.length = 4
        self.volpole = (0.1, self.length)
        self.poscart = (0, 2.01)
        self.istrain = settings.istrain
        self.observation_space = ((self.num_of_pendulum + 1) * 2,)

        # angle at which this simulation will fail 
        self.theta_threshold_radians = 12 * 3 * pi / 360
        self.x_threshold = 50

        # The boundaries
        ground = self.world.CreateBody(position=(0, 0), shapes=b2PolygonShape(box=(50, 0.01)),) 

        self.car = None
        self.pendulum = []
        self.CreateInvertedPendulum()

    def CreateInvertedPendulum(self):
        self.car = self.world.CreateDynamicBody(
            position= self.poscart,
            shapes=b2PolygonShape(box=self.volcart),
            shapeFixture=b2FixtureDef(density=self.masscart),
        )
        self.pendulum = []
        for i in range(self.num_of_pendulum):
            self.CreatePendulum(i)

    def CreatePendulum(self,i):
        fixtures = b2FixtureDef(shape=b2PolygonShape(box=self.volpole),density=self.masspole)
        self.pendulum.append(self.world.CreateDynamicBody(fixtures=fixtures,angle = 0,position =  (0,12+8 * i )))  
        # create Joint. i == 0 for the joint beween pendulum and car. others for joints beween pendulums
        self.world.CreateRevoluteJoint(
            bodyA=self.car if i == 0 else self.pendulum[i-1],
            bodyB=self.pendulum[i],
            localAnchorA=(0, 2) if i == 0 else (0,4),
            localAnchorB=(0,-4),
            collideConnected=False,
        )

    def DestoryInvertedPendulum(self):
        self.world.DestroyBody(self.car)
        for pendulum in self.pendulum:
            self.world.DestroyBody(pendulum)

    def Reset(self):
        self.DestoryInvertedPendulum()
        self.CreateInvertedPendulum()
        state = [self.car.position[0],self.car.linearVelocity[0]]
        state.extend([f(x) for x in self.pendulum for f in (lambda x:x.angle,lambda x : x.angularVelocity)])
        return np.array(state)
    
    def Force(self, action):
        force = (self.force_mag if action = 1 else -self.force_mag, 0 )
        f = self.car.GetWorldVector(localVector = force)
        p = self.car.GetWorldVector(localPoint=self.poscart)
        self.car.ApplyForce(f, p, True)
    
    def Step(self, action, settings = settings):
        # TODO 1. use DRL to control our car
        # TODO 2. use classic control method
        # TODO Step1 get position get angle
        # TODO postion - > algorithm (dqn pid) -> +1 -1
        # if 1
        self.Force(action)
        Framework.Step(self, settings)
        done = self.car.position[0] < -self.x_threshold \
               or self.car.position[0] > self.x_threshold
        for angle in [x.angle for x in self.pendulum]:
        	done  = done \
                    or angle < -self.theta_threshold_radians \
                    or angle > self.theta_threshold_radians
        state = [self.car.position[0],self.car.linearVelocity[0]]
        state.extend([f(x) for x in self.pendulum for f in (lambda x:x.angle,lambda x : x.angularVelocity)])
        if done:
            reward = 0.0    
            self.reset()
        else:
            reward = 1.0 
        return np.array(state), reward, done, {}

if __name__ == "__main__":
    main(InvertedPendulum)

   
