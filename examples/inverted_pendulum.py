#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version Copyright (c) 2010 kne / sirkne at gmail dot com
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions: 
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

from examples.framework import (Framework, Keys, main)
from math import sqrt
from math import pi 
from Box2D import (b2FixtureDef, b2PolygonShape,
                   b2Transform, b2Mul,
                   b2_pi)


class InvertedPendulum(Framework):
    name = "InvertedPendulum"
    description = "Multi Inverted Pendulum Control Framework."

    def __init__(self):
        super(InvertedPendulum, self).__init__()
        self.theta_threshold_radians = 12 * 3 * pi / 360
        self.x_threshold = 40
        self.world.gravity = (0.0, -9.8)

        # The boundaries
        ground = self.world.CreateBody(position=(0, 0), shapes=b2PolygonShape(box=(50, 0.01)),) 

        self.car = None
        self.pendulum = []
        self.CreateInvertedPendulum()

    def CreateInvertedPendulum(self):
        self.car = self.world.CreateDynamicBody(
            position=(0, 2.01),
            shapes=b2PolygonShape(box=(3, 2)),
            shapeFixture=b2FixtureDef(density=9.0),
        )
        self.pendulum = []
        for i in range(2):
            self.CreatePendulum(i)

    def CreatePendulum(self,i):
        fixtures = b2FixtureDef(shape=b2PolygonShape(box=(0.1, 4)),density=0.1)
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

    def reset(self):
        self.DestoryInvertedPendulum()
        self.CreateInvertedPendulum()

    def Step(self, settings):
        # TODO 1. use DRL to control our car
        # TODO 2. use classic control method
        # TODO Step1 get position get angle
        # TODO postion - > algorithm (dqn pid) -> +1 -1
        # if 1
        f = self.car.GetWorldVector(localVector=(-12000.0, 0))
        p = self.car.GetWorldPoint(localPoint=(3, 0))
        self.car.ApplyForce(f, p, True)

        Framework.Step(self, settings)
        
        done = self.car.position[0] < -self.x_threshold \
               or self.car.position[0] > self.x_threshold
        for angle in [x.angle for x in self.pendulum]:
        	done  = done \
                    or angle < -self.theta_threshold_radians \
                    or angle > self.theta_threshold_radians
        if done:
            self.reset()

if __name__ == "__main__":
    main(InvertedPendulum)

   
