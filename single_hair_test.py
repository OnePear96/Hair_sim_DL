import os
import numpy as np
import pyflex
import time
import math as m


pyflex.init()
scene_params = np.array([0.1,1.5])


pyflex.set_scene(12, scene_params, 0)

N_particles = pyflex.get_n_particles()
N_hairs = 1
N_particles_per_hair = N_particles // N_hairs
print ("num particles: {},\n num hairs: {},\n num particles per hair: {}".format(N_particles,N_hairs,N_particles_per_hair))

'''
How to get index of particles for each hair?
For hair number j, j in [0,N_hairs)
the first particle index of this hair: j*N_particles_per_hair, last: (j+1)*N_particles_per_hair-1
index to get the position : idx_particle*4
'''

p = np.array([[0.        , 4.        , 0.        , 0.        ],
       [0.        , 3.88348913, 0.        , 0.        ],
       [0.        , 3.76832294, 0.        , 0.        ],
       [0.        , 3.64844704, 0.        , 0.        ],
       [0.        , 3.53010917, 0.        , 0.        ],
       [0.        , 3.4117341 , 0.        , 0.        ],
       [0.        , 3.29377675, 0.        , 0.        ],
       [0.        , 3.17614532, 0.        , 0.        ],
       [0.        , 3.05889988, 0.        , 0.        ],
       [0.        , 2.94205379, 0.        , 0.        ],
       [0.        , 2.82562876, 0.        , 0.        ],
       [0.        , 2.70964026, 0.        , 0.        ],
       [0.        , 2.59410095, 0.        , 0.        ],
       [0.        , 2.47902036, 0.        , 0.        ],
       [0.        , 2.36440516, 0.        , 0.        ],
       [0.        , 2.25025892, 0.        , 0.        ],
       [0.        , 2.13658237, 0.        , 0.        ],
       [0.        , 2.02337432, 0.        , 0.        ],
       [0.        , 1.91063178, 0.        , 0.        ],
       [0.        , 1.79834974, 0.        , 0.        ],
       [0.        , 1.68652236, 0.        , 0.        ],
       [0.        , 1.57514274, 0.        , 0.        ],
       [0.        , 1.46420372, 0.        , 0.        ],
       [0.        , 1.35369825, 0.        , 0.        ],
       [0.        , 1.24361885, 0.        , 0.        ],
       [0.        , 1.13395953, 0.        , 0.        ],
       [0.        , 1.02471292, 0.        , 0.        ],
       [0.        , 0.91587913, 0.        , 0.        ],
       [0.        , 0.80743492, 0.        , 0.        ],
       [0.        , 0.6994499 , 0.        , 0.        ],
       [0.        , 0.5916518 , 0.        , 0.        ]])

startTime = 3.0
translationSpeed = 2.0

for i in range(500):
    t = i/60
    time = max(0,t-startTime)
    last_time = max(0,t-startTime-1/60)
    x = translationSpeed*(0.5*m.cos(time))
    last_x = translationSpeed*(0.5*m.cos(last_time))
    if i == 0:
        print ("number of shapes: ",pyflex.get_n_shapes())
      #  print ("shape of get positions: ", pyflex.get_positions().shape)
     #   print ("number of rigid: ", pyflex.get_n_rigids())
        print ("position of rigid: ", pyflex.get_rigidLocalPositions())
        print ('state of shapes: ', pyflex.get_shape_states())
     #   pos = pyflex.get_positions()
        # get the begin of each hair
    s_gt = np.array([x, 2.0, 0.0, last_x, 2.0, 0.0, 0.0, 0.70710677, 0.0, 0.70710677, 0.0, 0.70710677, 0.0, 0.70710677])
    pyflex.set_shape_states(s_gt)
    pyflex.set_positions(p)
    pyflex.render()
 #   shape_state = pyflex.get_shape_states()
#    shape_position = shape_state[:3].reshape(-1,3)
 #   print ("shape position: ", shape_position)
 #   pyflex.step(capture=0, path='/')
    

pyflex.clean()