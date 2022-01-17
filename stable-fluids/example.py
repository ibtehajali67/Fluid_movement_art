import numpy as np
from PIL import Image
from scipy.special import erf
import random
from fluid import Fluid
import cv2
RESOLUTION = 1024,1024
DURATION = 300
count=500
b=1
s=185
for i in range(1):
    INFLOW_PADDING = random.randint(2,5)
    # INFLOW_PADDING = 0
    INFLOW_DURATION = random.randint(299,300)
    INFLOW_RADIUS = random.randint(7,10)
    INFLOW_VELOCITY = 1
    INFLOW_COUNT = random.randint(50,60)

    print('Generating fluid solver, this may take some time.')
    fluid = Fluid(RESOLUTION, 'dye')

    center = np.floor_divide(RESOLUTION, 2)
    r = np.min(center) - INFLOW_PADDING
    points = np.linspace(-np.pi, np.pi, INFLOW_COUNT, endpoint=True)
    

    # m=[np.sin(p),np.cos(p),np.tan(p),np.arccos(p),np.arcsin(p),np.arctan(p)]
        # points = tuple(np.array(random.uniform(m[0],m[5]),random.uniform(m[0],m[5])))
    pt=list()
    for p in points:
        m=[np.sin(p),np.cos(p),np.tan(p),np.arccos(p),np.arcsin(p),np.arctan(p)]

        points = (random.choice(m),random.choice(m))
        pt.append(np.array(points))


    # points = tuple(np.array((np.cos(p), np.sin(p))) for p in points)
    points=tuple(pt)
    normals = tuple(-p for p in points)
    points = tuple(random.randint(0,r) * p + center for p in points)
    inflow_velocity = np.zeros_like(fluid.velocity)
    inflow_dye = np.zeros(fluid.shape)
    for p, n in zip(points, normals):
        mask = np.linalg.norm(fluid.indices - p[:, None, None], axis=0) <= INFLOW_RADIUS
        inflow_velocity[:, mask] += n[:, None] * INFLOW_VELOCITY
        inflow_dye[mask] = 1

    frames = []
    for f in range(DURATION):
        print(f'Computing frame {f + 1} of {DURATION}.')
        if f <= INFLOW_DURATION:
            fluid.velocity += inflow_velocity
            fluid.dye += inflow_dye

        curl = fluid.step()[0]
        # print(curl)
    # Using the error function to make the contrast a bit higher. 
        curl = (erf(curl * 2) + 1) /4
        # print(curl)
        color = np.dstack((curl, np.ones(fluid.shape), fluid.dye))
        # print(color)
        color = (np.clip(color, 0, 1) * s).astype('uint8')
        
        color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)
        if(b>290):
            cv2.imwrite('h_frame_result/example_1'+str(count)+str(b)+'_.jpg',color)
        b+=1
        frames.append(Image.fromarray(color))
    print('Saving simulation result.')
    frames[0].save('gif_hd/example_'+str(count)+'.gif', save_all=True, append_images=frames[1:], duration=20, loop=0)
    fre=frames[-1]
    fre=np.array(fre)
    cv2.imwrite('h_last/example_'+str(count)+'.jpg',fre)
    b=1
    count+=1
    s+=5
    if(s>255):
        s=185
