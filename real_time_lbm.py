from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import numexpr as ne


name = 'LBM - D2Q9'


def get_cmap_from_file(filename='cmap.dat'):
    f = open(filename, 'r')
    ncol = int(f.readline())
    values = f.read()
    f.close()
    
    rgb = values.split('\n')
    rgb.pop()
    rgb.pop()
    cmap_rgba = []
    for val in rgb:
        r,g,b = val.split()
        cmap_rgba.append(int(255.0) << 24 | 
            (int(float(r) * 255.0) << 16) | 
            (int(float(g) * 255.0) << 8 ) | 
            (int(float(b) * 255.0) << 0 ) )
    
    return np.array(cmap_rgba), len(cmap_rgba)


def get_cmap_from_matplotlib(cmap=cm.Reds):
    ncol = cmap.N
    cmap_rgba = []
    for i in xrange(ncol):
        b, g, r, _ = cmap(i) # Not sure why this is inverted, I was expecting r, g, b.
        cmap_rgba.append(int(255.0) << 24 | 
            (int(float(r) * 255.0) << 16) | 
            (int(float(g) * 255.0) << 8 ) | 
            (int(float(b) * 255.0) << 0 ) )

    return np.array(cmap_rgba), len(cmap_rgba)


###### Flow definition #########################################################
ni = 300
nj = 100
Re = 220.0  # Reynolds number.
nx, ny = ni, nj; ly=ny-1.0; q = 9 # Lattice dimensions and populations.
cx = nx/4; cy=ny/2; r=ny/9;          # Coordinates of the cylinder.
uLB = 0.04                       # Velocity in lattice units.
nulb = uLB*r/Re; omega = 1.0 / (3.0*nulb+0.5); # Relaxation parameter.

###### OpenGL Properties #######################################################
solid = np.ones((nj, ni))
plot_rgba = np.zeros(ni*nj, dtype=np.uint32)
cmap_rgba, ncol = get_cmap_from_matplotlib()
#cmap_rgba = get_cmap_from_file(filename='cmap.dat')

###### Lattice Constants #######################################################
c = np.array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) # Lattice velocities.
t = 1.0/36.0 * np.ones(q)                                   # Lattice weights.

t[np.asarray([np.linalg.norm(ci)<1.1 for ci in c])] = 1.0 / 9.0
t[0] = 4.0 / 9.0

noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)] 
i1 = np.arange(q)[np.asarray([ci[0]<0  for ci in c])] # Unknown on right wall.
i2 = np.arange(q)[np.asarray([ci[0]==0 for ci in c])] # Vertical middle.
i3 = np.arange(q)[np.asarray([ci[0]>0  for ci in c])] # Unknown on left wall.

###### Function Definitions ####################################################
sumpop = lambda fin: np.sum(fin,axis=0) # Helper function for density computation.

def equilibrium(rho, u):              # Equilibrium distribution function.    
    # No multi-thread
    #cu = 3.0 * np.einsum('ij,jkl', c, u)    
    # Multi-thread
    cu = 3.0 * np.tensordot(c, u, axes=([1],[0]))    
    
    u0, u1 = u[0], u[1]
    usqr = ne.evaluate("3.0 / 2.0 * (u0**2 + u1**2)")
        
    feq = (ne.evaluate("rho * (1.0 + cu + 0.5 * (cu ** 2) - usqr)").T * t).T    
    return feq

###### Setup: cylindrical obstacle and velocity inlet with perturbation ########
circle_obstacle = np.fromfunction(lambda x,y: (x-cx)**2+(y-cy)**2<r**2, (nx,ny))
vel = np.fromfunction(lambda d,x,y: (1-d)*uLB*(1.0+1e-4*np.sin(y/ly*2*np.pi)),(2,nx,ny))
feq = equilibrium(np.ones((nx,ny)),vel); fin = feq.copy()


def solve_lbm(obstacle):  
    fin[i1,-1,:] = fin[i1,-2,:] # Right wall: outflow condition.

    rho = sumpop(fin)           # Calculate macroscopic density and velocity.

    # No multi-thread
    #u = np.einsum('ij,ikl', c, fin) / rho
    # Multi-thread
    u = np.tensordot(c, fin, axes=([0],[0])) / rho        

    u[:,0,:] = vel[:,0,:] # Left wall: compute density from known populations.
    rho[0,:] = 1.0 / (1.0 - u[0,0,:]) * (sumpop(fin[i2,0,:]) + 2.0 * sumpop(fin[i1,0,:]))

    feq = equilibrium(rho,u) # Left wall: Zou/He boundary condition.
    fin[i3,0,:] = fin[i1,0,:] + feq[i3,0,:] - fin[i1,0,:]

    # Collision step
    #fout = fin - omega * (fin - feq)  
    fout = ne.evaluate("fin - omega * (fin - feq)")

    for i in range(q): 
        fout[i, obstacle] = fin[noslip[i], obstacle]

    for i in range(q): # Streaming step.
        fin[i,:,:] = np.roll(np.roll(fout[i,:,:], c[i,0], axis=0), c[i,1], axis=1)

    v_sqrt = np.sqrt(u[0]**2 + u[1]**2).T

    return v_sqrt.flatten()


def display():
    '''
    Set upper and lower limits for plotting.
    Do one Lattice Boltzmann step: stream, BC, collide.
    
    '''   

    #obstacle = circle_obstacle
    obstacle = (1 - solid.T).astype(np.bool)
    plotvar = solve_lbm(obstacle=obstacle)
    
    minvar = 0.0
    maxvar = 1.1 * np.max(plotvar)
    
    # convert the plotvar array into an array of colors to plot
    # if the mesh point is solid, make it black
    frac = (plotvar[:] - minvar)/(maxvar - minvar)
    icol = frac * ncol    
    plot_rgba[:] = solid.flatten().astype(np.int) * cmap_rgba[icol.astype(np.int)] 
    
    # Fill the pixel buffer with the plot_rgba array
    glBufferData(GL_PIXEL_UNPACK_BUFFER, plot_rgba.nbytes, plot_rgba, GL_STREAM_COPY)

    # Copy the pixel buffer to the texture, ready to display
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,ni,nj,GL_RGBA,GL_UNSIGNED_BYTE, None)
    
    # Render one quad to the screen and colour it using our texture
    # i.e. plot our plotvar data to the screen
    glClear(GL_COLOR_BUFFER_BIT)
    glBegin(GL_QUADS)
    
    x0, y0 = 0.0, 0.0
    x1, y1 = ni, nj
    
    glTexCoord2f(0.0, 0.0)
    glVertex3f(x0, y0, 0.0)
    
    glTexCoord2f(1.0, 0.0)
    glVertex3f(x1, y0, 0.0)
    
    glTexCoord2f (1.0, 1.0)
    glVertex3f(x1, y1, 0.0)
    
    glTexCoord2f (0.0, 1.0)
    glVertex3f(x0, y1, 0.0)
    
    glEnd()
    glutSwapBuffers()     


def resize(w,h):
    '''
    GLUT resize callback to allow us to change the window size.
    
    '''
    global width, height
    width = w
    height = h
    glViewport (0, 0, w, h)
    glMatrixMode (GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0., ni, 0., nj, -200. ,200.)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    
def mouse(button, state, x, y):    
    global draw_solid_flag, ipos_old, jpos_old
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:       
        draw_solid_flag = 0
        xx = x
        yy = y
        ipos_old = int(float(xx) / width * float(ni))
        jpos_old = int(float(height - yy) / height * float(nj))
        
    if button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN:        
        draw_solid_flag = 1
        xx = x
        yy = y
        ipos_old = int(float(xx) / width * float(ni))
        jpos_old = int(float(height - yy) / height * float(nj))


def mouse_motion(x,y):
    '''
    GLUT call back for when the mouse is moving
    This sets the solid array to draw_solid_flag as set in the mouse callback
    It will draw a staircase line if we move more than one pixel since the
    last callback - that makes the coding a bit cumbersome:
    '''
    global ipos_old, jpos_old
    
    xx = x
    yy = y
    ipos = int(float(xx) / width * float(ni))
    jpos = int(float(height-yy) / height * float(nj))

    if ipos <= ipos_old:
        i1 = ipos
        i2 = ipos_old
        j1 = jpos
        j2 = jpos_old
    
    else:
        i1 = ipos_old
        i2 = ipos
        j1 = jpos_old
        j2 = jpos
    
    jlast = j1   
    
    for i in xrange(i1, i2+1):
        if i1 != i2:
            frac = (i-i1) / (i2-i1)
            jnext = (frac * (j2-j1)) + j1        
        else:
            jnext=j2
        
        if jnext >= jlast:
            solid[jlast,i] = draw_solid_flag
            
            for j in xrange(jlast, jnext+1):    
                solid[j,i] = draw_solid_flag
        else:
            solid[jlast,i] = draw_solid_flag
            for j in xrange(jnext, jlast+1):
                solid[j,i] = draw_solid_flag
        
        jlast = jnext
        
    ipos_old=ipos
    jpos_old=jpos

def run_opengl():
    glutInit(name)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(ni, nj)
    glutInitWindowPosition(50, 50)
    glutCreateWindow(name)

    glClearColor(1.0,1.0,1.0,1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0,ni,0.,nj, -200.0, 200.0)

    glEnable(GL_TEXTURE_2D)

    gl_Tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, gl_Tex)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ni, nj, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

    gl_PBO = glGenBuffers(1)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO)

    #setup callbacks
    glutDisplayFunc(display)
    glutReshapeFunc(resize)
    glutIdleFunc(display)
    glutMouseFunc(mouse)
    glutMotionFunc(mouse_motion)

    glutMainLoop()
    
if __name__ == "__main__":
    run_opengl()