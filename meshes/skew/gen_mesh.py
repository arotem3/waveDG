import numpy as np
import matplotlib.pyplot as plt

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

n = 20
th = np.pi / 4
r = np.sqrt(0.5)
c = np.cos(th)
s = np.sin(th)
alpha = (c + r*s - r)/c

g = cfg.Geometry()
g.point([0, 0]) # 0
g.point([c, -s])
g.point([c+r*s, -s+r*c])
g.point([ r,  -s+r*c+alpha*s]) # 3
g.point([ r,  1])
g.point([ 0, 1])

g.line([0, 1], el_on_curve=n)
g.line([1, 2], el_on_curve=n)
g.line([2, 3], el_on_curve=n)
g.line([3, 0], el_on_curve=n)

g.line([0, 5], el_on_curve=n)
g.line([5, 4], el_on_curve=n)
g.line([4, 3], el_on_curve=n)

g.struct_surf([0, 1, 2, 3])
g.struct_surf([3, 4, 5, 6])

# cfv.figure(fig_size=(10,10))
# cfv.draw_geometry(g, label_points=True, label_curves=True, draw_points=True)
# plt.show()

mesh = cfm.GmshMesh(g)

mesh.elType = 3
mesh.dofsPerNode = 1
coo, edof, dofs, bdofs, elmarkers = mesh.create()

cfv.figure(fig_size=(10, 10))
cfv.draw_mesh(
    coords=coo,
    edof=edof,
    dofs_per_node=mesh.dofsPerNode,
    el_type=mesh.elType,
    filled=True
)
plt.show()

edof = np.array(edof, dtype=int)-1
J = np.unique(edof)
Ji = np.zeros(len(coo), dtype=int)
for i, j in enumerate(J):
    Ji[j] = i

coo1 = coo[J]
edof1 = np.zeros(edof.shape, dtype=int)
for e, el in enumerate(edof):
    for j, el_j in enumerate(el):
        edof1[e,j] = Ji[el_j]

np.savetxt("coordinates.txt", coo1)
np.savetxt("elements.txt", edof1, fmt='%d')

info = open("info.txt",'w')
info.write("%d %d"%(len(coo1), len(edof1)))
info.close()

print('done')