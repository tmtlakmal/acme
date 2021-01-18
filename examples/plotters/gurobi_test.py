import gurobipy as gp
from gurobipy import GRB
import numpy as np


# Create a new model
m = gp.Model("mip1")

scheduled_time = 23.0
delta_t = 0.2
distance = 400
N = int(scheduled_time/delta_t)

max_velocity = 22.0
# Create variables
positions = m.addVars(1, N, lb=-distance, ub=0.0, vtype='C', name='p')
velocity = m.addVars(1, N, lb=0.0, ub=max_velocity, vtype='C', name='v')
acceleration = m.addVars(1, N, lb=-1, ub=1, vtype='I', name='a')

for i in range(N-1):
    m.addConstr(positions[0, i+1] == positions[0, i] + (velocity[0, i] + velocity[0, i+1])*delta_t/2)
    m.addConstr(velocity[0, i+1] == velocity[0, i] + acceleration[0, i]*2*delta_t)

m.addConstr(positions[0, 0] == -400)
m.addConstr(positions[0, N-1] == 0)
m.addConstr(velocity[0, 0] == 6.4)
m.addConstr(velocity[0, N-1] <= max_velocity)
m.addConstr(velocity[0, N-1] >= max_velocity-4)

m.setObjective(positions.sum(), GRB.MAXIMIZE)

#x = m.addVar(vtype='B', name="x")
#y = m.addVar(vtype=GRB.BINARY, name="y")
#z = m.addVar(vtype=GRB.BINARY, name="z")
#
#
#
#
## Set objective
#m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
#
## Add constraint: x + 2 y + 3 z <= 4
#m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
#
## Add constraint: x + y >= 1
#m.addConstr(x + y >= 1, "c1")
#
## Optimize model
m.optimize()

print("done")
for v in m.getVars():
    print('%s %g' % (v.varName, v.x))

data = np.zeros(len(velocity), dtype=np.float)
for i in range(len(velocity)):
    data[i] = velocity[0,i].x

print('Obj: %g' % (m.objVal/400))
