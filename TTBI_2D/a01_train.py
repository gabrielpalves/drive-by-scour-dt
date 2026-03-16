import numpy as np

class EmptyObj:
    pass

def train_prop_obrien_calibrate(Train, veh_num, x):
    """
    Irish Rail Hyundai Rotem InterCity fleet (Obrien et al., 2018)
    """
    var_BodyMass = (x[0] * 0.10 * 36852)
    var_Prim_k = (x[1] * 0.05 * (2779e3 * 2))
    var_Sec_k = (x[2] * 0.05 * (1000e3 * 2))
    
    Train.Veh[veh_num].Body = EmptyObj()
    Train.Veh[veh_num].Body.m = 36852 + var_BodyMass
    Train.Veh[veh_num].Body.I = 560342
    Train.Veh[veh_num].Body.L = 8 * 2
    Train.Veh[veh_num].Body.Le = np.array([1, 1]) * (1.5 + 3.0 / 2.0)
    
    Train.Veh[veh_num].Bogie = EmptyObj()
    Train.Veh[veh_num].Bogie.num = 2
    Train.Veh[veh_num].Bogie.m = np.array([1, 1]) * 3910
    Train.Veh[veh_num].Bogie.I = np.array([1, 1]) * 10024
    Train.Veh[veh_num].Bogie.L = np.array([1, 1]) * 2.3
    
    Train.Veh[veh_num].Wheels = EmptyObj()
    Train.Veh[veh_num].Wheels.num = 4
    Train.Veh[veh_num].Wheels.m = np.array([1, 1, 1, 1]) * 1407
    
    Train.Veh[veh_num].Susp = EmptyObj()
    Train.Veh[veh_num].Susp.Prim = EmptyObj()
    Train.Veh[veh_num].Susp.Prim.k = np.array([1, 1, 1, 1]) * 2779e3 * 2 + var_Prim_k
    Train.Veh[veh_num].Susp.Prim.c = np.array([1, 1, 1, 1]) * 29.4e3 * 2
    
    Train.Veh[veh_num].Susp.Sec = EmptyObj()
    Train.Veh[veh_num].Susp.Sec.k = np.array([1, 1]) * 1000e3 * 2 + var_Sec_k
    Train.Veh[veh_num].Susp.Sec.c = np.array([1, 1]) * 60e3 * 2
    
    return Train

def a01_train(velocidade, x):
    """
    Definitions for Train
    """
    Train = EmptyObj()
    Train.Load = EmptyObj()
    Train.Load.path = ''
    
    num_vehicles = x.shape[0]
    Train.Veh = [EmptyObj() for _ in range(num_vehicles)]
    
    for i in range(num_vehicles):
        Train = train_prop_obrien_calibrate(Train, i, x[i, :])
        
    # Copy of Train velocity
    Train.Veh[0].vel = velocidade
    Train.vel = velocidade
    
    return Train