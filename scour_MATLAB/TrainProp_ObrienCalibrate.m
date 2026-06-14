function Train = TrainProp_ObrienCalibrate(Train, veh_num, x)
% Irish Rail Hyundai Rotem InterCity fleet (Obrien et al., 2018)
% The installation was carried out in December 2015 on the leading car 
%(22337) of Set 37, a five-car train set. DOI: 10.1177/1475921717744479

% Note: The vehicle model presented in the source document is in 3D. Therefore,
%   some of the properties have to be multiplied by 2 to reduce the model to 2D

%!!Variabilidade dos parâmetros: massa da caixa, rigidez primaria e
%rigidez secundaria

%Cada veículo possui +- 60 assentos/pasageiros (60*75kg=4500kg)
% var_BodyMass = (randn*0.1*36852);  % 10% da media - Gaussiana
% 
% var_Prim_k = (randn*0.05*(2779e3*2)); % 5% da media - Gaussiana
% var_Sec_k = (randn*0.05*(1000e3*2));  % 5% da media - Gaussiana

% var_BodyMass = 0;
% var_Prim_k = 0;
% var_Sec_k = 0;

var_BodyMass = (x(1)*0.10*36852);  % 10% da media - Gaussiana
var_Prim_k = (x(2)*0.05*(2779e3*2)); % 5% da media - Gaussiana
var_Sec_k = (x(3)*0.05*(1000e3*2));  % 5% da media - Gaussiana

Train.Veh(veh_num).Body.m = 36852+var_BodyMass;          % Main body mass [kg]
Train.Veh(veh_num).Body.I = 560342;                      % Main body mass moment of inertia [kg*m^2]
Train.Veh(veh_num).Body.L = 8*2;                        % Main body length [m] (From bogie to bogie)
Train.Veh(veh_num).Body.Le = [1,1]*(1.5+3/2);             % Additional distance [m], front and back
Train.Veh(veh_num).Bogie.num = 2;                         % Number of bogies
Train.Veh(veh_num).Bogie.m = [1,1]*3910;                  % Bogies masses [kg]
Train.Veh(veh_num).Bogie.I = [1,1]*10024;                  % Bogies mass moments of inertia [kg*m^2]
Train.Veh(veh_num).Bogie.L = [1,1]*2.3;                  % Bogies length [m] (From wheel to wheel)
Train.Veh(veh_num).Wheels.num = 4;                        % Number of wheels
Train.Veh(veh_num).Wheels.m = [1,1,1,1]*1407;             % Wheel masses [kg]
Train.Veh(veh_num).Susp.Prim.k = [1,1,1,1]*2779e3*2+var_Prim_k;      % Primary suspension stiffness [N/m]
Train.Veh(veh_num).Susp.Prim.c = [1,1,1,1]*29.4e3*2;         % Primary suspension viscous damping [Ns/m]
Train.Veh(veh_num).Susp.Sec.k = [1,1]*1000e3*2+var_Sec_k;            % Secondary suspension stiffness [N/m]
Train.Veh(veh_num).Susp.Sec.c = [1,1]*60e3*2;             % Secondary suspension viscous damping [Ns/m]

% Manchester
% Train.Veh(veh_num).Body.m = 32000;                        % Main body mass [kg]
% Train.Veh(veh_num).Body.I = 1970000;                      % Main body mass moment of inertia [kg*m^2]
% Train.Veh(veh_num).Body.L = 9.5*2;                        % Main body length [m] (From bogie to bogie)
% Train.Veh(veh_num).Body.Le = [1,1]*(1.5+3/2);             % Additional distance [m], front and back
% Train.Veh(veh_num).Bogie.num = 2;                         % Number of bogies
% Train.Veh(veh_num).Bogie.m = [1,1]*2615;                  % Bogies masses [kg]
% Train.Veh(veh_num).Bogie.I = [1,1]*1476;                  % Bogies mass moments of inertia [kg*m^2]
% Train.Veh(veh_num).Bogie.L = [1,1]*1.28*2;                % Bogies length [m] (From wheel to wheel)
% Train.Veh(veh_num).Wheels.num = 4;                        % Number of wheels
% Train.Veh(veh_num).Wheels.m = [1,1,1,1]*1813;             % Wheel masses [kg]
% Train.Veh(veh_num).Susp.Prim.k = [1,1,1,1]*1200e3*2;      % Primary suspension stiffness [N/m]
% Train.Veh(veh_num).Susp.Prim.c = [1,1,1,1]*4e3*2;         % Primary suspension viscous damping [Ns/m]
% Train.Veh(veh_num).Susp.Sec.k = [1,1]*430e3*2;            % Secondary suspension stiffness [N/m]
% Train.Veh(veh_num).Susp.Sec.c = [1,1]*20e3*2;             % Secondary suspension viscous damping [Ns/m]

% ---- End of script ----
end