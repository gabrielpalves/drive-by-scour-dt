function [data] = D01_DataProcessing(i, j, Sol, Train, Calc, Damage, data)
%% Spacial domain
% AA{i,j} = Sol.Veh(1).A(1:3,:);  %Primeiro vagao do comboio

data.AceleracaoPrimVag{i,j} = Sol.Veh(1).A(1:3,:); % Primeiro vagao do comboio
% data.AceleracaoIntVag{i,j} = Sol.Veh(3).A(1:3,:); % Terceiro vagao do comboio
% data.AceleracaoUltVag{i,j} = Sol.Veh(5).A(1:3,:); % Último vagao do comboio (5)

data.PitchPrimVag{i,j} = Sol.Veh(1).V(4:6,:); % Pitch rate primeiro vagão

data.AcelRodaPrimVag{i,j} = Sol.Veh(1).acc_under(1:4,:); % Aceleração da roda do primeiro vagao

data.AcelRodaPrimVag{i,j} = data.AcelRodaPrimVag{i,j} + ... Acrescenta ruído na roda
    Damage.desvio * data.AcelRodaPrimVag{i,j} .* randn(size(data.AcelRodaPrimVag{i,j}));

% data.AcelRodaIntVag{i,j} = Sol.Veh(3).acc_under(1:4,:);  %Aceleracao da roda do primeiro vagao
% data.AcelRodaUltVag{i,j} = Sol.Veh(5).acc_under(1:4,:);  %Aceleracao da roda do ultimo vagao

data.Velocidade{i,j} = Train.vel; 
data.Posicao{i,j} = Train.vel*(Calc.Solver.num_t/1000);


%% Discretizacao no dominio do espaco

DimAcel = length(Sol.Veh(1).A(1,:));
DimSpace = round(data.Posicao{i,j}*100);

AcelEspacoPrimVag = zeros(length(data.AceleracaoPrimVag{i,j}(:,1)),DimSpace);
% AcelEspacoIntVag = zeros(length(data.AceleracaoIntVag{i,j}(:,1)),DimSpace);
% AcelEspacoUltVag = zeros(length(data.AceleracaoPrimVag{i,j}(:,1)),DimSpace);

AcelRodaPrimVag = zeros(length(data.AcelRodaPrimVag{i,j}(:,1)),DimSpace);
% AcelRodaIntVag = zeros(length(data.AcelRodaIntVag{i,j}(:,1)),DimSpace);
% AcelRodaUltVag = zeros(length(data.AcelRodaPrimVag{i,j}(:,1)),DimSpace);

PitchPrimVag = zeros(length(data.PitchPrimVag{i,j}(:,1)),DimSpace);

%Transformacao para o dominio do espaco
xi = (1:DimSpace);
xx = linspace(1,DimSpace,DimAcel);

for ii = 1:length(data.AceleracaoPrimVag{i,j}(:,1))
    
    %----Truques e Vagao -----
    x_intpPrim = interp1(xx,data.AceleracaoPrimVag{i,j}(ii,:),xi);
    AcelEspacoPrimVag(ii,:) = x_intpPrim;
    
%     x_intpInt = interp1(xx,data.AceleracaoIntVag{i,j}(ii,:),xi);
%     AcelEspacoIntVag(ii,:) = x_intpInt;
%     
%     x_intpUlt = interp1(xx,data.AceleracaoUltVag{i,j}(ii,:),xi);
%     AcelEspacoUltVag(ii,:) = x_intpUlt;

    x_pitchPrim = interp1(xx,data.PitchPrimVag{i,j}(ii,:),xi);
    PitchPrimVag(ii,:) = x_pitchPrim;
     
    %------Roda-----
    acel_intpPrim = interp1(xx,data.AcelRodaPrimVag{i,j}(ii,:),xi);
    AcelRodaPrimVag(ii,:) = acel_intpPrim; 
    
%     acel_intpInt = interp1(xx,data.AcelRodaIntVag{i,j}(ii,:),xi);
%     AcelRodaIntVag(ii,:) = acel_intpInt; 
%     
%     acel_intpUlt = interp1(xx,data.AcelRodaUltVag{i,j}(ii,:),xi);
%     AcelRodaUltVag(ii,:) = acel_intpUlt;
    
end

data.AceleracaoPrimVag{i,j} = AcelEspacoPrimVag(:, 1001:5000+1831);
data.AcelRodaPrimVag{i,j} = AcelRodaPrimVag(:, 1001:5000+1831);
data.PitchPrimVag{i,j} = PitchPrimVag(:, 1001:5000+1831);

% Plot Vehicle vertical acceleration
% inputFolder = 'C:\Users\CLIENTE\Desktop\Thiago\Scour_TTBaseline';
end