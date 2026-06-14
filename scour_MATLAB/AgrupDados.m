clc;
clear all;
close all;

Vagao = 'PrimVag';

%% Vagao, Truque frontal e Truque traseiro
for jj=1:3
    if jj == 1
        SensorPosition = 'VG';  %VG: vagão, TF: Truque Frontal, TT: Truque Traseiro, RF: Roda Frontal
    elseif jj == 2
        SensorPosition = 'TF';
    else
        SensorPosition = 'TT';
    end
    %---------------------- Agrupamento dos dados ------------------
    load('Data09-07-25_Case_1')

    for i=1:200

          txt1 = ['Case1(i,:) = data.AceleracaoPrimVag{1,i}(jj,:)'];
          eval(txt1);

    end
   
    save(['Data09-07-25_' num2str(SensorPosition) '_' num2str(Vagao) '_Case1_Cut.mat'],...
    'Case1','Temperatura','Velocidade')
    
    for i=1:200
          txt2 = ['Case1(i,:) = data.AcelRodaPrimVag{1,i}(jj,:)'];
          eval(txt2);

    end
    
    save(['Data09-07-25_RF_' num2str(Vagao) '_Case1_Cut.mat'],...
    'Case1','Temperatura','Velocidade')
    
end
 

    