clear all
clc
close all
N=10;
for ii=1:1:N
    load(['data_' num2str(ii) '.mat'])
    seq=randperm(length(LTra1));
    Input.x=DTra1(seq,:);Input.y=(LTra1(seq)-1.5)*2;
    tic
    [Output]=SALFS(Input,'L');
    tt(ii)=toc
    
    Input.x=DTes1;Input.Syst=Output.Syst;
    
    [Output]=SALFS(Input,'T');
    label_est=Output.Ye;
    label_est(Output.Ye>0)=2;
    label_est(Output.Ye<=0)=1;
    
    A3(ii)=sum(sum(confusionmat(LTes1,label_est).*(eye(length(unique(LTes1))))))/length(LTes1);  
    NN(ii)=Input.Syst.ModelNumber;
    
end
'Training'
'Testing'

[mean(A3),std(A3),max(A3)]
[mean(NN),std(NN),max(NN)]
[mean(tt),std(tt),max(tt)]
