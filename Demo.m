clear all
clc
close all
N=10;
for ii=1:1:N
    load(['data_' num2str(ii) '.mat'])
    %% Vectorizing labels of training data
    X=full(ind2vec(LTra1')');
    X(X==0)=-1;
    %% Training
    Input.x=DTra1;Input.y=X;
    tic
    [Output]=SAFLS(Input,'L');
    texe(ii)=toc;
    %% Testing
    Input.x=DTes1;Input.Syst=Output.Syst;
    [Output]=SAFLS(Input,'T');
    label_est=Output.Ye;
    [~,label_est]=max(label_est,[],2);
    Acc(ii)=sum(sum(confusionmat(LTes1,label_est).*(eye(length(unique(LTes1))))))/length(LTes1);
end
[mean(Acc),std(Acc),max(Acc)]
[mean(texe),std(texe),max(texe)]