function [Output]=SAFLS(Input,Mode)
global threshold1
global threshold2
global threshold3
threshold1=exp(-1);
threshold2=0.5;
threshold3=exp(-1);
if strcmp(Mode,'L')==1
    [Output.Ye,Output.Syst.ModelNumber,Output.Syst.prototypes,Output.Syst.center,Output.Syst.Support,Output.Syst.local_delta,Output.Syst.Global_mean,Output.Syst.Global_X,Output.Syst.A,Output.Syst.L]=EFStrain(Input.x,Input.y);
end
if strcmp(Mode,'T')==1
    [Output.Ye]=EFStest(Input.x,Input.Syst.ModelNumber,Input.Syst.prototypes,Input.Syst.center,Input.Syst.local_delta,Input.Syst.Global_mean,Input.Syst.Global_X,Input.Syst.A,Input.Syst.L);
end
end
function [Ye,ModelNumber,prototypes,center,Support,local_delta,Global_mean,Global_X,A,L]=EFStrain(data0,Y0)
global threshold1
global threshold2
global threshold3
forgettingfactor=0.05;
CL=size(Y0,2);
omega=1000;
[L,W]=size(data0);
center=data0(1,:);
prototypes=data0(1,:);
local_X=data0(1,:).^2;
local_delta=zeros(1,W);
Global_mean=data0(1,:);
Global_X=data0(1,:).^2;
Support=1;
ModelNumber=1;
sum_lambda=1;
Index=1;
A=zeros(CL,W+1,1);
C=eye(W+1)*omega;
Ye=zeros(L,CL);
%%
for ii=2:1:L
    %     ii
    datain=data0(ii,:);
    yin=Y0(ii,:);
    Global_mean=Global_mean.*(ii-1)./ii+datain./ii;
    Global_X=Global_X.*(ii-1)./ii+datain.^2./ii;
    Global_Delta=abs(Global_X-Global_mean.^2);
    [centerlambda,LocalDensity]=firingstrength(datain,ModelNumber,prototypes,local_delta,Global_Delta,W);
    Ye(ii,:)=OutputGeneration(datain,A,centerlambda,LocalDensity,ModelNumber,CL,threshold2,threshold3);
    if max(LocalDensity)<threshold1
        %% new_cloud_add
        ModelNumber=ModelNumber+1;
        center=[center;datain];
        prototypes=[prototypes;datain];
        local_X=[local_X;datain.^2];
        Support=[Support,1];
        local_delta=[local_delta;zeros(1,W)];
        A(:,:,ModelNumber)=mean(A,3);
        C(:,:,ModelNumber)=eye(W+1)*omega;
        sum_lambda=[sum_lambda;0];
        Index=[Index;ii];
        LocalDensity=[LocalDensity;1];
        %%
    else
        %% local_parameters_update
        [~,label0]=max(LocalDensity);
        Support(label0)=Support(label0)+1;
        center(label0,:)=(Support(label0)-1)/Support(label0)*center(label0,:)+datain/Support(label0);
        local_X(label0,:)=(Support(label0)-1)/Support(label0)*local_X(label0,:)+datain.^2/Support(label0);
        local_delta(label0,:)=abs(local_X(label0,:)-center(label0,:).^2);
        LocalDensity(label0)=exp(-1*sum((datain-prototypes(label0,:)).^2)/(sum(Global_Delta+local_delta(label0,:))./2));
    end
    centerlambda=LocalDensity./sum(LocalDensity);
    %% stale_datacloud_remove
    sum_lambda=sum_lambda+LocalDensity;
    utility=sum_lambda./(ii-Index);
    seq=find(utility>=forgettingfactor);
    ModelNumber0=length(seq);
    if ModelNumber0<ModelNumber
        center=center(seq,:);
        local_X=local_X(seq,:);
        Index=Index(seq);
        sum_lambda=sum_lambda(seq);
        centerlambda=centerlambda(seq)./sum(centerlambda(seq));
        Support=Support(seq);
        A=A(:,:,seq);
        C=C(:,:,seq);
        prototypes=prototypes(seq,:);
        LocalDensity=LocalDensity(seq);
    end
    local_delta=abs(local_X-center.^2);
    ModelNumber=ModelNumber0;
    [centerlambda,seq1]=ActivatingRules(ModelNumber,centerlambda,LocalDensity,threshold2,threshold3);
    X=[1,datain];
    for jj=seq1
        C(:,:,jj)=C(:,:,jj)-centerlambda(jj)*C(:,:,jj)*X'*X*C(:,:,jj)/(1+centerlambda(jj)*X*C(:,:,jj)*X');
        A1=A(:,:,jj)'+centerlambda(jj)*C(:,:,jj)*X'*(yin-X*A(:,:,jj)');
        A(:,:,jj)=A1';
    end
end
end
function [Ye]=EFStest(data1,ModelNumber,prototypes,center,local_delta,Global_mean,Global_X,A,L1)
global threshold2
global threshold3
CL=size(A,1);
[L,W]=size(data1);
Ye=zeros(L,CL);
for ii=1:1:L
    datain=data1(ii,:);
    Global_mean1=Global_mean.*L1./(L1+1)+datain./(L1+1);
    Global_X1=Global_X.*L1./(L1+1)+datain.^2./(L1+1);
    Global_Delta1=abs(Global_X1-Global_mean1.^2);
    [centerlambda,LocalDensity]=firingstrength(datain,ModelNumber,prototypes,local_delta,Global_Delta1,W);
    Ye(ii,:)=OutputGeneration(datain,A,centerlambda,LocalDensity,ModelNumber,CL,threshold2,threshold3);
end
end
function Ye=OutputGeneration(datain,A,centerlambda,LocalDensity,ModelNumber,CL,threshold2,threshold3)
Ye=zeros(1,CL);
[centerlambda,seq1]=ActivatingRules(ModelNumber,centerlambda,LocalDensity,threshold2,threshold3);
for ii=seq1
    Ye=Ye+[1,datain]*A(:,:,ii)'*centerlambda(ii);
end
end
function [centerlambda,seq1]=ActivatingRules(ModelNumber,centerlambda,LocalDensity,threshold2,threshold3)
[values,seq]=sort(LocalDensity,'descend');
values=sum(triu(repmat(values,1,ModelNumber)),1);
a=find(values>=threshold2*sum(LocalDensity));
seq1=seq(1:1:a(1))';
centerlambda(seq1)=LocalDensity(seq1)./sum(LocalDensity(seq1));
end
function [centerlambda,LocalDensity]=firingstrength(datain,ModelNumber,center,local_delta,Global_Delta,W)
datain1=sum((repmat(datain,ModelNumber,1)-center).^2,2);
Global_Delta1=sum((repmat(Global_Delta,ModelNumber,1)+local_delta)/2,2);
LocalDensity=exp(-1*datain1./Global_Delta1);
centerlambda=LocalDensity./sum(LocalDensity);
end
%
%