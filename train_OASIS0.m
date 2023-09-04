function [BB,WW,HH,param] = train_OASIS0(XTrain_new,LTrain_new,GTrain_new,Kt,param)
    
    max_iter = param.max_iter;
    
    alpha = param.alpha; 
    beta = param.beta; 
    gamma = param.gamma; 
    thet = param.thet;
    delta = param.delta; 

    nbits = param.current_bits;
    n2 = size(LTrain_new,2);
    t = size(Kt,2);

    %% initization
    B_new = sign(randn(nbits,n2)); 
    B_new(B_new==0) = -1;
    
    U_new = randn(t,nbits);

    Xm1=XTrain_new(1:param.image_feature_size,:);
    Xm2=XTrain_new(param.image_feature_size+1:end,:);

   %% step of hash codes learning    
    for i = 1:max_iter       

        Z=nbits*alpha*((B_new*GTrain_new')*GTrain_new)...
          +beta*U_new'*GTrain_new...
          +B_new;
        
        Temp = Z*Z'-(1/n2)*Z*ones(n2,1)*ones(1,n2)*Z';
        [~,Lmd,GG] = svd(Temp); clear Temp
        idx = (diag(Lmd)>1e-6);
        G = GG(:,idx); 
        G_ = orth(GG(:,~idx));
        Q = Z'*G/(sqrt(Lmd(idx,idx)))-(1/n2)*ones(n2,1)*(ones(1,n2)*Z')*G/(sqrt(Lmd(idx,idx)));
        Q_ = orth(randn(n2,nbits-length(find(idx==1))));
        V_new = sqrt(n2)*[G G_]*[Q Q_]';
             
        
        % update U
        U_new = (GTrain_new*V_new')/(V_new*V_new'+thet*eye(nbits));
        
        % update W
        WW{1,1}=B_new*Xm1'/(Xm1*Xm1'+delta*eye(param.image_feature_size));
        WW{2,1}=B_new*Xm2'/(Xm2*Xm2'+delta*eye(param.text_feature_size));
%         WW{1,1}=B_new*Xm1'*pinv(Xm1*Xm1'+delta*eye(param.image_feature_size));
%         WW{2,1}=B_new*Xm2'*pinv(Xm2*Xm2'+delta*eye(param.text_feature_size));

        % update B       
        B_new = sign(alpha*nbits*(V_new*GTrain_new')*GTrain_new...
            +V_new+...
            gamma*(WW{1,1}*Xm1+WW{2,1}*Xm2));
    end

    %% save results
    H1_new = B_new*GTrain_new';
    H2_new = GTrain_new*V_new';
    H3_new = V_new*V_new';
    H4_new = V_new*GTrain_new';
    H5_new = B_new*Xm1';
    H6_new = Xm1*Xm1';
    H7_new = B_new*Xm2';
    H8_new = Xm2*Xm2';

    HH{1,1} = H1_new;
    HH{1,2} = H2_new;
    HH{1,3} = H3_new;
    HH{1,4} = H4_new;
    HH{1,5} = H5_new;
    HH{1,6} = H6_new;
    HH{1,7} = H7_new;
    HH{1,8} = H8_new;
    BB{1,1} = B_new';
end

