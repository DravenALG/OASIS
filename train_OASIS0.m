function [BB,WW,DD,param] = train_OASIS0(XTrain_new,LTrain_new,GTrain_new,Kt,param)
    %%  parameters
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

    %% step of pre-process
    % zero-mean
    Xm1=XTrain_new(1:param.image_feature_size,:);
    Xm2=XTrain_new(param.image_feature_size+1:end,:);
     
    mean_1 = sum(Xm1,2) / n2;
    mean1 = mean_1;
    param.previous_mean1 = mean1;
    mean_2 = sum(Xm2,2) / n2;
    mean2 = mean_2;
    param.previous_mean2 = mean2;
    
    param.previous_nt = n2;
    
    Xm1 = Xm1 - mean1;
    Xm2 = Xm2 - mean2;

   %% step of hash function learning
   
    
    for i = 1:max_iter       

        Z=2*nbits*alpha*((B_new*GTrain_new')*GTrain_new)...
            +beta*U_new'*GTrain_new...
            +B_new;
        
        Temp = Z*Z'-(1/n2)*Z*ones(n2,1)*ones(1,n2)*Z';
        [~,Lmd,OO] = svd(Temp); clear Temp
        idx = (diag(Lmd)>1e-6);
        O = OO(:,idx); 
        O_ = orth(OO(:,~idx));
        N = Z'*O/(sqrt(Lmd(idx,idx)))-(1/n2)*ones(n2,1)*(ones(1,n2)*Z')*O/(sqrt(Lmd(idx,idx)));
        N_ = orth(randn(n2,nbits-length(find(idx==1))));
        V_new = sqrt(n2)*[O O_]*[N N_]';
        
        % update U
        U_new = (GTrain_new*V_new')/(V_new*V_new'+thet);
        
        % update W
        WW{1,1}=B_new*Xm1'/(Xm1*Xm1'+delta*eye(param.image_feature_size));
        WW{2,1}=B_new*Xm2'/(Xm2*Xm2'+delta*eye(param.text_feature_size));
             
        % update B       
        
        B_new = sign(2*alpha*nbits*(V_new*GTrain_new')*GTrain_new...
            +V_new+...
            gamma*(WW{1,1}*Xm1+WW{2,1}*Xm2));
    end

    %% save results
    D1_new = B_new*GTrain_new';
    D2_new = GTrain_new*V_new';
    D3_new = V_new*V_new';
    D4_new = V_new*GTrain_new';
    D5_new = B_new*Xm1';
    D6_new = Xm1*Xm1';
    D7_new = B_new*Xm2';
    D8_new = Xm2*Xm2';

    DD{1,1} = D1_new;
    DD{1,2} = D2_new;
    DD{1,3} = D3_new;
    DD{1,4} = D4_new;
    DD{1,5} = D5_new;
    DD{1,6} = D6_new;
    DD{1,7} = D7_new;
    DD{1,8} = D8_new;
    BB{1,1} = B_new';
end

