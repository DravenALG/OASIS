function [eva,train_time_round] = evaluate_OASIS(XTrain,LTrain,XQuery,LQuery,K,OURparam)
    eva=zeros(1,OURparam.nchunks);
    train_time_round=zeros(1,OURparam.nchunks);

    for chunki = 1:OURparam.nchunks
        fprintf('-----chunk----- %3d\n', chunki);       
        
        Kt = K{chunki,:};      
        
        XTrain_new = XTrain{chunki,:};
        LTrain_new = LTrain{chunki,:};

        GTrain_new = (Kt'*LTrain_new')';
        GTrain_new = GTrain_new ./ sum(GTrain_new.^2,2).^0.5;
        
        XQueryt = XQuery{chunki,:};
        LQueryt = LQuery{chunki,:};
                              
        % Hash code learning
        tic
        if chunki == 1
            [BB,WW,PP,OURparam] = train_OASIS0(XTrain_new',LTrain_new',GTrain_new', Kt,OURparam);
        else
            [BB,WW,PP,OURparam] = train_OASIS(XTrain_new',LTrain_new',GTrain_new', Kt,BB,PP,OURparam);
        end
        train_time_round(1,chunki) = toc;

        fprintf('test beginning\n');
        XQ_1=XQueryt(:,1:OURparam.image_feature_size)';
        XQ_2=XQueryt(:,OURparam.image_feature_size+1:end)';
        XQuery_B = compactbit((WW{1,1}*XQ_1+WW{2,1}*XQ_2)'>0);
        
        B = cell2mat(BB(1:chunki,:));
        XTrain_B = compactbit(B>0);

        %mAP
        DHamm = hammingDist(XQuery_B, XTrain_B);
        [~, orderH] = sort(DHamm, 2);
      
        label_count=size(LTrain_new,2);
        LBase=[];
        for i=1:chunki
            LBase=[LBase; [zeros(size(LTrain{i,1},1) , label_count-size(LTrain{i,1},2)) LTrain{i,1}]];
        end
        
        
        eva(1,chunki) = mAP(orderH', LBase, LQueryt);
        fprintf('the %i chunk : mAP=%d train_time=%d \n', chunki,eva(1,chunki), train_time_round(1,chunki));
        
    end
end

