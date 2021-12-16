function [train_param,XTrain,LTrain,LTrain_All,XQuery,LQuery,LQuery_All,K,K_All] = load_dataset(train_param)
    fprintf(['-------load dataset------', '\n']);
    load([train_param.ds_name,'_deep.mat']);
    load([train_param.ds_name,'_Groundtrue_Vec.mat']);

    if strcmp(train_param.ds_name, 'MIRFLICKR')

        train_param.image_feature_size=4096;
        train_param.text_feature_size=1386;
        
        if strcmp(train_param.load_type, 'first_setting')
            expected_chunksize=2000;
            
            X = [I_tr T_tr; I_te T_te];L = [L_tr; L_te];

            R = randperm(size(L,1));
            queryInds = R(1:2000);
            sampleInds = R(2001:end);

            train_param.nchunks = ceil(length(sampleInds)/expected_chunksize);
            
            train_param.chunksize = cell(train_param.nchunks,1);
            train_param.test_chunksize = cell(train_param.nchunks,1);

            XTrain = cell(train_param.nchunks,1);
            LTrain = cell(train_param.nchunks,1);
            LTrain_All = cell(train_param.nchunks,1);

            XQuery = cell(train_param.nchunks,1);
            LQuery = cell(train_param.nchunks,1);
            LQuery_All = cell(train_param.nchunks,1);

            K = cell(train_param.nchunks,1);

            for subi = 1:train_param.nchunks-1
                XTrain{subi,1} = X(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
                LTrain{subi,1} = L(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
                LTrain_All{subi,1} = LTrain{subi,1};
                [train_param.chunksize{subi, 1},~] = size(XTrain{subi,1});

                XQuery{subi,1} = X(queryInds, :);
                LQuery{subi,1} = L(queryInds, :);
                LQuery_All{subi,1} = LQuery{subi,1};
                [train_param.test_chunksize{subi, 1},~] = size(XQuery{subi,1});

                K{subi,1} = mir_gt_vec;
            end

            XTrain{train_param.nchunks,1} = X(sampleInds(expected_chunksize*subi+1:end),:);
            LTrain{train_param.nchunks,1} = L(sampleInds(expected_chunksize*subi+1:end),:);
            LTrain_All{train_param.nchunks,1} = LTrain{train_param.nchunks,1};
            [train_param.chunksize{train_param.nchunks, 1},~] = size(XTrain{train_param.nchunks,1});

            XQuery{train_param.nchunks,1} = X(queryInds, :);
            LQuery{train_param.nchunks,1} = L(queryInds, :);
            LQuery_All{train_param.nchunks,1} = LQuery{train_param.nchunks,1};
            [train_param.test_chunksize{train_param.nchunks, 1},~] = size(XQuery{train_param.nchunks,1});

            K{train_param.nchunks,1} = mir_gt_vec;           
            K_All=K;
            
        elseif strcmp(train_param.load_type, 'second_setting')
            X = [I_tr T_tr; I_te T_te];
            L = [L_tr; L_te];
            
            [~,L_idx]=sort(sum(L),'descend');
            L=L(:,L_idx);
            mir_gt_vec=mir_gt_vec(L_idx,:);
            
            train_param.nchunks=10;
            
            labels=linspace(1,24,24);
            seperate=cell(train_param.nchunks,1);
            seperate{1,1}=[1];
            seperate{2,1}=[2,23];
            seperate{3,1}=[3,22];
            seperate{4,1}=[4,21];
            seperate{5,1}=[5];
            seperate{6,1}=[6,19];
            seperate{7,1}=[16,17,18,20];
            seperate{8,1}=[8,7,24];
            seperate{9,1}=[9,10,15,13];
            seperate{10,1}=[11,12,13,14];
            
            train_param.chunksize = cell(train_param.nchunks,1);
            train_param.test_chunksize = cell(train_param.nchunks,1);

            XTrain = cell(train_param.nchunks,1);
            LTrain = cell(train_param.nchunks,1);
            LTrain_All = cell(train_param.nchunks,1);

            XQuery = cell(train_param.nchunks,1);
            LQuery = cell(train_param.nchunks,1);
            LQuery_All = cell(train_param.nchunks,1);

            K = cell(train_param.nchunks,1);
            K_All = cell(train_param.nchunks,1);
            
            label_allow=[];
            last_found_idx=[];
            
            for l=1:train_param.nchunks
                label_allow=[seperate{l,1} label_allow];
                label_notallow=setdiff(labels,label_allow);
                idx_find_all=find(sum(L(:,label_notallow),2)==0);
                idx_find=setdiff(idx_find_all,last_found_idx);
                last_found_idx=idx_find_all;
                
                R = randperm(size(idx_find,1));
                queryInds = R(1,1:floor(size(idx_find,1)*0.1));
                sampleInds = R(1,floor(size(idx_find,1)*0.1)+1:end);
                
                X_tmp=X(idx_find,:);
                L_tmp=L(idx_find,label_allow);
                L_all_tmp=L(idx_find,:);
                
                XTrain{l,1}=X_tmp(sampleInds,:);
                LTrain{l,1}=L_tmp(sampleInds,:);
                LTrain_All{l,1}=L_all_tmp(sampleInds,:);
                
                XQuery{l,1}=X_tmp(queryInds,:);
                LQuery{l,1}=L_tmp(queryInds,:);
                LQuery_All{l,1}=L_all_tmp(queryInds,:);
                
                K{l,1}=mir_gt_vec(label_allow,:);
                K_All{l,1}=mir_gt_vec;
                
                train_param.chunksize{l,1}=size(sampleInds,2);
                train_param.test_chunksize{l,1}=size(queryInds,2);
            end
            
            
        elseif strcmp(train_param.load_type, 'third_setting')
            X = [I_tr T_tr; I_te T_te];
            L = [L_tr; L_te];
            
            [~,L_idx]=sort(sum(L),'descend');
            L=L(:,L_idx);
            mir_gt_vec=mir_gt_vec(L_idx,:);
            
            train_param.nchunks=7;
            
            labels=linspace(1,24,24);
            seperate=cell(train_param.nchunks,1);
            seperate{1,1}=[1];
            seperate{2,1}=[2,23,8,24,4,21,16,17,18,20];
            seperate{3,1}=[3,22];
            seperate{4,1}=[5];
            seperate{5,1}=[6,7,19];
            seperate{6,1}=[9,10,15,13];
            seperate{7,1}=[11,12,13,14];
            
            train_param.chunksize = cell(train_param.nchunks,1);
            train_param.test_chunksize = cell(train_param.nchunks,1);

            XTrain = cell(train_param.nchunks,1);
            LTrain = cell(train_param.nchunks,1);
            LTrain_All = cell(train_param.nchunks,1);

            XQuery = cell(train_param.nchunks,1);
            LQuery = cell(train_param.nchunks,1);
            LQuery_All = cell(train_param.nchunks,1);

            K = cell(train_param.nchunks,1);
            K_All= cell(train_param.nchunks,1);
            
            label_appear=[];
            
            for l=1:train_param.nchunks
                label_allow=seperate{l,1};
                label_appear=[label_allow label_appear];
                label_notallow=setdiff(labels,label_allow);
                idx_find=find(sum(L(:,label_notallow),2)==0);
                
                R = randperm(size(idx_find,1));
                queryInds = R(1,1:floor(size(R,2)*0.1));
                sampleInds = R(1,floor(size(R,2)*0.1)+1:end);
                
                X_tmp=X(idx_find,:);
                L_tmp=L(idx_find,label_appear);
                L_all_tmp=L(idx_find,:);
                
                XTrain{l,1}=X_tmp(sampleInds,:);
                LTrain{l,1}=L_tmp(sampleInds,:);
                LTrain_All{l,1}=L_all_tmp(sampleInds,:);
                
                XQuery{l,1}=X_tmp(queryInds,:);
                LQuery{l,1}=L_tmp(queryInds,:);
                LQuery_All{l,1}=L_all_tmp(queryInds,:);
                
                K{l,1}=mir_gt_vec(label_appear,:);
                K_All{l,1}=mir_gt_vec;
                
                train_param.chunksize{l,1}=size(sampleInds,2);
                
                train_param.test_chunksize{l,1}=size(queryInds,2);
            end
        end
        
        clear X L subi queryInds sampleInds R Image Tag Label

    elseif strcmp(train_param.ds_name, 'NUSWIDE21')       

        train_param.image_feature_size=4096;
        train_param.text_feature_size=5018;
        
        if strcmp(train_param.load_type, 'first_setting')
            expected_chunksize=10000;

            X = [I_tr T_tr; I_te T_te];L = [L_tr; L_te];

            R = randperm(size(L,1));
            queryInds = R(1:2000);
            sampleInds = R(2001:end);

            train_param.nchunks = ceil(length(sampleInds)/expected_chunksize);
            
            train_param.chunksize = cell(train_param.nchunks,1);
            train_param.test_chunksize = cell(train_param.nchunks,1);

            XTrain = cell(train_param.nchunks,1);
            LTrain = cell(train_param.nchunks,1);
            LTrain_All = cell(train_param.nchunks,1);

            XQuery = cell(train_param.nchunks,1);
            LQuery = cell(train_param.nchunks,1);
            LQuery_All = cell(train_param.nchunks,1);

            K = cell(train_param.nchunks,1);

            for subi = 1:train_param.nchunks-1
                XTrain{subi,1} = X(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
                LTrain{subi,1} = L(sampleInds(expected_chunksize*(subi-1)+1:expected_chunksize*subi),:);
                LTrain_All{subi,1} = LTrain{subi,1};
                [train_param.chunksize{subi, 1},~] = size(XTrain{subi,1});

                XQuery{subi,1} = X(queryInds, :);
                LQuery{subi,1} = L(queryInds, :);
                LQuery_All{subi,1} = LQuery{subi,1};
                [train_param.test_chunksize{subi, 1},~] = size(XQuery{subi,1});

                K{subi,1} = nus_gt_vec;
            end

            XTrain{train_param.nchunks,1} = X(sampleInds(expected_chunksize*subi+1:end),:);
            LTrain{train_param.nchunks,1} = L(sampleInds(expected_chunksize*subi+1:end),:);
            LTrain_All{train_param.nchunks,1} = LTrain{train_param.nchunks,1};
            [train_param.chunksize{train_param.nchunks, 1},~] = size(XTrain{train_param.nchunks,1});

            XQuery{train_param.nchunks,1} = X(queryInds, :);
            LQuery{train_param.nchunks,1} = L(queryInds, :);
            LQuery_All{train_param.nchunks,1} = LQuery{train_param.nchunks,1};
            [train_param.test_chunksize{train_param.nchunks, 1},~] = size(XQuery{train_param.nchunks,1});

            K{train_param.nchunks,1} = nus_gt_vec;
            K_All=K;
            
        elseif strcmp(train_param.load_type, 'second_setting')
            X = [I_tr T_tr; I_te T_te];
            L = [L_tr; L_te];
            
            train_param.nchunks=20;
            
            labels=linspace(1,21,21);
            seperate=cell(train_param.nchunks,1);
            seperate{1,1}=[1];
            seperate{2,1}=[2];
            seperate{3,1}=[3];
            seperate{4,1}=[4];
            seperate{5,1}=[5];
            seperate{6,1}=[6];
            seperate{7,1}=[7];
            seperate{8,1}=[8];
            seperate{9,1}=[9];
            seperate{10,1}=[10];
            seperate{11,1}=[11];
            seperate{12,1}=[12];
            seperate{13,1}=[13];
            seperate{14,1}=[14];
            seperate{15,1}=[15];
            seperate{16,1}=[16];
            seperate{17,1}=[17];
            seperate{18,1}=[18];
            seperate{19,1}=[19];
            seperate{20,1}=[20,21];
            
            train_param.chunksize = cell(train_param.nchunks,1);
            train_param.test_chunksize = cell(train_param.nchunks,1);

            XTrain = cell(train_param.nchunks,1);
            LTrain = cell(train_param.nchunks,1);
            LTrain_All = cell(train_param.nchunks,1);

            XQuery = cell(train_param.nchunks,1);
            LQuery = cell(train_param.nchunks,1);
            LQuery_All = cell(train_param.nchunks,1);

            K = cell(train_param.nchunks,1);
            K_All = cell(train_param.nchunks,1);
            
            label_allow=[];
            last_found_idx=[];
            
            for l=1:train_param.nchunks
                label_allow=[seperate{l,1} label_allow];
                label_notallow=setdiff(labels,label_allow);
                idx_find_all=find(sum(L(:,label_notallow),2)==0);
                idx_find=setdiff(idx_find_all,last_found_idx);
                last_found_idx=idx_find_all;
                
                R = randperm(size(idx_find,1));
                queryInds = R(1,1:floor(size(idx_find,1)*0.1));
                sampleInds = R(1,floor(size(idx_find,1)*0.1)+1:end);
                
                X_tmp=X(idx_find,:);
                L_tmp=L(idx_find,label_allow);
                L_all_tmp=L(idx_find,:);
                
                XTrain{l,1}=X_tmp(sampleInds,:);
                LTrain{l,1}=L_tmp(sampleInds,:);
                LTrain_All{l,1}=L_all_tmp(sampleInds,:);
                
                XQuery{l,1}=X_tmp(queryInds,:);
                LQuery{l,1}=L_tmp(queryInds,:);
                LQuery_All{l,1}=L_all_tmp(queryInds,:);
                
                K{l,1}=nus_gt_vec(label_allow,:);
                K_All{l,1}=nus_gt_vec;
                
                train_param.chunksize{l,1}=size(sampleInds,2);
                train_param.test_chunksize{l,1}=size(queryInds,2);
            end
            
        elseif strcmp(train_param.load_type, 'third_setting')
            X = [I_tr T_tr; I_te T_te];
            L = [L_tr; L_te];
            
            
            train_param.nchunks=10;
            
            labels=linspace(1,21,21);
            seperate=cell(train_param.nchunks,1);
            seperate{1,1}=[1,21];
            seperate{2,1}=[2,20];
            seperate{3,1}=[3,19];
            seperate{4,1}=[4,18];
            seperate{5,1}=[5,17];
            seperate{6,1}=[6,16];
            seperate{7,1}=[7,15];
            seperate{8,1}=[8,14];
            seperate{9,1}=[9,13];
            seperate{10,1}=[10,11,12];
            
            train_param.chunksize = cell(train_param.nchunks,1);
            train_param.test_chunksize = cell(train_param.nchunks,1);

            XTrain = cell(train_param.nchunks,1);
            LTrain = cell(train_param.nchunks,1);
            LTrain_All = cell(train_param.nchunks,1);

            XQuery = cell(train_param.nchunks,1);
            LQuery = cell(train_param.nchunks,1);
            LQuery_All = cell(train_param.nchunks,1);

            K = cell(train_param.nchunks,1);
            K_All= cell(train_param.nchunks,1);
            
            label_appear=[];
            
            for l=1:train_param.nchunks
                label_allow=seperate{l,1};
                label_appear=[label_allow label_appear];
                label_notallow=setdiff(labels,label_allow);
                idx_find=find(sum(L(:,label_notallow),2)==0);
                
                R = randperm(size(idx_find,1));
                queryInds = R(1,1:floor(size(R,2)*0.1));
                sampleInds = R(1,floor(size(R,2)*0.1)+1:end);
                
                X_tmp=X(idx_find,:);
                L_tmp=L(idx_find,label_appear);
                L_all_tmp=L(idx_find,:);
                
                XTrain{l,1}=X_tmp(sampleInds,:);
                LTrain{l,1}=L_tmp(sampleInds,:);
                LTrain_All{l,1}=L_all_tmp(sampleInds,:);
                
                XQuery{l,1}=X_tmp(queryInds,:);
                LQuery{l,1}=L_tmp(queryInds,:);
                LQuery_All{l,1}=L_all_tmp(queryInds,:);
                
                K{l,1}=nus_gt_vec(label_appear,:);
                K_All{l,1}=nus_gt_vec;
                
                train_param.chunksize{l,1}=size(sampleInds,2);            
                train_param.test_chunksize{l,1}=size(queryInds,2);
            end
        end  

        clear X L subi queryInds sampleInds R
        
    end
    fprintf('-------load data finished-------\n');
    clear I_tr I_te L_tr L_te T_tr T_te nus_gt_vec mir_gt_vec nus_gt mir_gt
end

