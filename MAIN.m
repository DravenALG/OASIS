%% init the workspace
close all; clear; clc; warning off;

%% globel settings
train_param.current_bits = 16;
train_param.max_iter=5;

%% load dataset
train_param.ds_name='MIRFLICKR';  % NUSWIDE21  MIRFLICKR
train_param.load_type='first_setting'; %settings
% train_param.load_type='second_setting';
% train_param.load_type='third_setting'; 
[train_param,XTrain,LTrain,LTrain_All,XQuery,LQuery,LQuery_All,K,K_All] = load_dataset(train_param);

%% OASIS
train_param.current_hashmethod='OASIS';
OURparam = train_param;
OURparam.alpha = 100;
OURparam.beta = 0.01;
OURparam.gamma = 10;
OURparam.thet = 0.1;
OURparam.delta = 1;
[eva,t] = evaluate_OASIS(XTrain,LTrain,XQuery,LQuery,K,OURparam);
