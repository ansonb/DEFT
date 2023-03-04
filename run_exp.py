import utils as u
import torch
import torch.distributed as dist
import numpy as np
import time
import random
import torch_geometric
import dgl

#datasets
import bitcoin_dl as bc
import elliptic_temporal_dl as ell_temp
import uc_irv_mess_dl as ucim
import auto_syst_dl as aus
import sbm_dl as sbm
import reddit_dl as rdt
import brain_dl as brain


#taskers
import link_pred_tasker as lpt
import edge_cls_tasker as ect
import node_cls_tasker as nct
import brain_node_cls_tasker as brain_nct

#models
import models as mls
import egcn_h
import egcn_o
import models_transformer
import model_deft, model_deft_h

import splitter as sp
import Cross_Entropy as ce

import trainer as tr

import logger

def random_param_value(param, param_min, param_max, type='int'):
	if str(param) is None or str(param).lower()=='none':
		if type=='int':
			return random.randrange(param_min, param_max+1)
		elif type=='logscale':
			interval=np.logspace(np.log10(param_min), np.log10(param_max), num=100)
			return np.random.choice(interval,1)[0]
		else:
			return random.uniform(param_min, param_max)
	else:
		return param

def build_random_hyper_params(args):
	if args.model == 'all':
		model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_nogcn':
		model_types = ['egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_noegcn3':
		model_types = ['gcn', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
	elif args.model == 'all_nogruA':
		model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruB','egcn','lstmA', 'lstmB']
		args.model=model_types[args.rank]
		args.model=model_types[args.rank]
	elif args.model == 'saveembs':
		model_types = ['gcn', 'gcn', 'skipgcn', 'skipgcn']
		args.model=model_types[args.rank]

	args.learning_rate =random_param_value(args.learning_rate, args.learning_rate_min, args.learning_rate_max, type='logscale')
	# args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')

	if args.model == 'gcn':
		args.num_hist_steps = 0
	else:
		args.num_hist_steps = random_param_value(args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type='int')

	args.gcn_parameters['feats_per_node'] =random_param_value(args.gcn_parameters['feats_per_node'], args.gcn_parameters['feats_per_node_min'], args.gcn_parameters['feats_per_node_max'], type='int')
	args.gcn_parameters['layer_1_feats'] =random_param_value(args.gcn_parameters['layer_1_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
	if args.gcn_parameters['layer_2_feats_same_as_l1'] or str(args.gcn_parameters['layer_2_feats_same_as_l1']).lower()=='true':
		args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
	else:
		args.gcn_parameters['layer_2_feats'] =random_param_value(args.gcn_parameters['layer_2_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
	args.gcn_parameters['lstm_l1_feats'] =random_param_value(args.gcn_parameters['lstm_l1_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
	if args.gcn_parameters['lstm_l2_feats_same_as_l1'] or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower()=='true':
		args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
	else:
		args.gcn_parameters['lstm_l2_feats'] =random_param_value(args.gcn_parameters['lstm_l2_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
	args.gcn_parameters['cls_feats']=random_param_value(args.gcn_parameters['cls_feats'], args.gcn_parameters['cls_feats_min'], args.gcn_parameters['cls_feats_max'], type='int')

	# if 'transformer' in args.model:
	# 	args.transformer_parameters = args.transformer_parameters
	return args

def build_dataset(args):
	if args.data == 'bitcoinotc' or args.data == 'bitcoinalpha':
		if args.data == 'bitcoinotc':
			args.bitcoin_args = args.bitcoinotc_args
		elif args.data == 'bitcoinalpha':
			args.bitcoin_args = args.bitcoinalpha_args
		return bc.bitcoin_dataset(args)
	elif args.data == 'aml_sim':
		return aml.Aml_Dataset(args)
	elif args.data == 'elliptic':
		return ell.Elliptic_Dataset(args)
	elif args.data == 'elliptic_temporal':
		return ell_temp.Elliptic_Temporal_Dataset(args)
	elif args.data == 'uc_irv_mess':
		return ucim.Uc_Irvine_Message_Dataset(args)
	elif args.data == 'dbg':
		return dbg.dbg_dataset(args)
	elif args.data == 'colored_graph':
		return cg.Colored_Graph(args)
	elif args.data == 'autonomous_syst':
		return aus.Autonomous_Systems_Dataset(args)
	elif args.data == 'reddit':
		return rdt.Reddit_Dataset(args)
	elif args.data.startswith('sbm'):
		if args.data == 'sbm20':
			args.sbm_args = args.sbm20_args
		elif args.data == 'sbm50':
			args.sbm_args = args.sbm50_args
		return sbm.sbm_dataset(args)
	elif args.data == 'brain':
		return brain.Brain_Dataset(args)
	else:
		raise NotImplementedError('only arxiv has been implemented')

def build_tasker(args,dataset):
	if args.task == 'link_pred':
		return lpt.Link_Pred_Tasker(args,dataset)
	elif args.task == 'edge_cls':
		return ect.Edge_Cls_Tasker(args,dataset)
	elif args.task == 'node_cls':
		if args.data=='brain':
			return brain_nct.Node_Cls_Tasker(args,dataset)
		else:
			return nct.Node_Cls_Tasker(args,dataset)
	elif args.task == 'static_node_cls':
		return nct.Static_Node_Cls_Tasker(args,dataset)

	else:
		raise NotImplementedError('still need to implement the other tasks')

def build_gcn(args,tasker):
	gcn_args = u.Namespace(args.gcn_parameters)
	gcn_args.feats_per_node = tasker.feats_per_node
	gcn_args.use_2_hot_node_feats = args.use_2_hot_node_feats
	gcn_args.use_1_hot_node_feats = args.use_1_hot_node_feats 

	if 'transformer' in args.model:
		args.transformer_parameters['use_transformer'] = args.transformer_parameters.get('use_transformer', True)
		args.transformer_parameters['concat_in_skipfeat'] = args.transformer_parameters.get('concat_in_skipfeat', False)
		# args.transformer_parameters['rt_residual'] = args.transformer_parameters.get('rt_residual', True)
		args.transformer_parameters['skip_in_feat'] = args.transformer_parameters.get('skip_in_feat', True)
		args.transformer_parameters['use_spatial_feat_in_lpe'] = args.transformer_parameters.get('use_spatial_feat_in_lpe', False)
		args.transformer_parameters['use_spectral_in_lpe'] = args.transformer_parameters.get('use_spectral_in_lpe', True)
		args.transformer_parameters['num_filter_subspaces'] = args.transformer_parameters.get('num_filter_subspaces', 1)
		# Note: depends on use_spatial_feat_in_lpe being set to True to be used
		args.transformer_parameters['use_spatial_feat_in_rgt_ip'] = args.transformer_parameters.get('use_spatial_feat_in_rgt_ip', False)
		# Note: for this to work, rt_residual flag must also be true
		args.transformer_parameters['skip_rgt_in_feat'] = args.transformer_parameters.get('skip_rgt_in_feat', False)
		args.transformer_parameters['use_static_spectral_wavelets'] = args.transformer_parameters.get('use_static_spectral_wavelets', False)
		args.transformer_parameters['use_sgnn_dgl'] = args.transformer_parameters.get('use_sgnn_dgl', False)
		args.transformer_parameters['use_pe_module'] = args.transformer_parameters.get('use_pe_module', True)
		

		transformer_args = u.Namespace(args.transformer_parameters)
		transformer_args.feats_per_node = tasker.feats_per_node
		transformer_args.use_2_hot_node_feats = args.use_2_hot_node_feats
		transformer_args.use_1_hot_node_feats = args.use_1_hot_node_feats 

		transformer_args.aggregator = args.aggregator

	if args.model == 'gcn':
		return mls.Sp_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	elif args.model == 'skipgcn':
		return mls.Sp_Skip_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	elif args.model == 'skipfeatsgcn':
		return mls.Sp_Skip_NodeFeats_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	else:
		assert args.num_hist_steps > 0, 'more than one step is necessary to train LSTM'
		if args.model == 'lstmA':
			return mls.Sp_GCN_LSTM_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'gruA':
			return mls.Sp_GCN_GRU_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'lstmB':
			return mls.Sp_GCN_LSTM_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'gruB':
			return mls.Sp_GCN_GRU_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'egcn':
			return egcn.EGCN(gcn_args, activation = torch.nn.RReLU()).to(args.device)
		elif args.model == 'egcn_h':
			return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
		elif args.model == 'skipfeatsegcn_h':
			return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device, skipfeats=True)
		elif args.model == 'egcn_o':
			return egcn_o.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
		elif args.model == 'GWNN':
			return mls.GWNN(gcn_args, args.device).to(args.device)
		elif args.model == 'PNA':
			return mls.StaticPNA(gcn_args, activation = torch.nn.RReLU(), device = args.device, lappe=True, timepe=False).to(args.device)
		elif args.model == 'GraphSAGE':
			return mls.StaticSAGE(gcn_args, activation = torch.nn.RReLU(), device = args.device, lappe=True, timepe=False).to(args.device)
		elif args.model == 'spatial_transformer':
			return models_transformer.Transformer_LSTM(transformer_args, gcn_args, activation = torch.nn.RReLU(), device = args.device, lappe=True, timepe=False).to(args.device)
		elif args.model == 'temporal_transformer':
			# Note: timepe parameter is not being used; default time pe is being used
			return models_transformer.GCN_Transformer(transformer_args, gcn_args, activation = torch.nn.RReLU(), device = args.device, lappe=False, timepe=True).to(args.device)
		elif args.model == 'spatio_temporal_transformer':
			return models_transformer.Spatio_Temporal_Transformer(transformer_args, gcn_args, activation = torch.nn.RReLU(), device = args.device, lappe=True, timepe=True).to(args.device)
		elif args.model == 'static_spatial_transformer':
			return models_transformer.StaticGraphTransformer(transformer_args, gcn_args, activation = torch.nn.RReLU(), device = args.device, lappe=True, timepe=False).to(args.device)
		elif args.model == 'DEFT':
			return model_deft.DEFT(transformer_args, gcn_args, activation = torch.nn.RReLU(), device = args.device, data=args.data)
		elif args.model == 'DEFT_h':
			return model_deft_h.DEFT(transformer_args, gcn_args, activation = torch.nn.RReLU(), device = args.device)
		else:
			raise NotImplementedError('need to finish modifying the models')

def add_laplacian_positional_encodings(splitter, pos_enc_dim):
    # Graph positional encoding v/ Laplacian eigenvectors

    splitter.train.pos_enc_list = [[u.laplacian_positional_encoding(g['idx'], pos_enc_dim) for g in hist_graphs['hist_adj_list']] for hist_graphs in splitter.train]
    splitter.dev.pos_enc_list = [[u.laplacian_positional_encoding(g['idx'], pos_enc_dim) for g in hist_graphs['hist_adj_list']] for hist_graphs in splitter.dev]
    splitter.test.pos_enc_list = [[u.laplacian_positional_encoding(g['idx'], pos_enc_dim) for g in hist_graphs['hist_adj_list']] for hist_graphs in splitter.test]




def build_classifier(args,tasker):
	if 'node_cls' == args.task or 'static_node_cls' == args.task:
		mult = 1
	else:
		mult = 2
	if 'gru' in args.model or 'lstm' in args.model:
		in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
	elif args.model == 'skipfeatsgcn' or args.model == 'skipfeatsegcn_h':
		in_feats = (args.gcn_parameters['layer_2_feats'] + args.gcn_parameters['feats_per_node']) * mult
	elif 'GWNN' in args.model:
		in_feats = args.gcn_parameters['filters'] * mult
	elif 'spatio_temporal_transformer' in args.model:
		in_feats = args.transformer_parameters['out_dim'] * mult
	elif 'DEFT' in args.model:
		in_feats = args.transformer_parameters['out_dim'] * mult
	else:
		in_feats = args.gcn_parameters['layer_2_feats'] * mult

	return mls.Classifier(args,in_features = in_feats, out_features = tasker.num_classes).to(args.device)

def get_model_params(gcn):
	num_parameters = 0
	trainable_parameters = 0
	for param in gcn.parameters():
		if param.requires_grad:
			trainable_parameters += param.numel()
		num_parameters += param.numel()
	return num_parameters, trainable_parameters

if __name__ == '__main__':

	parser = u.create_parser()
	args = u.parse_args(parser)

	global rank, wsize, use_cuda
	args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
	args.device='cpu'
	if args.use_cuda:
		args.device='cuda'
	print ("use CUDA:", args.use_cuda, "- device:", args.device)
	try:
		dist.init_process_group(backend='mpi') #, world_size=4
		rank = dist.get_rank()
		wsize = dist.get_world_size()
		print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
		if args.use_cuda:
			torch.cuda.set_device(rank )  # are we sure of the rank+1????
			print('using the device {}'.format(torch.cuda.current_device()))
	except:
		rank = 0
		wsize = 1
		print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank,wsize)))

	if args.seed is None and args.seed!='None':
		seed = 123+rank#int(time.time())+rank
	else:
		if args.cmd_seed==-1:
			seed=args.seed#+rank
		else:
			seed=args.cmd_seed

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch_geometric.seed_everything(seed)
	dgl.seed(seed)


	args.seed=seed
	args.rank=rank
	args.wsize=wsize

	# Assign the requested random hyper parameters
	args = build_random_hyper_params(args)

	#build the dataset
	dataset = build_dataset(args)
	#build the tasker
	tasker = build_tasker(args,dataset)
	#build the splitter
	splitter = sp.splitter(args,tasker)
	if args.model in ['spatial_transformer','spatio_temporal_transformer','static_spatial_transformer'] and args.transformer_parameters['lap_pos_enc']:
		add_laplacian_positional_encodings(splitter, args.transformer_parameters['pos_enc_dim'])
	#build the models
	gcn = build_gcn(args, tasker)
	classifier = build_classifier(args,tasker)
	num_parameters, trainable_parameters = get_model_params(gcn)
	print('====== num_parameters: ',num_parameters)
	print('====== trainable_parameters: ',trainable_parameters)
	#build a loss
	cross_entropy = ce.Cross_Entropy(args,dataset).to(args.device)

	#trainer
	trainer = tr.Trainer(args,
						 splitter = splitter,
						 gcn = gcn,
						 classifier = classifier,
						 comp_loss = cross_entropy,
						 dataset = dataset,
						 num_classes = tasker.num_classes)

	trainer.train()
	if args.save_model:
		import os
		model_folder = args.model_folder
		os.makedirs(model_folder, exist_ok=True)
		trainer.save_checkpoint(trainer.gcn._parameters.state_dict(), os.path.join(model_folder,'gcn.pth'))
		trainer.save_checkpoint(trainer.classifier.state_dict(), os.path.join(model_folder,'classifier.pth'))
		trainer.save_checkpoint(trainer.gcn_opt.state_dict(), os.path.join(model_folder,'gcn_opt.pth'))
		trainer.save_checkpoint(trainer.classifier_opt.state_dict(), os.path.join(model_folder,'cls_opt.pth'))



