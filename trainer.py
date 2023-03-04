import torch
import utils as u
import logger
import brain_logger
import time
import pandas as pd
import numpy as np
import dgl
import networkx as nx
from models import WaveletSparsifier

class Trainer():
	def __init__(self,args, splitter, gcn, classifier, comp_loss, dataset, num_classes):
		self.args = args
		self.splitter = splitter
		self.tasker = splitter.tasker
		self.gcn = gcn
		self.classifier = classifier
		self.comp_loss = comp_loss

		self.num_nodes = dataset.num_nodes
		self.data = dataset
		self.num_classes = num_classes

		if args.data=='brain':
			self.logger = brain_logger.Logger(args, self.num_classes)
		else:
			self.logger = logger.Logger(args, self.num_classes)

		self.init_optimizers(args)

		if self.tasker.is_static:
			adj_matrix = u.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size = [self.num_nodes], ignore_batch_dim = False)
			self.hist_adj_list = [adj_matrix]
			self.hist_ndFeats_list = [self.tasker.nodes_feats.float()]

	def init_optimizers(self,args):
		params = self.gcn.parameters()
		self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
		params = self.classifier.parameters()
		self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
		self.gcn_opt.zero_grad()
		self.classifier_opt.zero_grad()

	def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
		torch.save(state, filename)

	def load_checkpoint(self, filename, model):
		if os.path.isfile(filename):
			print("=> loading checkpoint '{}'".format(filename))
			checkpoint = torch.load(filename)
			epoch = checkpoint['epoch']
			self.gcn.load_state_dict(checkpoint['gcn_dict'])
			self.classifier.load_state_dict(checkpoint['classifier_dict'])
			self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
			self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
			self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
			return epoch
		else:
			self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
			return 0

	def train(self):
		self.tr_step = 0
		best_eval_valid = 0
		eval_valid = 0
		epochs_without_impr = 0

		for e in range(self.args.num_epochs):

			import time
			start_time = time.time()

			eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)
			
			end_time = time.time()
			print('======= Time taken for 1 epoch: ',end_time-start_time)

			if len(self.splitter.dev)>0 and e>self.args.eval_after_epochs:
				eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad = False)
				if eval_valid>best_eval_valid:
					best_eval_valid = eval_valid
					epochs_without_impr = 0
					print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Best valid measure:'+str(eval_valid))
				else:
					epochs_without_impr+=1
					if epochs_without_impr>self.args.early_stop_patience:
						print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Early stop.')
						break

			if len(self.splitter.test)>0 and eval_valid==best_eval_valid and e>self.args.eval_after_epochs:
				eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = False)

				if self.args.save_node_embeddings:
					self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, log_file+'_valid_nodeembs.csv.gz')
					self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')


	def run_epoch(self, split, epoch, set_name, grad):
		t0 = time.time()
		log_interval=999
		if set_name=='TEST':
			log_interval=1
		self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

		torch.set_grad_enabled(grad)
		for sidx, s in enumerate(split):
			if self.tasker.is_static:
				s = self.prepare_static_sample(s)
			else:
				if 'transformer' in self.args.model:
					if self.args.transformer_parameters.get('lap_pos_enc',False):
						pos_enc_list = split.pos_enc_list[sidx]
					else:
						pos_enc_list = []
					if self.args.model=='temporal_transformer':
						use_graph_list = False
					else:
						use_graph_list = True
					if self.args.transformer_parameters['lap_pos_enc']:
						graph_pe = 'lap_pos_enc'
					else:
						graph_pe = ''
					s = self.prepare_sample(s,graph_list=use_graph_list,graph_pe=graph_pe,pos_enc_list=pos_enc_list,full_graph=self.args.transformer_parameters.get('full_graph',False))
				elif self.args.model=='GWNN':
					s = self.prepare_sample(s, graph_list=True, full_graph=False)
				elif self.args.model=='PNA' or self.args.model=='GraphSAGE':
					s = self.prepare_sample(s, graph_list=True, full_graph=False)
				else:
					s = self.prepare_sample(s)

			if 'transformer' in self.args.model:
				if self.args.model=='DEFT':
					predictions, nodes_embs = self.predict(s.hist_adj_list,s.hist_ndFeats_list,s.label_sp['idx'],s.node_mask_list,graph_list=s.graph_list,hist_adj_list_u=s.hist_adj_list_u)
				elif self.args.model=='DEFT_h':
					predictions, nodes_embs = self.predict(s.hist_adj_list,s.hist_ndFeats_list,s.label_sp['idx'],s.node_mask_list,graph_list=s.graph_list,hist_adj_list_u=s.hist_adj_list_u)
				else:
					predictions, nodes_embs = self.predict(s.hist_adj_list,s.hist_ndFeats_list,s.label_sp['idx'],s.node_mask_list,graph_list=s.graph_list,pos_enc_list=s.pos_enc_list)
			elif self.args.model=='GWNN':
				predictions, nodes_embs = self.predict_gwnn(s.graph_list[-1], s.hist_ndFeats_list[-1], s.label_sp['idx'])
				# adj['idx'].t(), adj['vals'].type(torch.float)
			elif self.args.model=='PNA' or self.args.model=='GraphSAGE':
				predictions, nodes_embs = self.predict(s.hist_adj_list,s.hist_ndFeats_list,s.label_sp['idx'],s.node_mask_list,graph_list=s.graph_list)
			else:
				predictions, nodes_embs = self.predict(s.hist_adj_list,s.hist_ndFeats_list,s.label_sp['idx'],s.node_mask_list)

			loss = self.comp_loss(predictions,s.label_sp['vals'])
			# print(loss)
			if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			else:
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
			if grad:
				self.optim_step(loss)

		torch.set_grad_enabled(True)
		eval_measure = self.logger.log_epoch_done()

		return eval_measure, nodes_embs

	def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,mask_list,graph_list=[],pos_enc_list=[],hist_adj_list_u=[]):
		if len(graph_list)>0:
			if len(hist_adj_list_u)>0:
				nodes_embs = self.gcn(hist_adj_list,
							  hist_ndFeats_list,
							  mask_list,graph_list,hist_adj_list_u)
			else:
				nodes_embs = self.gcn(hist_adj_list,
							  hist_ndFeats_list,
							  mask_list,graph_list=graph_list,pos_enc_list=pos_enc_list)
		else:
			nodes_embs = self.gcn(hist_adj_list,
							  hist_ndFeats_list,
							  mask_list)

		predict_batch_size = 100000
		gather_predictions=[]
		for i in range(1 +(node_indices.size(1)//predict_batch_size)):
			cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
			predictions = self.classifier(cls_input)
			gather_predictions.append(predictions)
		gather_predictions=torch.cat(gather_predictions, dim=0)
		return gather_predictions, nodes_embs

	def predict_gwnn(self, graph, features, node_indices):
		graph = graph.cpu().to_networkx()
		scale = 1
		tolerance = 1e-4
		approximation_order = 3
		sparsifier = WaveletSparsifier(graph, scale, approximation_order, tolerance)
		sparsifier.calculate_all_wavelets()
		# take features at last timestep for this static case
		# features = s.hist_ndFeats_list[-1]
		phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, feature_indices, feature_values = self.gcn.setup_features(sparsifier, features)

		nodes_embs = self.gcn(phi_indices, phi_values, phi_inverse_indices, phi_inverse_values, feature_indices, feature_values)

		predict_batch_size = 100000
		gather_predictions=[]
		for i in range(1 +(node_indices.size(1)//predict_batch_size)):
			cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
			predictions = self.classifier(cls_input)
			gather_predictions.append(predictions)
		gather_predictions=torch.cat(gather_predictions, dim=0)
		return gather_predictions, nodes_embs

	def gather_node_embs(self,nodes_embs,node_indices):
		cls_input = []

		for node_set in node_indices:
			cls_input.append(nodes_embs[node_set])
		return torch.cat(cls_input,dim = 1)

	def optim_step(self,loss):
		self.tr_step += 1
		loss.backward()

		if self.tr_step % self.args.steps_accum_gradients == 0:
			self.gcn_opt.step()
			self.classifier_opt.step()

			self.gcn_opt.zero_grad()
			self.classifier_opt.zero_grad()

	def make_full_graph(self, g):

	    
	    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

	    #Here we copy over the node feature data and laplace encodings
	    if 'feat' in g.ndata:
	    	full_g.ndata['feat'] = g.ndata['feat']

	    try:
	        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
	        full_g.ndata['EigVals'] = g.ndata['EigVals']
	    except:
	        pass
	    
	    #Populate edge features w/ 0s
	    full_g.edata['feat']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
	    full_g.edata['real']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
	    
	    #Copy real edge data over
	    if 'feat' in g.edata:
	    	full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
	    	# full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 
	    	full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(g.number_of_edges(), dtype=torch.long) 
	    	
	    
	    return full_g

	def prepare_sample(self,sample,graph_pe='',graph_list=False,pos_enc_list=[],full_graph=False):
		sample = u.Namespace(sample)
		sample.graph_list = []
		if graph_pe=='lap_pos_enc':
			sample.pos_enc_list = []
		else:
			sample.pos_enc_list = []
		for i,adj in enumerate(sample.hist_adj_list):
			adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])
			sample.hist_adj_list[i] = adj.to(self.args.device)
			if graph_pe!='':
				if graph_pe=='lap_pos_enc':
					pe = pos_enc_list[i].ndata['lap_pos_enc']
					sample.pos_enc_list.append(torch.tensor(pe).to(self.args.device))
				else:
					raise('Graph PE {} not implemented'.format(graph_pe))

			nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

			sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
			node_mask = sample.node_mask_list[i]
			sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

			if graph_list:
				if not full_graph:
					# g = g.cpu().detach().numpy()
					src, dst = adj._indices()[0,:], adj._indices()[1,:]
					g = dgl.graph((src, dst)).to(self.args.device)
					sample.graph_list.append(g)
				else:
					num_nodes = adj.shape[0]
					src = [i for i in range(num_nodes)]
					dst = [i for i in range(num_nodes)]
					for i in range(num_nodes):
						for j in range(i+1,num_nodes):
							src.append(i);dst.append(j)
							src.append(j);dst.append(i)
					g = dgl.graph((src, dst)).to(self.args.device)

					sample.graph_list.append(g)

		label_sp = self.ignore_batch_dim(sample.label_sp)

		if self.args.task in ["link_pred", "edge_cls"]:
			label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   
		else:
			label_sp['idx'] = label_sp['idx'].to(self.args.device)

		label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
		sample.label_sp = label_sp

		return sample

	def prepare_static_sample(self,sample):
		sample = u.Namespace(sample)

		sample.hist_adj_list = self.hist_adj_list

		sample.hist_ndFeats_list = self.hist_ndFeats_list

		label_sp = {}
		label_sp['idx'] =  [sample.idx]
		label_sp['vals'] = sample.label
		sample.label_sp = label_sp

		return sample

	def ignore_batch_dim(self,adj):
		if self.args.task in ["link_pred", "edge_cls"]:
			adj['idx'] = adj['idx'][0]
		adj['vals'] = adj['vals'][0]
		return adj

	def save_node_embs_csv(self, nodes_embs, indexes, file_name):
		csv_node_embs = []
		for node_id in indexes:
			orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

			csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

		pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
		#print ('Node embs saved in',file_name)