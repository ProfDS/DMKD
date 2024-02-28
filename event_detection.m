clear;clc;

fid = fopen('edgelist.txt', 'rt');
datacell = textscan(fid, '%f %f %f', 'Delimiter',' ', 'CollectOutput', 1);
fclose(fid);
data = datacell{1};    % as a numeric array


size(data)
tot_edges = size(data,1);
W = ones(tot_edges,1);
data = [data W];
%plot(G)

window_size = 4;

F = 12; % 13 #features
%t = 1;  %% timestamp

edges = data(data(:,1)==1,[2:end]);
N = max(edges(:));
A = sparse(edges(:,1),edges(:,2),edges(:,3),N,N);
M = size(edges,1);

features_c = zeros(N,F);

features_c(:,1) = sum(A~=0,1);% ./ N; %in_degree
features_c(:,2) = sum(A~=0,2);% ./ N; %out_degree
features_c(:,3) = sum(A,1);% ./ M; %in_edges
features_c(:,4) = sum(A,2);% ./ M; %out_edges
	
[r,c] = find(A);
egos = accumarray(r,c,[size(A,1), 1],@(x) {sort(x).'});
node_idx = [1:length(A)]';
node_idx = num2cell(node_idx);
egos  = cellfun(@(x, y)[x,y], node_idx,egos,'UniformOutput',false); % append ego node in the neighborhood
ego_matrix = cellfun(@(x)A(x,x),egos,'UniformOutput',false);
ego_size = cellfun(@numel,egos);	% #nodes in egonets
ego_edges = cellfun(@numel,cellfun(@find,ego_matrix,'UniformOutput',false));	% #edge count in egonets
ego_weights = cellfun(@sum,cellfun(@sum,ego_matrix,'UniformOutput',false),'UniformOutput',false);	% #total edge weight in egonets
ego_weights = cell2mat(ego_weights);

boundary_in_nodes = zeros(length(egos),1);
boundary_out_nodes = zeros(length(egos),1);
boundary_in_edges = zeros(length(egos),1);
boundary_out_edges = zeros(length(egos),1);
boundary_in_weights = zeros(length(egos),1);
boundary_out_weights = zeros(length(egos),1);


%% ego-boundary edges and weights
for i=1:length(egos)

	ego_nodes = egos{i};
	
	temp_nodes = setdiff([1:N],ego_nodes);
	[r,~] = find(A(temp_nodes,ego_nodes));
	boundary_in_nodes(i) = numel(unique(r));
	
	[r,c,~] = find(A(ego_nodes,temp_nodes));
	boundary_out_nodes(i) = numel(unique(c));
	
	B = A(:,ego_nodes);
	boundary_in_edges(i) = numel(find(B));
	boundary_in_edges(i) = boundary_in_edges(i) - ego_edges(i); % #boundary-in edge count
	
	boundary_in_weights(i) = sum(B(:));
	boundary_in_weights(i) = boundary_in_weights(i) - ego_weights(i);	% #boundary-in edge weight count
	
	B = A(ego_nodes,:);
	boundary_out_edges(i) = numel(find(B));
	boundary_out_edges(i) = boundary_out_edges(i) - ego_edges(i);	% #boundary-out edge count

	boundary_out_weights(i) = sum(B(:));
	boundary_out_weights(i) = boundary_out_weights(i) - ego_weights(i);	% #boundary-out edge weight count
end


features_c(:,5) = ego_edges;% ./ ego_size;
features_c(:,6) = ego_weights;% ./ ego_size;
features_c(:,7) = boundary_in_nodes;% ./ ego_size;
features_c(:,8) = boundary_out_nodes ;%./ ego_size;
features_c(:,9) = boundary_in_edges ;%./ ego_size;
features_c(:,10) = boundary_in_weights ;%./ ego_size;
features_c(:,11) = boundary_out_edges ;%./ ego_size;
features_c(:,12) = boundary_out_weights;% ./ ego_size;
feature_matrix = cell(N,1);
for i=1:N
	feature_matrix{i}(:,1) = features_c(i,:);
end
 
meanFeatures = [mean(features_c,1)'];

%---------------------------------------------------------------------------------edge streams-------------------------------------------------------------------------------


%%%%%%%%% 4. Define LSTM Network Architecture
numHiddenUnits = 200;

layers = [ ...
sequenceInputLayer(F)
%lstmLayer(numHiddenUnits)
lstmLayer(numHiddenUnits)
dropoutLayer(0.2)
%lstmLayer(100)
%dropoutLayer(0.2)
fullyConnectedLayer(F)
regressionLayer];

%%%%%%%%% 4.1 Specify the training options. Set the solver to 'adam' and train for 200 epochs. To prevent the gradients from exploding, set the gradient threshold to 1. Specify the initial learn rate 0.005, and drop the learn rate after 125 epochs by multiplying by a factor of 0.2.
miniBatchSize = 128;

options = trainingOptions('adam', ...
'MaxEpochs',200, ...
'MiniBatchSize',miniBatchSize, ...
'GradientThreshold',1, ...
'InitialLearnRate',0.001, ...
'SequencePaddingDirection', 'left', ...
'SequenceLength','longest', ...
'LearnRateSchedule','piecewise', ...
'LearnRateDropPeriod',127, ...
'LearnRateDropFactor',0.1, ...
'Verbose',1, ...
'VerboseFrequency', 20,...
'Shuffle', 'every-epoch', ...
'L2Regularization',0.0005,...
'Plots','none');


num_timestampes = numel(unique(data(:,1)))
t=2;
result = [];
while t<= num_timestampes %and(t <= window_size+2,t<=num_timestampes)
	edges = data(data(:,1)==t,[2:end]);
	N_old = N;
	N = max(N_old, max(edges(:))); 	%%%total nodes received till time t
	new_nodes = N - N_old;
	%%% add zeros rows and columns in adjacency matrix for new nodes
	A = [A;zeros(new_nodes,size(A,2))];
	A = [A zeros(size(A,1),new_nodes)];
	M = M + size(edges,1);
	egos_to_update = [];
	for i = 1:size(edges,1)
		u = edges(i,1);
		v = edges(i,2);
		w = edges(i,3);
		A(u,v) = A(u,v) + w;
		idx = [find(A(u,:)),find(A(:,u))',find(A(v,:)),find(A(:,v))'];
		egos_to_update =  union(egos_to_update,idx);
	end
	egos_to_update = sort(unique(egos_to_update))';
	disp(sprintf('Timestep = %d, New nodes = %d, New edges = %d,Total nodes = %d, Total edges = %d, Nodes to update = %d', t, new_nodes, size(edges,1), N, M, numel(egos_to_update)));
	%egos_to_update
	features_c = zeros(N,F);

	%%%%update node's features
	features_c(:,1) = sum(A~=0,1); % ./ N; %in_degree
	features_c(:,2) = sum(A~=0,2); %  ./ N; %out_degree
	features_c(:,3) = sum(A,1); %  ./ M; %in_edges
	features_c(:,4) = sum(A,2); % ./ M; %out_edges

	%%update egonet features
	for j = 1:length(egos_to_update)
		node = egos_to_update(j);
		
		[~,c] = find(A(node,:));
		egos{node} = unique([node c]); %update ego neighbors
		ego = egos{node};
		
		% update community features
		ego_matrix = A(ego,ego);
		ego_size(node) = numel(ego);	% #nodes in egonets
		ego_edges(node) = numel(find(ego_matrix));	
		ego_weights(node) = sum(sum(ego_matrix));	
		
		% update boundary features		
		temp_nodes = setdiff([1:N],ego);
		[r,~] = find(A(temp_nodes,ego));
		boundary_in_nodes(node) = numel(unique(r));
	
		[r,c,~] = find(A(ego,temp_nodes));
		boundary_out_nodes(node) = numel(unique(c));
	
		B = A(:,ego);
		boundary_in_edges(node) = numel(find(B));
		boundary_in_edges(node) = boundary_in_edges(node) - ego_edges(node);	% #boundary-in edge count
	
		boundary_in_weights(node) = sum(B(:));
		boundary_in_weights(node) = boundary_in_weights(node) - ego_weights(node);	% #boundary-in edge weight count
	
		B = A(ego,:);
		boundary_out_edges(node) = numel(find(B));
		boundary_out_edges(node) = boundary_out_edges(node) - ego_edges(node);	% #boundary-out edge count

		boundary_out_weights(node) = sum(B(:));
		boundary_out_weights(node) = boundary_out_weights(node) - ego_weights(node);	% #boundary-out edge weight count
	end	
	features_c(:,5) = ego_edges; %  ./ ego_size;
	features_c(:,6) = ego_weights ; % ./ ego_size;
	features_c(:,7) = boundary_in_nodes; %  ./ ego_size;
	features_c(:,8) = boundary_out_nodes; %  ./ ego_size;
	features_c(:,9) = boundary_in_edges; %  ./ ego_size;
	features_c(:,10) = boundary_in_weights; %  ./ ego_size;
	features_c(:,11) = boundary_out_edges ; % ./ ego_size;
	features_c(:,12) = boundary_out_weights ; % ./ ego_size;
	
	for i=1:N-new_nodes
		feature_matrix{i}(:,end+1) = features_c(i,:);
	end
	for i=N-new_nodes+1:N  %%%new nodes
		feature_matrix{i}(:,end+1) = features_c(i,:);
	end
	meanFeatures = [meanFeatures, mean(features_c(N-new_nodes+1:N,:),1)'];
	meanFeatures(isnan(meanFeatures)) = 0';

	if t >= window_size+2
		dataTrain = cellfun(@(x) x(:,max(1,end-window_size-1):end-1), feature_matrix, 'UniformOutput', false);
		dataTest = cellfun(@(x) x(:,max(2,end-window_size):end), feature_matrix, 'UniformOutput', false);
		temp1 = dataTrain(N-new_nodes+1:N);
		temp2 = dataTest(N-new_nodes+1:N);
		dataTrain = dataTrain(1:N-new_nodes);
		dataTest(N-new_nodes+1:N)=cellfun(@(x,y) [x,y], temp1, temp2,'UniformOutput', false);
		
		dataTest = dataTest(egos_to_update);

		meanFeatures = meanFeatures(:,2:end);
		[net nodes_score dataTrainStandardized dataTestStandardized YPred YTest meanerror maximum mean_score max_zscore] = rnn_test(egos_to_update, N, F, t, feature_matrix, dataTrain, dataTest,layers,options,miniBatchSize);
		result = [result;t meanerror maximum mean_score max_zscore]
	end
	t = t+1;
	
end

