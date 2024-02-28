clear;clc;

num_edges = [50,30,25,35,50,20,35];
num_nodes = [25,35,45,58,65,80,100];
y = [25,2,45,58,65,80,100];
t = [1:7];
data = [];
for i=1:length(num_edges)
 data = [data; repmat(t(i),num_edges(i),1),randi(y(i),num_edges(i),1),randi([1 num_nodes(i)],num_edges(i),1), ones(num_edges(i),1)];
end
data = [data;repmat(8,20,1)  randi([2 10],20,1) randi(2,20,1) ones(20,1); repmat(9,10,1) randi([1 105],10,1) randi([1 105],10,1) ones(10,1)];
data = [data; repmat(10,25,1),randi([25 110],25,1),randi(2,25,1), ones(25,1)];

size(data)
data = data(data(:,2) ~= data(:,3),:);
sd = data(:,[2,3]);
unique_id = unique(sd','stable');
n = length(unique_id);
[~,sd] = ismember(sd, unique_id);

data = [data(:,1) sd data(:,4)];

size(data)
tot_edges = size(data,1);
F = 12; 


edges = data(data(:,1)==1,[2:end]);
N = max(edges(:));
A = sparse(edges(:,1),edges(:,2),edges(:,3),N,N);
M = size(edges,1);
t_O = ones(N,1);

features_c(:,1) = sum(A~=0,1) ./ N; %in_degree
features_c(:,2) = sum(A~=0,2) ./ N; %out_degree
features_c(:,3) = sum(A,1)./ M; %in_edges
features_c(:,4) = sum(A,2) ./ M; %out_edges
	
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

features_c(:,5) = ego_edges./ ego_size;
features_c(:,6) = ego_weights ./ ego_size;
features_c(:,7) = boundary_in_nodes ./ ego_size;
features_c(:,8) = boundary_out_nodes ./ ego_size;
features_c(:,9) = boundary_in_edges ./ ego_size;
features_c(:,10) = boundary_in_weights ./ ego_size;
features_c(:,11) = boundary_out_edges ./ ego_size;
features_c(:,12) = boundary_out_weights ./ ego_size;

%---------------------------------------------------------------------------------edge streams-------------------------------------------------------------------------------
avg_score = [];
max_zscore = [];
features_a = zeros(N,F);
graph_scores = [];
num_timestampes = numel(unique(data(:,1)))
zscores = [];
t=2;
while t<= num_timestampes
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
	features_s = features_c;
	features_c = zeros(N,F);
	%%%%update node's features
	features_c(:,1) = sum(A~=0,1)./ N; %in_degree
	features_c(:,2) = sum(A~=0,2)./ N; %out_degree
	features_c(:,3) = sum(A,1)./ M; %in_edges
	features_c(:,4) = sum(A,2)./ M; %out_edges
	
	%%update egonet features
	for i = 1:length(egos_to_update)
		node = egos_to_update(i);
		
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
	features_c(:,5) = ego_edges ./ ego_size;
	features_c(:,6) = ego_weights ./ ego_size;
	features_c(:,7) = boundary_in_nodes  ./ ego_size;
	features_c(:,8) = boundary_out_nodes ./ ego_size;
	features_c(:,9) = boundary_in_edges ./ ego_size;
	features_c(:,10) = boundary_in_weights ./ ego_size;
	features_c(:,11) = boundary_out_edges ./ ego_size;
	features_c(:,12) = boundary_out_weights ./ ego_size;
	[features_s sum(features_s,2)];
	[features_c sum(features_c,2)];
	
	f_s = features_s;
	f_c = features_c;
	
	temp = [features_c;features_s];
	temp = (temp - min(temp)) ./ (max(temp)-min(temp));
	
	%f_s =  (f_s - min(f_s)) ./ (max(f_s)-min(f_s));
	%f_c =  (f_c - min(f_c)) ./ (max(f_c)-min(f_c));

	f_c = temp(1:N,:);
	f_s = temp(N+1:end,:);
	
	f_c = f_c+0.01; %sum(features_c,2);
	f_s = f_s+0.01; %sum(features_s,2);
	
	
	f_a = abs(f_c(1:N_old,:) - f_s);
	f_a = [f_a; f_c(N_old+1:end,:)];
	
	f_c = f_c(egos_to_update,:);
	f_a = f_a(egos_to_update,:);
	
	t_O = t_O + 1;
	t_O = [t_O ; ones(new_nodes,1)];

	t_Ot = t_O(egos_to_update);
	

	score = ((f_a - (f_c ./ t_Ot)).^2) .* ((t_Ot) ./ (f_c .* (t_Ot-1)));

	tmp = find(t_Ot==1);
	score(tmp,:) = (f_a(tmp,:) - f_c(tmp,:)).^2 ./ f_c(tmp,:);

	score = sum(score,2);	
	[max_anomaly,idx] = max(score);
	[ss id]=sort(score);
	s = sort(score,'desc');
	s = s(find(s>0));
	avg_score = [avg_score mean(s)];
	max_zscore = [max_zscore max(zscore(s))];

	disp(sprintf('timesemp = %d, anomaly score = %4.8f, zscore = %4.8f, node = %d, average score = %4.8f\n ', t, max_anomaly, max(zscore(s)),egos_to_update(idx),mean(s)));
	t = t+1;
end
graph_scores = [2 0 avg_score(1)-avg_score(2)];
graph_zscores = [2 0 max_zscore(1)-max_zscore(2)];


for i = 2:numel(avg_score)-1
	graph_scores = [graph_scores; i+1 avg_score(i)-avg_score(i-1) avg_score(i) - avg_score(i+1)];
	graph_zscores = [graph_zscores; i+1 max_zscore(i)-max_zscore(i-1) max_zscore(i) - max_zscore(i+1)];
end
i=i+1;
graph_scores = [graph_scores; i+1 avg_score(i)-avg_score(i-1) 0];
graph_zscores = [graph_zscores; i+1 max_zscore(i)-max_zscore(i-1) 0];


[graph_scores graph_scores(:,2)+graph_scores(:,3)]
[graph_zscores graph_zscores(:,2)+graph_zscores(:,3)]
