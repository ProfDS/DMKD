clear;clc;

%
fid = fopen('data.txt', 'rt');
tline = fgetl(fid);
datacell = textscan(fid, '%f %f %f', 'Delimiter',' ', 'CollectOutput', 1);
fclose(fid);
data = datacell{1};    %as a numeric array

N = max(max(data(:,[2,3])))
edges = data(data(:,1)==1,[2,3]);
w = ones(size(edges,1), 1);
A = sparse(edges(:,1),edges(:,2),w,N,N);
M = numel(find(A));

G = digraph(A);
%plot(G)

F = 13; % #features
features_c = zeros(N,F);

features_c(:,1) = sum(A~=0,1); %in_degree
features_c(:,2) = sum(A~=0,2); %out_degree
features_c(:,3) = sum(A,1); %in_weight
features_c(:,4) = sum(A,2); %out_weight

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

boundary_out_weights = zeros(length(egos),1);

%% ego-boundary edges and weights
for i=1:length(egos)
i
	ego_nodes = egos{i};
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

features_c(:,7) = ego_size;
features_c(:,8) = ego_edges;
features_c(:,9) = ego_weights;
features_c(:,10) = boundary_in_edges;
features_c(:,11) = boundary_in_weights;
features_c(:,12) = boundary_out_edges;
features_c(:,13) = boundary_out_weights;

%---------------------------------------------------------------------------------edge streams-------------------------------------------------------------------------------

features_a = zeros(N,F);
graph_scores = [];
num_timestampes = numel(unique(data(:,1)))

for timestamp = 2:num_timestampes
	new_edges = data(data(:,1)==timestamp,[2,3]);
	features_s = features_c;
	num_edges = size(new_edges,1);
	W = ones(num_edges,1);
	new_edges = [new_edges W];
		
	egos_to_update = [];
	for i = 1:size(new_edges,1)
		u = new_edges(i,1);
		v = new_edges(i,2);
		w = new_edges(i,3);
		A(u,v) = A(u,v) + w;
		[r c ~] = find([A(u,:)',A(:,u),A(v,:)',A(:,v)]);
		egos_to_update =  union(egos_to_update,unique([r;c])');
	end
	
	%%update node's features
	features_c(:,1) = sum(A~=0,1); %in_degree
	features_c(:,2) = sum(A~=0,2); %out_degree
	features_c(:,3) = sum(A,1); %in_weight
	features_c(:,4) = sum(A,2); %out_weight

	%%update egonet features
	for i = 1:length(egos_to_update)
		node = egos_to_update(i);
		
		[~,c] = find(A(node,:));
		egos{node} = [node c]; %update ego neighbors
		ego = egos{node};
		
		% update community features
		ego_matrix = A(ego,ego);
		ego_size(node) = numel(ego);	% #nodes in egonets
		ego_edges(node) = numel(find(ego_matrix));	
		ego_weights(node) = sum(sum(ego_matrix));	
		
		% update boundary features
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
	features_c(:,7) = ego_size;
	features_c(:,8) = ego_edges;
	features_c(:,9) = ego_weights;
	features_c(:,10) = boundary_in_edges;
	features_c(:,11) = boundary_in_weights;
	features_c(:,12) = boundary_out_edges;
	features_c(:,13) = boundary_out_weights;

	features_at = features_c - features_s;
	%features_a = features_at;
	features_at = abs(features_at);
	features_a(egos_to_update,:) = (features_a(egos_to_update,:) * rand()) + features_at(egos_to_update,:);
	
	avg_features_s = features_s ./ ego_size;
	avg_features_c = features_c ./ ego_size;
	
	avg_features_at = avg_features_c - avg_features_s;
	avg_features_at = abs(avg_features_at);
	avg_features_a = avg_features_at;
	
	score = ((avg_features_a(egos_to_update,:) - (avg_features_c(egos_to_update,:) ./ timestamp)).^2) .* ((timestamp^2) ./ (avg_features_c(egos_to_update,:).* (timestamp-1)));
	score(isnan(score))=0;
	score(isinf(score))=0;

	node_score = sum(score,2);
	znode_score = zscore(node_score).^2;

	[score node_score];
	disp(sprintf('timestamp = %d, num_edges = %d, max-score = %f, avg-score = %f, zmax-score = %f\n', timestamp, num_edges, max(node_score), mean(node_score), max(znode_score)));
	graph_scores = [graph_scores; timestamp, num_edges, max(node_score), mean(node_score), max(znode_score)];
end