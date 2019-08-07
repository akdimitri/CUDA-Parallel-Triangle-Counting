%% clen up
clear
clc

%% Load Matrix
% Matrix delaunay_n22
% cd '~/Desktop/Parallel Triangle Counting/data/delaunay_n22'
% load delaunay_n22.mat

% Matrix auto
% cd '~/Desktop/Parallel Triangle Counting/data/auto'
% load auto.mat

% Matrix great britain
% cd '~/Desktop/Parallel Triangle Counting/data/great_britain'
% load great-britain_osm.mat

% Matrix delaunay_n23
%cd '~/Desktop/Parallel Triangle Counting/data/delaunay_n23'
%load delaunay_n23.mat

% % fe_tooth
% cd '~/Desktop/Parallel Triangle Counting/data/fe_tooth'
% load fe_tooth.mat

% % fe_tooth
% cd '~/Desktop/Parallel Triangle Counting/data/144'
% load 144.mat

% citationCiteseer
% cd '~/Desktop/Parallel Triangle Counting/data/citationCiteseer'
% load citationCiteseer.mat

% road_central
% cd '~/Desktop/Parallel Triangle Counting/data/road_central'
% load road_central.mat

% germany_osm
% cd '~/Desktop/Parallel Triangle Counting/data/germany_osm'
% load germany_osm.mat

% road_usa
cd '~/Desktop/Parallel Triangle Counting/data/road_usa'
load road_usa.mat



% keep only adjacency matrix (logical values)
A = Problem.A > 0; 
clear Problem;

fprintf( '   - DONE\n');

%% TRIANGLE COUNTING

fprintf( '...triangle counting...\n' ); 
ticCnt = tic;

nT = full( sum( sum( A^2 .* A ) ) / 6 );

fprintf( '   - DONE: %d triangles found in %.2f sec\n', nT, toc(ticCnt) );

%% Find indices of non zero elements
%  COO Format
[row, col] = find(A);
COO = [row, col];
clearvars row col
COO = sortrows(COO, 1);
COO_base_0 = COO-1;
nnz = length(COO_base_0(:,1));
sparsity = nnz/(length(A)^2);
fprintf("COO format has been completed\n");
fprintf("Number of Non-Zero elements: %d\n", nnz);
fprintf("Length A: %d\n", length(A));
fprintf("Sparsity %f\n", sparsity);
%% Write COO Format to file
% fileID = fopen("./COO_format.txt", 'w');
% for i = 1:nnz
%    fprintf( fileID, "%d %d 1\n", COO_base_0(i,1), COO_base_0(i,2));
% end
% fclose(fileID);

%% CSR Format
CSR_rows = zeros(length(A) + 1, 1);
CSR_rows(1) = 1;
temp = tabulate(COO(:,1));
for i=1:(length(CSR_rows) - 1)
   CSR_rows(i+1) = CSR_rows(i) + temp(i,2); 
end

CSR_rows_base_0 = (CSR_rows-1);
fprintf("CSR rows pointer has been completed\n");

% for i=1:length(CSR_rows_base_0)
%     fprintf("%d, ", CSR_rows_base_0(i));
% end
% fprintf("\n");
% 
% for i=1:length(COO_base_0(:,2))
%     fprintf("%d, ", COO_base_0(i,2));
% end
% fprintf("\n");
clearvars i temp 
%% Write CSR to file
fprintf("Printing CSR_ROWS.txt\n");
fileID = fopen("./CSR_ROWS.txt", 'w');
for i = 1:length(CSR_rows_base_0)
   fprintf( fileID, "%d\n", CSR_rows_base_0(i));
end
fclose(fileID);

fprintf("CSR_ROWS.txt printed\n")
fprintf("Printing CSR_COLS.txt\n");
fileID = fopen("./CSR_COLS.txt", 'w');
for i = 1:length(COO_base_0(:,2))
   fprintf( fileID, "%d\n", COO_base_0(i,2));
end
fclose(fileID);
fprintf("CSR_COLS.txt printed\n")
fprintf("CSR format printed to files\n");

