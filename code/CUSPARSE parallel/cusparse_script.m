clear 
clc

cd ../../data/auto/

load auto.mat

A = Problem.A > 0;

fprintf( '...triangle counting...\n' ); 
ticCnt = tic;

nT = full( sum( sum( A^2 .* A ) ) / 6 );

fprintf( '   - DONE: %d triangles found in %.2f sec\n', nT, toc(ticCnt) );


%% Make A suitable format
A = double(A);
A = triu(A);

%% Result
%tic
C = A*A;
%toc


%% Write to file
%% Write Sparse Matrix A to COO format
SA = sparse(A);

[ rowA, colA, vA] = find(SA);
nnzA = length(rowA);
% Cusparse COO format is row-major.
TableA = [rowA, colA, vA];
TableA = sortrows(TableA, 1);

fid = fopen("MatrixACOO-R-C-V.txt", "w");

for i = 1:length(rowA)
    fprintf( fid, "%.1d %.1d %.1d\n", TableA(i,1) -1, TableA(i,2)-1,  TableA(i,3));
end

fclose(fid);
fprintf("MatrixACOO-R-C-V.txt DONE\n");


%% Write Sparse Matrix C to COO format
SC = sparse(C);

[ rowC, colC, vC] = find(SC);
nnzC = length(rowC);
% Cusparse COO format is row-major.
TableC = [rowC, colC, vC];
TableC = sortrows(TableC, 1);

fid = fopen("MatrixCCOO-R-C-V.txt", "w");

for i = 1:length(rowC)
    fprintf( fid, "%.1d %.1d %.1d\n", TableC(i,1) -1, TableC(i,2)-1,  TableC(i,3));
end

fclose(fid);

%% Sequential mask test
% triangles = 0;
% i = 1;
% k = 1;
% 
% 
% while i <= nnzA && k <= nnzC
%     if (TableA(i,1) == TableC(k,1) &&  TableA(i,2) == TableC(k,2))
%         triangles = triangles + TableC(k,3);
%         i = i + 1;
%         k = k + 1;
%     else
%         if( TableA(i,1) < TableC(k,1))
%             i = i + 1;
%         elseif TableA(i,1) > TableC(k,1)
%             k = k + 1;
%         elseif TableA(i,2) < TableC(k,2)
%             i = i + 1;
%         elseif TableA(i,2) > TableC(k,2)
%             k = k + 1;
%         end
%     end
% end






