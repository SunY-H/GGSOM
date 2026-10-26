
% function [Xiajinsi,int_relation,fuzzy_neighbor_group] = MyNeighbor(X,Y)
% 
% data = [X,Y];
% [row,~]=size(X);
% 
% %% 生成粒球
% [~,index] = my_GBC(data);
% 
% %% 模糊邻域相似关系
% [int_relation,fuzzy_neighbor_group]= my_FN_Relation(data,index);
% 
% %%  模糊决策
% [partion,~,cj] = unique(Y);
% class_num = length(partion);
% 
% % 预先计算每个类的成员索引 使用arrayfun来替代原始的循环创建class_indices，这可以更高效地处理类成员的索引。
% class = arrayfun(@(x) find(cj == x), 1:class_num, 'UniformOutput', false);
% 
% % class = cell(1,class_num);
% % for i = 1:class_num
% %     class{i} = find(cj==i);
% % end
% 
% FD = zeros(row,class_num);
% for i = 1: row
%     sum_int_relation_i = sum(int_relation(i, :));  % 分母预计算
%     for j = 1:class_num
%         FD(i,j) = (sum(int_relation(i,class{j})))/sum_int_relation_i;
%     end
% end
% 
% %% 下近似
% % -------------------------------------将每个决策类对应的下近似分开存储/有效标记-----------------------------------------------
% 
% % Xiajinsi = zeros(class_num, row);
% % 
% % for i = 1: row
% %     BD = zeros(class_num,row);
% %     for k= 1:class_num
% %         sign = ismember(i,class{k});
% %         for j=1: row
% %             if sign==1
% %                 BD(k,j)=max(1-int_relation(i,j),FD(j,k));
% %             else
% %                 BD(k,j)=0;
% %             end
% %         end
% %         Xiajinsi(k,i) = min(BD(k,:));
% %     end
% % end
% % 
% 
% % 以上下近似和以下下近似的区别在于：上面的下近似只保留了最大的隶属度，其余的隶属度为0；而下面的下近似中没有将其余的隶属度设置为0.
% 
% 
% % 初始化下近似矩阵，大小为 class_num x row
% Xiajinsi = zeros(class_num, row);
% Shangjinsi = zeros(class_num, row);
% 
% % 对于每个类别
% for k=1:class_num
%     % 临时存储该类别的所有样本的下近似值
%     BD_temp = zeros(1, row);
%     UA_temp = zeros(1, row);
%     
%     % 对于每个样本i
%     for i=1:row
%         % 对于当前类别k，计算样本i的下近似值
%         % 下近似值是基于模糊相似度和模糊决策值的最小max-min配对
%         BD = max(1 - int_relation(i,:), FD(:,k)');
%         BD_temp(i) = min(BD);
% 
%         UA = min(1 - int_relation(i,:), FD(:,k)');
%         UA_temp(i) = max(UA);
%     end
%     
%     % 更新该类别的所有样本的下近似值
%     Xiajinsi(k, :) = BD_temp;
% 
%     Shangjinsi(k, :) = UA_temp;
% end
% 
% %Positive = sum(Xiajinsi);
% 
% %AvgPos = mean(Positive);
% 
% end

function [Xiajinsi,int_relation,fuzzy_neighbor_group,ballCenters] = MyNeighbor(X, Y)
    data = [X, Y];
    [row, ~] = size(X);

    %% 生成粒球
    [ballCenters, index] = my_GBC(data);

    %% 模糊邻域相似关系
    [int_relation, fuzzy_neighbor_group] = my_FN_Relation(data, index);

    %% 模糊决策
    [~, ~, cj] = unique(Y);
    class_num = max(cj);

    % 预先计算每个类的成员索引
    class = arrayfun(@(x) cj == x, 1:class_num, 'UniformOutput', false);

    FD = zeros(row, class_num);
    for j = 1:class_num
        FD(:, j) = sum(int_relation(:, class{j}), 2) ./ sum(int_relation, 2);
    end

    %% 下近似
    Xiajinsi = zeros(class_num, row);

    % 对于每个类别
    for k = 1:class_num
        BD = max(1 - int_relation, FD(:, k));  % 自动进行广播比较
        Xiajinsi(k, :) = min(BD, [], 2); % 沿第二维找到最小值
    end

%     % 上近似（如果需要计算）
%     Shangjinsi = zeros(class_num, row);
%     for k = 1:class_num
%         UA = min(1 - int_relation, FD(:, k)', [], 2);
%         Shangjinsi(k, :) = max(UA, [], 1);
%     end
end


