function [updatedCenters, newIDX] = incrementalGBC(combinedData, existingCenters)
    % 增量粒球计算
    % 输入:
    % combinedData - 合并后的数据集（现有数据和新数据）
    % existingCenters - 现有的粒球中心
    % 输出:
    % updatedCenters - 更新后的粒球中心
    % newIDX - 全部数据对应的粒球索引

    numClusters = size(existingCenters, 1);
    updatedCenters = existingCenters;
    newClusterCount = numClusters;
    
    % 使用 k-means 算法对合并后的数据进行聚类，初始中心为现有粒球中心
    [combinedIDX, ~] = litekmeans(combinedData(:, 1:end-1), numClusters, 'Start', existingCenters);
    
    % 预分配新的中心列表和索引
    maxNewClusters = size(combinedData, 1); % 最坏情况下每个数据点成为一个新中心
    newCenters = zeros(maxNewClusters, size(combinedData, 2) - 1);
    newIndices = combinedIDX;
    
    currentIdx = 1;
    
    % 更新粒球的中心和半径
    for i = 1:numClusters
        % 找到属于当前粒球的数据点
        clusterData = combinedData(combinedIDX == i, :);
        
        if ~isempty(clusterData)
            % 计算新的中心
            updatedCenter = mean(clusterData(:, 1:end-1), 1);
            
            % 检查纯度，如果不纯则分裂粒球
            labels = unique(clusterData(:, end)); % 获取类别标签
            if length(labels) > 1
                % 使用 k-means 进行再次分裂，确保纯度为1
                [subIDX, ~] = litekmeans(clusterData(:, 1:end-1), length(labels));
                for j = 1:length(labels)
                    subClusterData = clusterData(subIDX == j, :);
                    % 更新分裂后的粒球中心
                    newCenters(currentIdx, :) = mean(subClusterData(:, 1:end-1), 1);
                    newClusterIndices = find(combinedIDX == i);
                    newIndices(newClusterIndices(subIDX == j)) = newClusterCount + 1; % 更新粒球索引
                    newClusterCount = newClusterCount + 1;
                    currentIdx = currentIdx + 1;
                end
            else
                newCenters(currentIdx, :) = updatedCenter;
                currentIdx = currentIdx + 1;
            end
        else
            newCenters(currentIdx, :) = existingCenters(i, :);
            currentIdx = currentIdx + 1;
        end
    end
    
    % 移除未使用的预分配空间
    newCenters = newCenters(1:currentIdx-1, :);
    
    % 如果没有分裂的新中心，保留旧的更新的中心
    if isempty(newCenters)
        newCenters = updatedCenters;
    end
    
    updatedCenters = newCenters;
    newIDX = newIndices;
    
    % 返回更新后的粒球中心和全部数据的粒球索引
    return
end
