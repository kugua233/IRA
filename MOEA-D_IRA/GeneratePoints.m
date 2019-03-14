function [ref,Wm] = GeneratePoints(Global)
    
    Population = Global.Initialization();
    w=ones(1,Global.M); % 中心向量 （1,1,1,1,..）
    [FrontNo,~] = NDSort(Population.objs,Population.cons,1);    
        % FrontNo返回每个解对应的level,MaxFNo返回总的level数
        % 注意：非支配解是没有level对应的，它们在FrontNo中对应Inf(无穷大)表示
        
        NDpopulation=Population(find(FrontNo==1));  % 找出第一层级的解
        Zmin=min(Population.objs,[],1);     % 找出每个目标函数的最小值
        Znad=max(NDpopulation.objs,[],1);   % 找出每个目标函数的最大值
        
       disp('------------------------第一阶段：确定方向----------------------------------');
        while (Global.NotTermination(Population) &&Global.evaluated < Global.evaluation/4)
        % 在前1/4进化代数中进行边界点查找
             [Zmin,Znad] = Normalization_EFRRR(Population.objs,Zmin,Znad);  % 找出各目标函数的极值
           
            % Classification
            Subpopulation= Classifica(Population,w,Global.M,Global.N); 
            for i = 1 : Global.M    % flag=1##在子区域内各自进化
                temp=length(Subpopulation{i});
                MatingPool = randi(temp,1,temp);    % 交配池
                Offspring  = Global.Variation(Subpopulation{i}(MatingPool),temp,@EAreal,{10,15,1,30});  % 交叉变异
                Subpopulation{i} = [Subpopulation{i},Offspring];    %元胞？##合并该区域内的父代和子代
                Fitness=Cal_fitness(Subpopulation{i},w,Zmin,Znad,i,Global.M,Global.evaluated,Global.evaluation);  %要注重找到的Zmin确实为（0,0,0..）
                % 计算区域内个体的适应度值
                
                [~,rank] = sort(Fitness,'descend');     % 降序排列
                Subpopulation{i} = Subpopulation{i}(rank(1:temp));  % 只挑选排在前头的个体进入到下一代
            end
            Population = [Subpopulation{:}];
        end
        
        
        disp('------------------------将方向归为向量----------------------------------');  % disp(X) 显示变量 X 的值，而不打印变量名称
        % 在前1/4代数中已经找完了边界区域解，现在在每个区域内各找出一个最好的解
        
        Wm=ones(Global.M,Global.M);
        for i=1:Global.M
            [FrontNo,~] = NDSort(Subpopulation{i}.objs,1);    % 区域内非支配排序分层，FrontNo返回各解对应的level
            NP_loc=find(FrontNo==1);    % 得到在Subpopulation{i}的序列号 ##找出非支配解
            NP_POP=Subpopulation{i}(NP_loc);    % 获取区域内的非支配个体
            NP_pop=NP_POP.objs;                 % 获取区域内的非支配个体的目标值
            [~,rank] = sort(NP_pop(:,i),'descend'); % 在对应维度上降序排列
            disp(NP_pop(rank(1),:))             % 只要对应维度上排名第一的解
            %Wm(i,:)=NP_pop(rank(1),:)./sum(NP_pop(rank(1),:));
            Wm(i,:)=NP_pop(rank(1),:);          % 将排头解设置为该维度上对应的边界点
        end
        
        %%
        disp('------------------------第二阶段：根据边界点自适应生成参考向量---------------------------------');
        centrepoint=sum(Wm,1)./Global.M;    % 中心点计算：各个维度分量求和算平均
        %klayer=6;  % 需要进行设置 ##klayer是要分割层数，参考SPEAR
        M = Global.M
        switch M
            case 3
                klayer = 8
            case 5
                klayer = 6
            case 8
                klayer = 6
            case 10
                klayer = 5
            otherwise
                klayer = 4
        end
        
        ref=[];     % ref存参考向量
        % create points on the sides of subsimplexes 
        for j=1:klayer    % 生成中心点与边界点连线上的点，一共有k*M个点
            ref=[ref;repmat(centrepoint,Global.M,1)+(Wm-repmat(centrepoint,Global.M,1)).*(j/klayer)]; 
        end
        disp(ref)
        
        for i=1:Global.M
            point_a = Wm(mod(i-1,Global.M)+1,:);    % mod(a,m) 返回用 m 除以 a 后的余数
            point_b = Wm(mod(i,Global.M)+1,:);
            for j=1:klayer    % 一共有k层，现在计算除端点外的内部点
                for t=1:j     % 每层除端点外，还有t个点
                    a1 = centrepoint + (point_a - centrepoint).*(j/klayer);
                    b1 = centrepoint + (point_b - centrepoint).*(j/klayer);
                    ref=[ref; a1+(b1-a1).*t./(j+1)];
                end
            end
        end
        ref=[ref;centrepoint];     % 参考点个数ref=M*k(k+3)/2+1，种群个数N=ref
        ref=ref./repmat(sum(ref,2),1,Global.M);     % 归一化 ##为什么要归一化？？
        % sum(A,2) 是包含每一行总和的列向量。
        
    end
    
%% 计算子区域内个体的适应度值
function Fitness=Cal_fitness(SubPopulation,w,Zmin,Znad,I,M,evaluated,evaluation)
    Theta=evaluated/(evaluation);
    
    wt=ones(M,M);
    for j=1:M-1
        wt(j,j+1)=-1;
    end
    wt(M,1)=-1;
    
    wtsingle=wt(I,:);   % I是传进来的参数，表示第I维度
    % wtsingle是第I维度对应要使用的相邻中心向量(即变换矩阵)
    
    Popobj=(SubPopulation.objs-repmat(Zmin,length(SubPopulation),1))./repmat(Znad-Zmin,length(SubPopulation),1);    % 归一化
    %Popobj1=(SubPopulation.objs-repmat(Zmin,length(SubPopulation),1))./repmat(Znad-Zmin,length(SubPopulation),1)+repmat(Zmin,length(SubPopulation),1);%Angle的判定应该用Zmin=(0,0,0...)
    
    Angle = acos(1-pdist2(Popobj,w,'cosine'));             % 目标向量与中心向量的角度，越大越好,数值在1.24左右，相差也不大
    wtAngle = acos(1-pdist2(Popobj,wtsingle,'cosine'));    % 目标向量与变换矩阵的角度，N*M的矩阵，越小越好；
    
    fitness=Angle.^2.*Popobj(:,I);                         % 找到边界，越大越好
    distance=pdist2(Popobj,Zmin,'chebychev');              % 目标向量与理想点的切比雪夫距离，越小越好,数值在0-1左右，基本是0.9-1.0
    %Fitness=(Angle.^(1)).*(Popobj(:,I)*Theta)./(distance.^(0))./(wtAngle.^(1)); %越大越好  log2(Theta)/log2(0.05)
    
    Fitness=(Angle.^(2+0.5*M)).*(Popobj(:,I).*(0.5))./(distance.^(0))./(wtAngle.^(1+0.5*M));    % 个体的适应度值
end

%% 将种群个体划归为M类，即M个边界点对应的子区域
function  Subpopulation= Classifica(Population,w,M,N)   % 算子可以参考Cal_fitness
    wt=ones(M,M);       
    for j=1:M-1
        wt(j,j+1)=-1;
    end
    wt(M,1)=-1;         % w是中心向量，wt是与w相邻的M个中心向量组成的向量组
    
    Zmin=min(Population.objs,[],1);
    Znad=max(Population.objs,[],1);
    Popobj=(Population.objs-repmat(Zmin,length(Population),1))./repmat(Znad-Zmin,length(Population),1);%归一化
    
    Angle = acos(1-pdist2(Popobj,w,'cosine')).*180./pi;     % 计算目标向量与中心向量的角度，N行1列   弧度值，应该*180/派
    wtAngle = acos(1-pdist2(Popobj,wt,'cosine')).*180./pi;  % 计算目标向量与变换矩阵的角度，N*M的矩阵，越小越好；
    TempAngle=repmat(Angle,1,M);
    
    Fitness=Popobj.*TempAngle.^2./wtAngle;  % Fitness是N行M列，每行表示一个解，每列表示对应目标函数值靠近边界点的程度得分
                                            % 目标函数值越大，与中心向量越远离，与对应相邻中心向量越靠近的分量，在此维度上的得分越高
    
    class =zeros(N,1);  % 标号
    Subpopulation = cell(1,M);
    Selected=[];
    for i=1:M-1
        [~,rank]=sort(Fitness(:,i),'descend');  % sort(,'descend')在第i列上降序排列，rank返回索引
        fixedRow=rank(1:floor(N/M));            
        % floor(X) 将 X 的每个元素四舍五入到小于或等于该元素的最接近整数
        % fixedRow是在第i维度上分量得分排在前1/M部分的解
        
        Selected=[Selected,fixedRow];   % Selected是(N/M)*M数组，每列表示各个子区域的解索引
        class(fixedRow)=i;              % 将解标号，划入对应区域
        Fitness(Selected,i+1)=-inf;     % 将被选中解的下一列设为无穷小，避免又被其他区域选中
    end
    class(class==0)=M;  % 将剩下的个体划入到第M维度对应区域，要单拿出来是因为N/M可能不是整数，那么余下就给最后一个维度
    for i = 1 :M
        Subpopulation{i} = Population(class==i);    % 根据个体标号对种群进行划分
    end
end

%%
function [TP2,TP2Fitness] = DefineTP2(Population,M,Wm,Zmin)
    TP2 = [];
    TP2Fitness=[];
    [FrontNo,~] = NDSort(Population.objs,1);
    NPLocation=find(FrontNo==1);%得到第一次非支配解的序号
    NPopulation=Population(NPLocation);%得到非支配解
    for i = 1:M
        ASF = max((NPopulation.objs-repmat(Zmin,length(NPopulation),1))./repmat(Wm(i,:),length(NPopulation),1),[],2);
        %[value,extreme] = max(ASF);
        [value,extreme] = min(ASF);
        TP2 = [TP2,NPopulation(extreme)];
        TP2Fitness=[TP2Fitness,value];
    end
end