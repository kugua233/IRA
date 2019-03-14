function [ref,Wm] = GeneratePoints(Global)
    
    Population = Global.Initialization();
    w=ones(1,Global.M); % �������� ��1,1,1,1,..��
    [FrontNo,~] = NDSort(Population.objs,Population.cons,1);    
        % FrontNo����ÿ�����Ӧ��level,MaxFNo�����ܵ�level��
        % ע�⣺��֧�����û��level��Ӧ�ģ�������FrontNo�ж�ӦInf(�����)��ʾ
        
        NDpopulation=Population(find(FrontNo==1));  % �ҳ���һ�㼶�Ľ�
        Zmin=min(Population.objs,[],1);     % �ҳ�ÿ��Ŀ�꺯������Сֵ
        Znad=max(NDpopulation.objs,[],1);   % �ҳ�ÿ��Ŀ�꺯�������ֵ
        
       disp('------------------------��һ�׶Σ�ȷ������----------------------------------');
        while (Global.NotTermination(Population) &&Global.evaluated < Global.evaluation/4)
        % ��ǰ1/4���������н��б߽�����
             [Zmin,Znad] = Normalization_EFRRR(Population.objs,Zmin,Znad);  % �ҳ���Ŀ�꺯���ļ�ֵ
           
            % Classification
            Subpopulation= Classifica(Population,w,Global.M,Global.N); 
            for i = 1 : Global.M    % flag=1##���������ڸ��Խ���
                temp=length(Subpopulation{i});
                MatingPool = randi(temp,1,temp);    % �����
                Offspring  = Global.Variation(Subpopulation{i}(MatingPool),temp,@EAreal,{10,15,1,30});  % �������
                Subpopulation{i} = [Subpopulation{i},Offspring];    %Ԫ����##�ϲ��������ڵĸ������Ӵ�
                Fitness=Cal_fitness(Subpopulation{i},w,Zmin,Znad,i,Global.M,Global.evaluated,Global.evaluation);  %Ҫע���ҵ���ZminȷʵΪ��0,0,0..��
                % ���������ڸ������Ӧ��ֵ
                
                [~,rank] = sort(Fitness,'descend');     % ��������
                Subpopulation{i} = Subpopulation{i}(rank(1:temp));  % ֻ��ѡ����ǰͷ�ĸ�����뵽��һ��
            end
            Population = [Subpopulation{:}];
        end
        
        
        disp('------------------------�������Ϊ����----------------------------------');  % disp(X) ��ʾ���� X ��ֵ��������ӡ��������
        % ��ǰ1/4�������Ѿ������˱߽�����⣬������ÿ�������ڸ��ҳ�һ����õĽ�
        
        Wm=ones(Global.M,Global.M);
        for i=1:Global.M
            [FrontNo,~] = NDSort(Subpopulation{i}.objs,1);    % �����ڷ�֧������ֲ㣬FrontNo���ظ����Ӧ��level
            NP_loc=find(FrontNo==1);    % �õ���Subpopulation{i}�����к� ##�ҳ���֧���
            NP_POP=Subpopulation{i}(NP_loc);    % ��ȡ�����ڵķ�֧�����
            NP_pop=NP_POP.objs;                 % ��ȡ�����ڵķ�֧������Ŀ��ֵ
            [~,rank] = sort(NP_pop(:,i),'descend'); % �ڶ�Ӧά���Ͻ�������
            disp(NP_pop(rank(1),:))             % ֻҪ��Ӧά����������һ�Ľ�
            %Wm(i,:)=NP_pop(rank(1),:)./sum(NP_pop(rank(1),:));
            Wm(i,:)=NP_pop(rank(1),:);          % ����ͷ������Ϊ��ά���϶�Ӧ�ı߽��
        end
        
        %%
        disp('------------------------�ڶ��׶Σ����ݱ߽������Ӧ���ɲο�����---------------------------------');
        centrepoint=sum(Wm,1)./Global.M;    % ���ĵ���㣺����ά�ȷ��������ƽ��
        %klayer=6;  % ��Ҫ�������� ##klayer��Ҫ�ָ�������ο�SPEAR
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
        
        ref=[];     % ref��ο�����
        % create points on the sides of subsimplexes 
        for j=1:klayer    % �������ĵ���߽�������ϵĵ㣬һ����k*M����
            ref=[ref;repmat(centrepoint,Global.M,1)+(Wm-repmat(centrepoint,Global.M,1)).*(j/klayer)]; 
        end
        disp(ref)
        
        for i=1:Global.M
            point_a = Wm(mod(i-1,Global.M)+1,:);    % mod(a,m) ������ m ���� a �������
            point_b = Wm(mod(i,Global.M)+1,:);
            for j=1:klayer    % һ����k�㣬���ڼ�����˵�����ڲ���
                for t=1:j     % ÿ����˵��⣬����t����
                    a1 = centrepoint + (point_a - centrepoint).*(j/klayer);
                    b1 = centrepoint + (point_b - centrepoint).*(j/klayer);
                    ref=[ref; a1+(b1-a1).*t./(j+1)];
                end
            end
        end
        ref=[ref;centrepoint];     % �ο������ref=M*k(k+3)/2+1����Ⱥ����N=ref
        ref=ref./repmat(sum(ref,2),1,Global.M);     % ��һ�� ##ΪʲôҪ��һ������
        % sum(A,2) �ǰ���ÿһ���ܺ͵���������
        
    end
    
%% �����������ڸ������Ӧ��ֵ
function Fitness=Cal_fitness(SubPopulation,w,Zmin,Znad,I,M,evaluated,evaluation)
    Theta=evaluated/(evaluation);
    
    wt=ones(M,M);
    for j=1:M-1
        wt(j,j+1)=-1;
    end
    wt(M,1)=-1;
    
    wtsingle=wt(I,:);   % I�Ǵ������Ĳ�������ʾ��Iά��
    % wtsingle�ǵ�Iά�ȶ�ӦҪʹ�õ�������������(���任����)
    
    Popobj=(SubPopulation.objs-repmat(Zmin,length(SubPopulation),1))./repmat(Znad-Zmin,length(SubPopulation),1);    % ��һ��
    %Popobj1=(SubPopulation.objs-repmat(Zmin,length(SubPopulation),1))./repmat(Znad-Zmin,length(SubPopulation),1)+repmat(Zmin,length(SubPopulation),1);%Angle���ж�Ӧ����Zmin=(0,0,0...)
    
    Angle = acos(1-pdist2(Popobj,w,'cosine'));             % Ŀ�����������������ĽǶȣ�Խ��Խ��,��ֵ��1.24���ң����Ҳ����
    wtAngle = acos(1-pdist2(Popobj,wtsingle,'cosine'));    % Ŀ��������任����ĽǶȣ�N*M�ľ���ԽСԽ�ã�
    
    fitness=Angle.^2.*Popobj(:,I);                         % �ҵ��߽磬Խ��Խ��
    distance=pdist2(Popobj,Zmin,'chebychev');              % Ŀ���������������б�ѩ����룬ԽСԽ��,��ֵ��0-1���ң�������0.9-1.0
    %Fitness=(Angle.^(1)).*(Popobj(:,I)*Theta)./(distance.^(0))./(wtAngle.^(1)); %Խ��Խ��  log2(Theta)/log2(0.05)
    
    Fitness=(Angle.^(2+0.5*M)).*(Popobj(:,I).*(0.5))./(distance.^(0))./(wtAngle.^(1+0.5*M));    % �������Ӧ��ֵ
end

%% ����Ⱥ���廮��ΪM�࣬��M���߽���Ӧ��������
function  Subpopulation= Classifica(Population,w,M,N)   % ���ӿ��Բο�Cal_fitness
    wt=ones(M,M);       
    for j=1:M-1
        wt(j,j+1)=-1;
    end
    wt(M,1)=-1;         % w������������wt����w���ڵ�M������������ɵ�������
    
    Zmin=min(Population.objs,[],1);
    Znad=max(Population.objs,[],1);
    Popobj=(Population.objs-repmat(Zmin,length(Population),1))./repmat(Znad-Zmin,length(Population),1);%��һ��
    
    Angle = acos(1-pdist2(Popobj,w,'cosine')).*180./pi;     % ����Ŀ�����������������ĽǶȣ�N��1��   ����ֵ��Ӧ��*180/��
    wtAngle = acos(1-pdist2(Popobj,wt,'cosine')).*180./pi;  % ����Ŀ��������任����ĽǶȣ�N*M�ľ���ԽСԽ�ã�
    TempAngle=repmat(Angle,1,M);
    
    Fitness=Popobj.*TempAngle.^2./wtAngle;  % Fitness��N��M�У�ÿ�б�ʾһ���⣬ÿ�б�ʾ��ӦĿ�꺯��ֵ�����߽��ĳ̶ȵ÷�
                                            % Ŀ�꺯��ֵԽ������������ԽԶ�룬���Ӧ������������Խ�����ķ������ڴ�ά���ϵĵ÷�Խ��
    
    class =zeros(N,1);  % ���
    Subpopulation = cell(1,M);
    Selected=[];
    for i=1:M-1
        [~,rank]=sort(Fitness(:,i),'descend');  % sort(,'descend')�ڵ�i���Ͻ������У�rank��������
        fixedRow=rank(1:floor(N/M));            
        % floor(X) �� X ��ÿ��Ԫ���������뵽С�ڻ���ڸ�Ԫ�ص���ӽ�����
        % fixedRow���ڵ�iά���Ϸ����÷�����ǰ1/M���ֵĽ�
        
        Selected=[Selected,fixedRow];   % Selected��(N/M)*M���飬ÿ�б�ʾ����������Ľ�����
        class(fixedRow)=i;              % �����ţ������Ӧ����
        Fitness(Selected,i+1)=-inf;     % ����ѡ�н����һ����Ϊ����С�������ֱ���������ѡ��
    end
    class(class==0)=M;  % ��ʣ�µĸ��廮�뵽��Mά�ȶ�Ӧ����Ҫ���ó�������ΪN/M���ܲ�����������ô���¾͸����һ��ά��
    for i = 1 :M
        Subpopulation{i} = Population(class==i);    % ���ݸ����Ŷ���Ⱥ���л���
    end
end

%%
function [TP2,TP2Fitness] = DefineTP2(Population,M,Wm,Zmin)
    TP2 = [];
    TP2Fitness=[];
    [FrontNo,~] = NDSort(Population.objs,1);
    NPLocation=find(FrontNo==1);%�õ���һ�η�֧�������
    NPopulation=Population(NPLocation);%�õ���֧���
    for i = 1:M
        ASF = max((NPopulation.objs-repmat(Zmin,length(NPopulation),1))./repmat(Wm(i,:),length(NPopulation),1),[],2);
        %[value,extreme] = max(ASF);
        [value,extreme] = min(ASF);
        TP2 = [TP2,NPopulation(extreme)];
        TP2Fitness=[TP2Fitness,value];
    end
end