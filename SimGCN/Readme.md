
./run_random_graph experiments: 

    Implementation of SimGNN on random graph. Since GIN is designed for graph classification rather than node 
    classification, we did not consider it as the baseline before. In this code, we additionally add GIN as 
    another baseline for the reviewer's request. The experimental results can be saw in the following lines. 
    Also, you can run it and obtain the results.
	
	
	`python random_graph_www.py`
    
	Dataset acc(GIN)	vs acc(SimGCN)
    RG1:    0.2714		vs 74.57
    RG2:    0.2743		vs 86.57
    RG3:    0.2686		vs 81.43   
 
./run_gemo_experiments: 

    Implementation of SimGNN on Gemo networks. GIN is also added and the experimental results are:
	
	`bash run_experiments.bash`
    Dataset     acc(GIN)  vs acc(SimGCN)
    Chameleon:  35.75 	vs 	49.06
    Cornell:    45.95 	vs 	64.32
    Squirrel:   21.23 	vs 	34.09
    Texas:      45.95 	vs 	75.41
    Wisconsin:  39.22 	vs 	78.04
    Actor:      16.38 	vs 	35.78
	
To obtain different results, you could comment or un-comment some lines in the code. In addition, we provide the DAL loss implementation file
in ./Implementation_of_DAL.

All the codes will be publicly avaliable.
