Python implementation of "Learning Privacy-Preserving Graph Convolutional Network with Partially Observed Sensitive Attributes" (WWW-2022).

### Abstract
Recent studies have shown Graph Neural Networks (GNNs) are extremely vulnerable to attribute inference attacks. To tackle this challenge, existing privacy-preserving GNNs research assumes that the sensitive attributes of all users are known beforehand. However, due to different privacy preferences, some users (i.e., private users) may prefer not to reveal sensitive information that others (i.e., non-private users) would not mind disclosing. For example, in social networks, male users are typically less sensitive to their age information than female users. The age disclosure of male users can lead to the age information of female users in the network exposed. This is partly because social media users are connected, the homophily property and message-passing mechanism of GNNs can exacerbate individual privacy leakage. In this work, we study a novel and practical problem of learning privacy-preserving GNNs with partially observed sensitive attributes.
 
In particular, we propose a novel privacy-preserving GCN model coined DP-GCN, which effectively protects private users' sensitive information which has been revealed by non-private users in the same network. DP-GCN consists of two modules: First, Disentangled Representation Learning Module, which disentangles the original non-sensitive attributes into sensitive and non-sensitive latent representations that are \textit{orthogonal} to each other. Second, Node Classification Module, which trains the GCN to classify unlabeled nodes in the graph with non-sensitive latent representations. Experimental results on five benchmark datasets demonstrate the effectiveness of DP-GCN in preserving private users' sensitive information while maintaining high node classification accuracy.

### Requirements

Python 3.8.5 and PyTorch 1.9.0.

### Run the code

Step 1: 

Step 2: 

Step 3: 

Step 4: 

Step 5: 

Step 6: 

### Citation
```
Update later
```

