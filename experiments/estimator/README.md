Check if the predicted $n^*$ actually yields reliable results.

Parameters: fix an interpretable (check paper) kernel, the desired reliability $\alpha \in [0, 1]$, and the minimum 
    average similarity $\delta \in [0, 1]$.

1. Get distribution $P$ of results (either synthetic or from resampling from real experimental results)
2. Get its reliability for a range of $n$-s
2. Get empirical sample $P^N$, estimate $n^*_N$
3. Check if the actual reliability evaluated for $n^*$, $\hat \alpha_N$ is above the desired reliability $\alpha$



