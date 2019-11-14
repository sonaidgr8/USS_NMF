# A Unified Non-Negative Matrix Factorization Framework for Semi Supervised Learning on Graphs

Semi-supervised clusterable node representation learning for homogeneous graphs.

> Please see our initial work:~ Vijayan, Priyesh, et al. [Semi-supervised learning for clusterable graph embeddings with NMF](https://priyeshv.github.io/R2L_SSNMF.pdf), Poster:~ [NeuIPS 2018 Relational learning Workshop](https://drive.google.com/file/d/1jqoo3cKJ-X_nFAeo7FdvJajWxANO8Vrb/view?usp=sharing)

### How to run
 ## Input:- 
    Input is organized as follows -
    Datasets/
        |_ _ _ <Dataset-name>
                    |_ _ _ <Dataset-name.mat>
                    |_ _ _ <Stats.txt>
                    |_ _ _ <Percentage of train-test splits>
                                        |_ _ _ <Fold-Number>
                                                    |_ _ _ train_ids.npy
                                                    |_ _ _ test_ids.npy
                                                    |_ _ _ val_ids.npy
   
 ## Usage:-
    python main_algo.py --DATA_DIR cora --ALPHA 1 --BETA 0.1 --THETA 0.5 --K 20 --L_COMPONENTS 128
        - ALPHA : Similarity matrix factorization weight
        - BETA : Community Indicator matrix factorization weight
        - THETA : Label matrix factorization weight
        - K : Number of clusters
        - L_COMPONENTS : Dimension of representation space
  
 > Please look at the *get_ArgumentParser()* function in **main_algo.py** to specify default values.
 
 ## Output:-
    1. The generated node and label embeddings are saved in emb/ folder as Q & U .npy files.
    2. The node and label embeddings are of dimension (#Nodes x L_COMPONENTS) & (#Labels x L_COMPONENTS).
    3. The Node Classification evaluation results are stored in **Results** folder 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

