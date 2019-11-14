# A Unified Non-Negative Matrix Factorization Framework for Semi Supervised Learning on Graphs

Semi-supervised clusterable node representation learning for homogeneous graphs.
Please see our initial work:- Vijayan, Priyesh, et al. ["Semi-supervised learning for clusterable graph embeddings with NMF."] (https://priyeshv.github.io/R2L_SSNMF.pdf)

### How to run
Input: Input is organized as follows -
    Datasets/
        |_ _ _ <Dataset-name>
                    |_ _ _ <Dataset-name.mat>
                    |_ _ _ <Stats.txt>
                    |_ _ _ <Percentage of train-test split>
                                        |_ _ _ <Fold-No>
                                                    |_ _ _ test_ids.npy
                                                    |_ _ _ train_ids.npy
                                                    |_ _ _ val_ids.npy
   
 python main_algo.py --DATA_DIR cora --ALPHA_BIAS -2 --ALPHA 1.0 --LAMBDA 1.0 --L_COMPONENTS 16
        * ALPHA_BIAS : Alpha bias level for biased random walk
        * ALPHA : Similarity matrix factorization weight
        * LAMBDA : L2 regularization weight
        * L_COMPONENTS : Dimension of projected space


## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

