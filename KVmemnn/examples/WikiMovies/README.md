# WikiMovies dataset with Key-Value Memory Networks

This example contains code for running [Key-Value Memory Networks](https://arxiv.org/abs/1606.03126) on the [WikiMovies](http://fb.ai/babi) dataset.

## Instructions

We include the code to preprocess the base text data into the final forms that we use to train the model, or you can directly download the processed versions.

To download the processed versions, run ./setup\_processed\_data.sh (46M download, unpacks to 277M).

To generate the data yourself, take the following steps (in order):

    ./setup_data.sh (downloads WikiMovies dataset to folder movieQA)
    ./gen_wiki_windows.sh (preprocesses wiki data)
    ./gen_multidict_questions.sh (preprocesses questions in train/dev/test)
    ./build_dict.sh
    ./build_data.sh
    ./build_hash.sh

If you want to recreate experiments on the templated data sources, run `gen_template_windows.sh` instead of `gen_wiki_windows.sh`.
These are not included in the processed data package above, since they're pretty sizable in aggregate and not the primary focus of the paper.
You'll have to edit the build\*.sh scripts above to point to the template you want to use.

Once you have either downloaded the data or generated it, then you can run...

    ./run_interactive.sh (to check that the data is loaded properly)
    ./run_train.sh (to train a model on the data)
    ./run_eval.sh (to evaluate a model on the data)

To customize the hyperparameters of the model, adjust the params.lua file or pass desired parameters in via the command line (e.g. `./run_train.sh numThreads=1 learningRate=0.1`).

Output will go into the directory `output`, which you can test out by running ./run\_eval.sh with modelFilename directed towards the .best\_valid\_model file from those logs.
You can download a pre-trained model by running the ./setup\_pretrained.sh file (334M download, unpacks to 737M).


## References

* Alexander H. Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, and Jason Weston, "[Key-Value Memory Networks for Directly Reading Documents](https://arxiv.org/abs/1606.03126)", *arXiv:1604.06045 [cs.CL]*.
