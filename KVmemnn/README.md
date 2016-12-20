# Key-Value Memory Networks for Directly Reading Documents

This project contains code for the Key-Value MemN2N setup in the following paper: "[Key-Value Memory Networks for Directly Reading Documents](https://arxiv.org/abs/1606.03126)".

## Setup

This code requires [Torch7](http://torch.ch) and its luarocks package tds.
You need to compile the c code in library/c--a script containing a default gcc command is provided as `setup.sh`.

## Examples

This directory contains scripts for running this code on specific datasets.
The initial release will include the [WikiMovies](http://fb.ai/babi) dataset.


## Library

This directory includes the main memory network files, listed below:

    base_model.lua: top-level shared model functions, extended by specific models
    cmd.lua: file for parsing options
    data.lua: file for iterating over data
    dict.lua: file for accessing dictionary
    eval_lib.lua: functions for evaluating dev/test-time evaluation
    hash.lua: provides hashing system for accessing knowledge entries
    interactive_lib.lua: interactive library for stepping through individual examples
    kvmemnn_model.lua: key-value memory network model
    memnn_model.lua: memory network model
    parse.lua: parsing methods for building dictionary and data vectors from text
    PositionalEncoder.lua: implements positional encoding system from "[End-To-End Memory Networks](http://arxiv.org/abs/1503.08895)"
    SumVecarr.lua: implementation of Sum nn module for vector_arrays instead of Tensors
    thread_utils.lua: utilities for torch multithreading
    util.lua: utility functions
    vector_array.lua: auto-resizable vector array implementation
    WeightedLookupTableSkinny.lua: weighted lookup table optimized for fixed dimensions

It also includes a directory named "c" which includes a number of .c files that speed up the performance of the library code.
The c files need to be compiled into libmemnn.so--default gcc parameters are provided in a script setup.sh in the top-level directory.


## Scripts

This directory includes scripts to access the library, listed below:

    build_dict.lua: run this on text first to create a dictionary
    build_data.lua: run this on text second to build vector arrays from a dictionary
    build_hash.lua: run this on text third to create hash access to knowledge entries (like the KB or wikipedia for WikiMovies)
    eval.lua: run this to evaluate a dataset
    interactive.lua: run this to walk through examples in a dataset
    train.lua: run this to start training

For examples of using these scripts, check out the examples directory.

## References

* Alexander H. Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, and Jason Weston, "[Key-Value Memory Networks for Directly Reading Documents](https://arxiv.org/abs/1606.03126)", *arXiv:1604.06045 [cs.CL]*.
