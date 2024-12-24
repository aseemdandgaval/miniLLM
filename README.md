# miniLLM

This project implements **GPT2-124M** from scratch and trains it  using distributed training principles on a cloud cluster of **8× 80GB NVIDIA A100** GPUs. The entire training process was completed in **~3.5 hours** on **10 Billion tokens**.

1. **Model**: [GPT-2 (124M)](https://github.com/openai/gpt-2)
2. **Dataset**: [FineWebEdu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 
3. **Hardware**: 8× 80GB NVIDIA A100 GPUs  
4. **Training Time**: ~3.5 hours
5. **Downloading and Sharding the Dataset:** ~1.25 hr
6. **Distributed Strategy**: Data parallelism

Cloud usage

![Results Overview](images/cloud_usage.png)


## Repository Structure

```
.
├── images/
│   ├──results.png               # Training parameters (e.g., batch size, LR, etc.)
├── eval/
│   ├── hellaswag.py             # Functions for running the HellaSwag eval
├── data/
│   └── dataset.py               # DataLoader definition
│   └── fineweb.py               # Download and shard the FineWebEdu-10B dataset 
├── src/
│   ├── gpt_train.py             # Script to train the model 
│   ├── model.py                 # GPT-2 model definition
│   ├── utils.py                 # Utility functions used in training
└── README.md                    # Project documentation 

```


## Results

After training for 10B tokens, this model achieved lower validation loss and higher accuracy on the HellaSwag eval than the openai GPT2-124 checkpoint.

![Results Overview](images/results.png)


Before any training:

```
This is a summary about wont sublimeettlerider Hera Citiz Links operationhittingBrainLeague enginesitalscies solve neglect missions soar abortionouls supernaturalfitolding Recogngithub
This is a summary aboutbinary Emb PI Cindy secure shades referencing IPOrenowers McF medicine interrupts hurry billing successfully submission academiceful locovy molecule416specplementation
This is a summary about Presence matanded Languages roomm rapport lb ranchitt Granderans policy Twice Gift then Questions fendFordhtarloe mixed Optional prud tsp especially
This is a summary aboutWo rumorsOption mineral mindlesstaker harrowing accidents>< ceremonamped nm erad Slater Panama431Dave mineruggish bandits RangersRus Cole silence0001
This is a summary about consent simBytesembedreportprint Bundesligaarantine sidelined776nceRay cues Ho duo seizing delve intest hill juggling finalists seeker meteorOct+)asonablemajor
```

And after 10 Billion tokens of training:

```
> This is a summary about the different components of a health-care continuum. Diseases occur more often with some chronic illnesses. In this
> This is a summary about how our species is evolving. We already know the different traits of different breeds that we are working on using the resources we have
> This is a summary about how to properly format my own thoughts. For example, my first paragraph is about my thoughts on the topic. There are
> This is a summary about the topic. The term 'fraud' is used to describe any scheme by which someone with a computer file has
> This is a summary about the use of and the impacts of oil and gas: As a long-term investment, oil and gas infrastructure is already
```

## Acknowledgments
This project was made following the ![Lets reproduce GPT-124M]("https://www.youtube.com/watch?v=l8pRSuU81PU&t=11363s) tutorial on youtube and the ![build-nanoogpt](https://github.com/karpathy/build-nanogpt) repo.

## License
This project is licensed under the MIT License.
    

