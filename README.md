# NLP_duplicates

##№ Describe
The idea of ​​the project is to test the hypothesis that bert-based models are universal approximators and can be trained on any supplied tokens.
In this work, we add distance values ​​between lines to regular embeddings by direct concatenation. Bert's tokenizer contains numbers as strings, which means they can be successfully recognized by a known character from the dictionary. We assume that this will be enough to learn this trait.

### Dataset
We used a dataset that is small enough to just sit in the repository.


### Train
Since we carried out custom tokenization, we had to make custom dataloaders, as well as padding for tags and batches.
Training was carried out using standard fi tuning of a pre-trained language model with the head of a linear layer screwed into 2 classes.

### Results

As a result, a functionality is illustrated that allows you to remove duplicates from the list of entities, replenishing the library of those entities that the model considered to be the same

## Team
- [Yulia Solomennikova](https://t.me/yul_solomen)
- [Mikhaylov Alexey](https://t.me/sp1derAlex) 
- [Baranov Vitaly](https://t.me/vitalybar)
