# LLMSeuss
## Motivation
To create an opportunity to learn more about Large Language Models and their underlying principals, we aim to write and train a tranformer language model from scratch using pytorch. As an intermediate step, we first write and train a bigram language model, before implementing self-attention heads, layer normalization, feed-forward layers, residual connections, and other techniquess used in transformer language models.

## Methodology
For our training texts, we used the two texts written by Dr. Suess: Green Eggs and Ham and The Cat in The Hat.

These two texts were written with a very limited vocabulary; famously, Dr. Seuss wrote Green Eggs and Ham with less than 50 distinct words. The Cat in the Hat uses around 200 distinct words.

As such, we write a tokenizer that parses on the granularity of single words, with additional tokens for spaces, newline, punctuation, and capitalization markings. In particular, we use one token to represent normal word capitalization (with the first letter capitalized), and one token to represent full word capitalization (where the entire word is capitalized for emphasis)

For the writing of the bigram and transformer language models, we use various papers (such as the famous "Attention is All You Need") to understand the high level framework for tranformer language models, pytorch documentation to understand the format and functionalities of the python code, and online tutorials - such as the ones created by Andrej Karpathy - to gain insight into details on how the high-level framework is implemented via pytorch.

We test the trained models by generating sample text, and examining its fidelity as comprehensible english. We also generate loss plots to visualize the training and validation loss.

## Results

### Bigram Language Model Results
Not unexpectedly, the bigram language model - which only predicts tokens based on the previous tokens - performs incredibly poorly; the generated text is garbled, and does not follow proper syntax structure, such as having spaces after each word.

For instance, the first generated line from the bigram language model in the file "Bigram_model_generations/bigram_generation_2025-01-18_12_29_42.txt" is:

*Who your was tall,trythere box we has strings, you fan, the the find fan, and shut onsaw the the this put cake,*

### Transformer Language Model
The transformer Language Model performs better, although unfortunately, the training text is far too little, causing the model to be overfit. For example, in "Transformer_model_generations/transformer_generation_2025-01-18_12_38_00.txt" the model exhibits a perfect recall of the last few stanzas of The Cat in The Hat. However, after it reaches the end of the text, and continues generating, it devolves into more garbled English. Nonetheless, the model respects a couple tenets of proper english syntax, such as capitalizing the first letter of a line, spaces between words, punctuation at the end of lines, etc. Although the model does fail to properly make use of quotation marks, with unpaired quotation marks being seen in the generation. Likely the use of quotation marks is too intermitent within the training data to properly train the model.

However, amidst the garbled mess, there is nonetheless some semblences of meaning, for example one more coherent subsection of "Transformer_model_generations/transformer_generation_2025-01-18_12_38_00.txt" we have:

*What THEN!*
*Well...*
*There is no way at all!*

## Conclusions
While this project served as a powerful learning tool for understanding the principles underlying Transformer-based Large Language Models, the limited training data in combination with the chosen method of tokenizing the textby words contributed to a less-than optimal trained model. It would be of great personal interest to consider future projects where the model written in this repository may be applied to larger-scale training data, with a more carefully chosen tokenizing method.