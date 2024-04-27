# Implementation of a transformer and fine-tuning

## The project is part of the linear algebra course

Team members: Bohdan Opyr, Radomyr Husiev, Iryna Kokhan

Imagine a box with special tools (models) that help computers understand/generate human-like text. These models (usually just pre-built) are based on Transformers - machines that can learn how words fit together in sentences well, figuring out the patterns and meanings in language. They generate text as how our brain creates sentences. Using these “boxes,” developers can create powerful chatbots, translators, narratives, etc. Their ability to capture and model intricate patterns in natural language data makes them invaluable tools in various domains, including digital assistants, content creation, and natural language understanding. 

In transformers, memory is vital because it helps the model understand and generate coherent text. Compared to primary recurrent neural networks, transformers do not rely on recurrent connections to maintain memory over time. By allowing each token to consider all other tokens in the input, transformers capture relationships across the entire sequence simultaneously. This allows the model to efficiently understand context and dependencies, making it great for tasks like translation and text understanding. With memory, transformers do not need to process data sequentially, making them efficient for understanding complex language patterns. 
Therefore, the topic of our project focuses on the implementation of such a language model and different parameter-efficient finetuning techniques for it. These “parameters” will be used to solve the problem that may occur with transformers. Its essence is that models are trained on a large amount of data, which can be quite diverse. 

More details you can find here: [link](https://www.overleaf.com/read/mqrwfrtjxsxz#86a7bd)
