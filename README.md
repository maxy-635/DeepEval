# How Do Large Language Models Perform in Deep Learning Code Generation? A Benchmark and Empirical Study

DeepEval is the first deep learning code generation benchmark described in the paper "How Do Large Language Models Perform in Deep Learning Code Generation? A Benchmark and Empirical Study". 
## Benchmark
DeepEval is constructed munally and saved the YAML format. DeepEval consists of 100 DL programming tasks, each associated with a Requirement and a Reference Code. The Requirement provides a detailed task requirement, including dataset and model structure. The Reference Code is
a correct task implementation, serving as a reference for evaluating
the generated code.

Path: benchmark/DeepEval

## Promptings:
We adapt 4 prompting to LLMs, as seen in prompts/deepeval. all prompting strategies use the same example, randomly selected from DeepEval, to ensure a fair comparison.

• Zero-shot Prompting [11] directly provides our task require-
ment to the LLMs without examples.

• One-shot Prompting [11] includes an example with the form
of <example task, example code> pair.

• One-shot Chain-of-Thought Prompting (oneshot-cot) [57]
is a variant of one-shot prompting that generates a chain-of-thought
[57] (CoT) for the example task. The prompting includes one exam-
ple with the form of <example task, CoT, example code> triple.

• Few-shot Prompting [9] includes multiple examples with the
form of <example task, example code> pairs.
## Implementation
First, for closed-source models like GPT-4o, we interact via the
OpenAI API interface 8, while for open-source models such as
Llama-3.1-8B, we deploy their publicly released versions on Hug-
gingFace 9. Second, we use nucleus sampling [22] to decode outputs
from the LLMs, setting the temperature to 0.8 and the top-p value
to 0.95 to maintain controlled randomness across all models. Third,
regarding length of promptings and responses, we set the maxi-
mum token limit to 3500 for OpenAI models and the maximum
new token count to 1500 for HuggingFace-hosted models. All ex-
periments are conducted on a uniform infrastructure comprising
two NVIDIA V100 GPUs. To ensure the reliability of the results,
we repeat each experiment 3 times. Finally, 1200 DL programs are
generated for each LLM under four different promptings on 100 DL
code generation tasks in DeepEval.

<!-- ## Results:
### RQ1: Benchmark Effectiveness
<img src="evaluation/dynamic_checking/presentation/RQ1/RQ1.png" width="500"/>

### RQ2: Code Syntax
<img src="evaluation/static_checking/presentation/RQ3_hist.png" width="500"/>
<img src="evaluation/static_checking/presentation/RQ3_barh.png" width="500"/>

### RQ3: Code Semantics
<img src="evaluation/semantic_checking/presentation/Semantic_Similarity.png"/>

### RQ4: Code Executability
<img src="evaluation/dynamic_checking/presentation/RQ4/barh.png"/>
<img src="evaluation/dynamic_checking/presentation/RQ4/all_sankeys.png"/> -->

## Usage
Ensure you're using the right setup and following the proper directory structure to seamlessly evaluate deep learning code generation with our tool.
### 🛠️ Setup
1. Environment Setup
```
$ conda create -n deepeavl python=3.10
$ conda activate classeval
```
2. Repository Setup
Clone the repository and install necessary dependencies:
```
$ git clone https://github.com/maxy-635/DeepEval
$ conda install environment.yml
```

## Installation

To set up the project, you need to add the project directory to your `PYTHONPATH`. You can do this by running the following command:

```bash
export PYTHONPATH=$PYTHONPATH:/your_local_path/DeepEval
```


## Contributing

If you would like to contribute to DeepEval, please fork the repository and submit a pull request. We welcome all contributions!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or inquiries, please contact the project maintainer at [xiangy_ma@buaa.edu.cn)](xiangy_ma@buaa.edu.cn)).
