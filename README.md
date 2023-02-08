# ytnmt
Basic neural network framework (Transformer-based) from scratch.

***Highlights:*** Lightweight and highly modular organization structure.

Suitable for beginners in **NLP** (Natural Language Processing), **Machine Translation**, **Text generation**...

**Note:** The current version only supports trained in a single GPU.

## Dependency
Recommend a ***conda*** virtual environment.
* pip install torch > 1.10
* pip install tqdm 

## How to use?

### train
~~~
bash scripts/train.sh
~~~

### inference or test
~~~
bash scripts/test.sh
~~~

### Acknowledgements
some code we refer to [joeynmt](https://github.com/joeynmt/joeynmt) codebase.