---
layout: post
title: "论文阅读-Repetition Improves Language Model Embeddings"
date:   2024-10-02
tags: [论文阅读]
comments: true
author: 一只大废柴
toc: true
---
# 总体
## 文献出处
arXiv:2402.15449
## 研究方向
通过重复问题的方式增强嵌入模型的嵌入质量
## 阅读原因
与re2同属一个技术方法，主要是通过重复输入来让大模型更好的适应上下文的信息
## 内容简述
本文提出了一种叫做 回声嵌入 的方法，通过将**需要进行编码的上下文**进行**重复的输入**来达到提高嵌入质量的目的
文章认为在使用自回归模型计算嵌入的时候，先输出的嵌入可能**由于因果注意力掩码而无法包含句子中后续词汇的信息**。而**通过上下文的重复输入**，在第二次输入单词的时候，模型可以去注意到第一次该单词在全部上下文中的位置，从而编码段落中后续词汇的信息。
## 总结思考
和之前的re2有一定的相似性 通过这种方法来了解re2是如何在不同的领域发挥作用的
## 论文阅读中的问题
### 如何从模型中提取嵌入？
按照惯例，我们从最后一层隐藏层的激活中提取嵌入。每个位于位置j的输入词元xj都与一个上下文化的词元嵌入相关联，该嵌入是隐藏层的表示φj(x)
从最后一层隐藏层的激活中输出嵌入，输出方法是找到输入句子句子对应位置的嵌入编码
### 证明回声模型可以解决问题的实验
1. 构造实验对 q : [A, B]; s+ : [A, B+]; s− : [A, B−]
2. b+是b的改写 b-和b完全没有关系
3. 全部输入回声模型 检测a部分的嵌入
4. 发现b+和b的相似度高于b-证明后文的部分被接受
通过构造实验可以发现，回声嵌入让前文的嵌入包含了后文的信息，这样做到了上下文一致性。
## 代码阅读
给出的代码非常的简单，仅仅包含一个用来演示回声模型是如何进行文本查找的例子
[代码链接](https:%20//github.com/jakespringer/echo-embeddings.)

example.py
```
from echo_embeddings import EchoEmbeddingsMistral, EchoPooling, EchoParser
import torch

# These are the templates for the model.
# Tips:
# - Always include a beginning of sentence tag <s> (it isn't added for you!)
# - The parser will replace variables and compute embeddings on things inside of braces, 
#   so be sure to reference variables inside of braces only (e.g. {!%%prompt%%,} will be 
#   replaced with the prompt, and {%%text%%} will be replaced with the text)
# - The pooling will take the {mean, last} of the token embeddings that are inside braces
#   except when the braces start with {! which means the text won't be included}. See usage
#   in the example below.
# - Example: "<s>The last-token of {this text %%text%% will be </s>} even though there
#             is {!text after it.</s>}"
# - When using max_tokens, the parser will enforce that every separate {} has at most 
#   max_tokens; this means that if you have multiple braces, the max_tokens will be
#   enforced for each set of braces separately. This is why {</s>} is enclosed in 
#   separate braces: so that </s> will not be cut off if %%text%% exceeds the max_tokens.
templates = {
    'query': '<s>Instruct:{!%%prompt%%,}\nQuery:{!%%text%%}\nQuery again:{%%text%%}{</s>}',
    'document': '<s>Document:{!%%text%%}\nDocument again:{%%text%%}{</s>}',
}

# Create the model  需要加载hf模型
path_to_model = 'jspringer/echo-mistral-7b-instruct-lasttoken'
model = EchoEmbeddingsMistral.from_pretrained(path_to_model)
model = model.eval()

# Create the parser 主要负责将文字变成embeding   
parser = EchoParser(path_to_model, templates, max_length=300)

# Create the pooling: strategy can either be mean or last  主要负责将嵌入提取出来
pooling = EchoPooling(strategy='last')

# specify the prompt, queries, and documents  例子
prompt = 'Retrieve passages that answer the question'
queries = [
    'What is the capital of France?',
    'What is the capital of Deutschland?',
]
documents = [
    'Paris is the capital of France.',
    'Berlin is the capital of Germany.',
]

query_variables = [{'prompt': prompt, 'text': q} for q in queries]
document_variables = [{'text': d} for d in documents]

query_tagged = [('query', q) for q in query_variables]
document_tagged = [('document', d) for d in document_variables]

# Get the tokenized embeddings   
# 先输入到parser中 获取token化后的文字 然后放进model中获取所有文字的嵌入 最后pooling获取需要的结果
with torch.no_grad():
    query_embeddings = pooling(model(parser(query_tagged)))['sentence_embedding']
    document_embeddings = pooling(model(parser(document_tagged)))['sentence_embedding']

# compute the cosine similarity  计算各种余弦相似度
sim = lambda x, y: torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

print('Similarity between the queries and documents:')
for i, q in enumerate(queries): #  两两配对
    for j, d in enumerate(documents):
        similarity_score = sim(query_embeddings[i], document_embeddings[j])
        print('Computing similarity between:')
        print(f'  - {q}')
        print(f'  - {d}')
        print(f'  Cosine similarity: {similarity_score:.4f}')
```
### 如何从预训练模型中提取嵌入？
就是从最后一层隐藏层的激活中提取嵌入入 代码如下：
```
 outputs = self.model(**inputs).last_hidden_state
```
