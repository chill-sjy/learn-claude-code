# LLM 深度解析：从 Transformer 到 API

> 从底层原理到上层应用：理解大模型如何工作、如何训练、如何省钱

## 目录

1. [Transformer 为什么是"续写引擎"](#transformer-为什么是续写引擎)
2. [KV 缓存如何省钱](#kv-缓存如何省钱)
3. [训练数据如何构造](#训练数据如何构造)
4. [API 消息处理机制](#api-消息处理机制)

---

## Transformer 为什么是"续写引擎"

### 你已经知道的：Encoder 和注意力机制

你理解 Transformer 的 Encoder，知道通过 QKV（Query、Key、Value）注意力机制，一段文字可以生成非常丰富的 tensor 表示。

```
输入: "The cat sat on the mat"
     ↓ (Embedding)
     [0.2, 0.5, ...], [0.1, 0.8, ...], ...
     ↓ (Self-Attention: Q·K^T)
     计算每个词和其他词的关系
     ↓ (Attention·V)
     生成融合了上下文信息的 tensor
```

但这只是 **Encoder** — 它把输入"编码"成一个丰富的表示，但不生成新内容。

### Decoder：自回归生成

大语言模型（如 GPT、Claude）用的是 **Decoder-only** 架构，核心是**自回归生成**（autoregressive generation）：

```
已有文本: "The cat sat on the"
         ↓
      [模型推理]
         ↓
      预测下一个词: "mat" (概率 0.8)
                    "floor" (概率 0.15)
                    "chair" (概率 0.05)
```

**关键机制：Causal Attention（因果注意力）**

```python
# 普通 Attention（Encoder 用）：每个词能看到所有词
Attention mask:
  The  cat  sat  on  the  mat
The  ✓    ✓    ✓   ✓   ✓   ✓
cat  ✓    ✓    ✓   ✓   ✓   ✓
sat  ✓    ✓    ✓   ✓   ✓   ✓

# Causal Attention（Decoder 用）：每个词只能看到之前的词
Attention mask:
  The  cat  sat  on  the  mat
The  ✓    ✗    ✗   ✗   ✗   ✗
cat  ✓    ✓    ✗   ✗   ✗   ✗
sat  ✓    ✓    ✓   ✗   ✗   ✗
on   ✓    ✓    ✓   ✓   ✗   ✗
the  ✓    ✓    ✓   ✓   ✓   ✗
mat  ✓    ✓    ✓   ✓   ✓   ✓
```

这个"只能看到之前的词"的限制，让模型学会了**根据前文预测下一个词**。

### 为什么是"续写引擎"？

**训练目标：预测下一个 token**

```python
# 训练数据
text = "The cat sat on the mat"

# 训练时的样本
输入: "The"              → 目标: "cat"
输入: "The cat"          → 目标: "sat"
输入: "The cat sat"      → 目标: "on"
输入: "The cat sat on"   → 目标: "the"
输入: "The cat sat on the" → 目标: "mat"
```

模型通过数万亿个这样的样本，学会了：
1. 语法规则（"The" 后面通常跟名词）
2. 语义关系（"cat" 和 "sat" 经常一起出现）
3. 世界知识（"sat on the mat" 是合理的场景）

**推理时：逐个生成 token**

```python
# 用户输入
prompt = "The capital of France is"

# 模型生成过程
step 1: "The capital of France is" → 预测 "Paris" (概率最高)
step 2: "The capital of France is Paris" → 预测 "." (概率最高)
step 3: "The capital of France is Paris." → 预测 <|end|> (停止)
```

每一步都是"给定前文，预测下一个词"，所以叫**续写引擎**。

### 从 Tensor 到概率分布

你提到"生成非常丰富的 tensor"，这个 tensor 最后会转换成**词表上的概率分布**：

```python
# 假设词表大小是 50000
输入: "The cat sat on the"
     ↓ (Embedding + Transformer layers)
     tensor: [768 维向量]  # 最后一个 token 的表示
     ↓ (Linear layer: 768 → 50000)
     logits: [50000 维向量]  # 每个词的"得分"
     ↓ (Softmax)
     probabilities: [50000 维向量]  # 每个词的概率

     例如:
     probabilities[词表中"mat"的位置] = 0.8
     probabilities[词表中"floor"的位置] = 0.15
     probabilities[词表中"chair"的位置] = 0.05
```

**采样策略：**
- **Greedy decoding**：总是选概率最高的词（确定性，但可能重复）
- **Top-k sampling**：从概率最高的 k 个词中随机选（更多样）
- **Temperature**：调整概率分布的"尖锐度"（temperature=0.7 更保守，1.5 更随机）

### 完整的生成流程

```python
def generate(prompt, max_tokens=100):
    tokens = tokenize(prompt)  # "The cat" → [123, 456]

    for _ in range(max_tokens):
        # 1. 把所有已生成的 token 喂给模型
        logits = model(tokens)  # shape: [seq_len, vocab_size]

        # 2. 只看最后一个 token 的预测
        next_token_logits = logits[-1]  # shape: [vocab_size]

        # 3. 转换成概率
        probs = softmax(next_token_logits)

        # 4. 采样下一个 token
        next_token = sample(probs)  # 例如: 789

        # 5. 追加到序列
        tokens.append(next_token)

        # 6. 如果生成了结束符，停止
        if next_token == END_TOKEN:
            break

    return detokenize(tokens)
```

**关键点：每次都要把整个序列重新喂给模型**，因为 Causal Attention 需要看到所有之前的 token。

这就是为什么需要 KV 缓存（下一节）。

---

## KV 缓存如何省钱

### 你的疑问

> tokenize 只是把文本转成 token，省略 tokenize 环节并不能省计算。模型的 Transformer 层还是要跑的，为什么 KV 缓存能省钱？

你的理解是对的：**tokenize 不是瓶颈，Transformer 计算才是**。

### KV 缓存省的不是 tokenize，而是 Attention 计算

#### 没有 KV 缓存时的问题

```python
# 第 1 轮：生成第 1 个 token
输入: "The cat sat on the"  # 5 个 token
模型计算: 5 个 token 的 Attention
输出: "mat"

# 第 2 轮：生成第 2 个 token
输入: "The cat sat on the mat"  # 6 个 token
模型计算: 6 个 token 的 Attention  # 前 5 个 token 又算了一遍！
输出: "."

# 第 3 轮：生成第 3 个 token
输入: "The cat sat on the mat."  # 7 个 token
模型计算: 7 个 token 的 Attention  # 前 6 个 token 又算了一遍！
输出: <|end|>
```

**问题：前面的 token 在每一轮都被重复计算**。

#### Attention 的计算过程

```python
# Self-Attention 的核心公式
Q = input @ W_q  # Query
K = input @ W_k  # Key
V = input @ W_v  # Value

Attention = softmax(Q @ K^T / sqrt(d)) @ V
```

**关键观察：K 和 V 只依赖于输入，不依赖于未来的 token**

```python
# 第 1 轮
输入: ["The", "cat", "sat", "on", "the"]
K1 = [k_The, k_cat, k_sat, k_on, k_the]  # 5 个 Key 向量
V1 = [v_The, v_cat, v_sat, v_on, v_the]  # 5 个 Value 向量

# 第 2 轮
输入: ["The", "cat", "sat", "on", "the", "mat"]
K2 = [k_The, k_cat, k_sat, k_on, k_the, k_mat]  # 前 5 个和 K1 完全一样！
V2 = [v_The, v_cat, v_sat, v_on, v_the, v_mat]  # 前 5 个和 V1 完全一样！
```

**KV 缓存的思路：把已经计算过的 K 和 V 存起来，不要重复计算**。

#### 有 KV 缓存时

```python
# 第 1 轮
输入: ["The", "cat", "sat", "on", "the"]
计算: K1, V1  # 5 个 token 的 K 和 V
缓存: cache_K = K1, cache_V = V1
输出: "mat"

# 第 2 轮
输入: ["mat"]  # 只需要处理新 token！
计算: k_mat, v_mat  # 只计算 1 个 token 的 K 和 V
缓存: cache_K = [K1, k_mat], cache_V = [V1, v_mat]  # 追加到缓存
Attention: Q_mat @ [K1, k_mat]^T @ [V1, v_mat]  # 用缓存的 K 和 V
输出: "."

# 第 3 轮
输入: ["."]
计算: k_dot, v_dot
缓存: cache_K = [K1, k_mat, k_dot], cache_V = [V1, v_mat, v_dot]
Attention: Q_dot @ cache_K^T @ cache_V
输出: <|end|>
```

**节省的计算：**
- 没有缓存：第 n 轮需要计算 n 个 token 的 K 和 V
- 有缓存：第 n 轮只需要计算 1 个新 token 的 K 和 V

### 计算量对比

假设生成 100 个 token：

**没有 KV 缓存：**
```
第 1 轮: 计算 1 个 token
第 2 轮: 计算 2 个 token
第 3 轮: 计算 3 个 token
...
第 100 轮: 计算 100 个 token

总计算量: 1 + 2 + 3 + ... + 100 = 5050 次 token 计算
```

**有 KV 缓存：**
```
第 1 轮: 计算 1 个 token，缓存
第 2 轮: 计算 1 个 token，追加缓存
第 3 轮: 计算 1 个 token，追加缓存
...
第 100 轮: 计算 1 个 token，追加缓存

总计算量: 100 次 token 计算
```

**节省：5050 / 100 = 50 倍！**

### Claude 的 Prompt 缓存

Claude API 的 prompt 缓存更进一步：**跨请求共享缓存**。

```python
# 第 1 次请求
system = "You are a helpful assistant. [很长的 system prompt，5000 tokens]"
messages = [{"role": "user", "content": "What is 2+2?"}]

# API 服务计算 system prompt 的 KV，缓存 5 分钟
# 计算量: 5000 tokens

# 第 2 次请求（5 分钟内）
system = "You are a helpful assistant. [同样的 system prompt]"
messages = [{"role": "user", "content": "What is 3+3?"}]

# API 服务发现 system prompt 的 hash 一样，直接用缓存
# 计算量: 0 tokens（system prompt 部分）
```

**省钱的地方：**
1. **计算成本** — 不需要重复计算 Attention（GPU 时间）
2. **输入 token 费用** — Claude 对缓存命中的 token 收费更低
   - 正常输入：$15 / 1M tokens
   - 缓存写入：$18.75 / 1M tokens（首次）
   - 缓存读取：$1.50 / 1M tokens（命中时）

如果你的 system prompt 有 5000 tokens，每次请求都用：
- 没有缓存：5000 tokens × $15 = $0.075
- 有缓存（命中）：5000 tokens × $1.50 = $0.0075

**省了 10 倍！**

### 为什么 API 提供商愿意这样做？

1. **降低服务器负载** — 减少 GPU 计算，可以服务更多用户
2. **提高响应速度** — 缓存命中时，首 token 延迟更低
3. **鼓励长 context 使用** — 用户不怕用长 system prompt，提升体验

这是双赢：用户省钱，服务商省算力。

---

## 训练数据如何构造

### 你的疑问

> 训练时需要构造成千上万的 `<|system|>...<|user|>...<|assistant|>` 对话吗？互联网上的文本大多不是对话格式，怎么构造这么多训练数据？

### 训练分为两个阶段

#### 阶段 1：预训练（Pretraining）— 学习语言本身

**数据来源：互联网上的所有文本**

```
维基百科、GitHub 代码、Reddit 讨论、新闻文章、书籍、论文...
```

**训练目标：预测下一个 token**

```python
# 原始文本（不需要对话格式）
text = "Paris is the capital of France. It is known for the Eiffel Tower."

# 训练样本（自动生成）
"Paris" → "is"
"Paris is" → "the"
"Paris is the" → "capital"
"Paris is the capital" → "of"
...
```

**不需要人工标注**，只要有文本就能训练。这一阶段的模型学会了：
- 语法（"is" 后面跟名词）
- 事实（"Paris" 和 "France" 的关系）
- 推理（"capital" 暗示政治地位）

**但这个模型不会"对话"**，你问它 "What is the capital of France?"，它可能续写成：

```
"What is the capital of France? What is the capital of Germany? What is..."
```

因为它只是在"续写"，不知道你在问问题。

#### 阶段 2：指令微调（Instruction Tuning）— 学习对话

**数据来源：人工标注的对话**

```python
# 对话格式的训练数据
{
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
}
```

**拼接成特殊格式：**

```
<|user|>What is the capital of France?<|end|>
<|assistant|>The capital of France is Paris.<|end|>
```

**训练目标：只预测 assistant 部分**

```python
# 训练时的 loss 计算
输入: "<|user|>What is the capital of France?<|end|><|assistant|>"
目标: "The capital of France is Paris.<|end|>"

# 关键：只对 assistant 的 token 计算 loss
loss = 0
for token in ["The", "capital", "of", "France", "is", "Paris", ".", "<|end|>"]:
    loss += cross_entropy(model_output, token)
# user 部分的 token 不计入 loss
```

这样模型学会了：
- 看到 `<|user|>` 后面是问题
- 看到 `<|assistant|>` 后面应该生成回答
- 回答应该直接、有用、礼貌

### 数据构造的实际做法

#### 1. 人工标注（高质量，但昂贵）

```python
# 标注员写的对话
{
    "user": "How do I sort a list in Python?",
    "assistant": "You can use the `sorted()` function:\n\n```python\nmy_list = [3, 1, 2]\nsorted_list = sorted(my_list)\n```"
}
```

**规模：** 几万到几十万条（GPT-3.5 用了约 13,000 条高质量对话）

#### 2. 从现有数据转换

**Stack Overflow → 问答对**

```python
# 原始数据
question = "How to reverse a string in Python?"
answer = "Use slicing: `s[::-1]`"

# 转换成对话格式
{
    "user": "How to reverse a string in Python?",
    "assistant": "Use slicing: `s[::-1]`"
}
```

**Reddit → 对话**

```python
# 原始数据
post = "What's your favorite Python library?"
comment = "I love `requests` for HTTP calls."

# 转换成对话格式
{
    "user": "What's your favorite Python library?",
    "assistant": "I love `requests` for HTTP calls."
}
```

#### 3. 模型生成（Self-Instruct）

用已有的模型生成训练数据：

```python
# Prompt 给 GPT-4
prompt = """
Generate 10 diverse user questions about Python programming.
For each question, provide a helpful answer.
"""

# GPT-4 生成
[
    {"user": "How to read a file in Python?", "assistant": "Use `open()`..."},
    {"user": "What is a list comprehension?", "assistant": "It's a concise way..."},
    ...
]
```

**规模：** 可以生成几十万条（Alpaca 用 GPT-3.5 生成了 52,000 条）

#### 4. RLHF（人类反馈强化学习）

```python
# 1. 模型生成多个回答
user_query = "Explain quantum computing"
responses = [
    model.generate(user_query) for _ in range(4)
]

# 2. 人类标注员排序
rankings = human_annotator.rank(responses)  # [3, 1, 4, 2]

# 3. 训练 reward model
reward_model.train(responses, rankings)

# 4. 用 reward model 优化生成模型
model = PPO_optimize(model, reward_model)
```

这一步让模型学会：
- 更有用（不是正确但无用的回答）
- 更安全（不生成有害内容）
- 更符合人类偏好（礼貌、清晰）

### 训练数据的规模

| 阶段 | 数据量 | 数据来源 |
|------|--------|---------|
| **预训练** | 数万亿 tokens | 互联网文本（自动爬取） |
| **指令微调** | 几万到几十万条对话 | 人工标注 + 数据转换 + 模型生成 |
| **RLHF** | 几万到几十万条排序 | 人类标注员排序 |

**关键点：**
- 预训练用的是**海量无标注数据**（便宜，但模型不会对话）
- 指令微调用的是**少量高质量对话**（贵，但让模型学会对话）
- RLHF 用的是**人类偏好排序**（最贵，但让模型更有用）

### 为什么不需要"亿亿的对话"？

因为**预训练已经学会了语言**，指令微调只是"教它怎么用"。

类比：
- **预训练** = 学习英语语法、词汇、常识（需要大量阅读）
- **指令微调** = 学习如何回答问题、写邮件（只需要几千个例子）

就像你学会了英语，不需要看"亿亿封邮件"就能写邮件，因为你已经懂语言了。

---

## API 消息处理机制

### 问题

当你调用 Claude API 时：

```python
client.messages.create(
    model="claude-opus-4-6",
    system="You are a helpful assistant",
    messages=[
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Let me calculate that."},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "123", "content": "4"}
        ]}
    ]
)
```

这些结构化的 JSON 数据是直接喂给模型的吗？还是 API 服务会做转换？

### 答案：API 服务会拼接和格式化

```
你的代码                API 服务层              模型
   |                      |                    |
   |  messages + system   |                    |
   |--------------------->|                    |
   |                      | 拼接成一个字符串    |
   |                      | 加特殊 token       |
   |                      |------------------->|
   |                      |                    | tokenize
   |                      |                    | 推理
   |                      |<-------------------|
   |<---------------------|                    |
```

### 模型实际"看到"的内容

**你发送的（高层抽象）：**

```python
{
    "system": "You are a helpful assistant",
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Let me calculate that."},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "123", "content": "4"}
        ]}
    ]
}
```

### API 服务拼接后（简化示例）

```
<|system|>You are a helpful assistant<|end|>
<|user|>What is 2+2?<|end|>
<|assistant|>Let me calculate that.<|end|>
<|user|><tool_result id="123">4</tool_result><|end|>
<|assistant|>
```

### 模型实际接收的（token 序列）

```
[1234, 5678, 9012, 3456, 7890, ...]
```

模型看到的是**一整段文本的 token 序列**，然后继续生成下一个 token（详见"Transformer 为什么是续写引擎"章节）。

### 特殊 token 是训练时学会的

训练数据中，`<|assistant|>` 后面总是跟着"助手应该说的话"，模型通过统计学习到了这个模式（详见"训练数据如何构造"章节）。

### 不同模型用不同的格式

**Claude (Anthropic):**
```
<|system|>...<|end|>
<|user|>...<|end|>
<|assistant|>...<|end|>
```

**GPT (OpenAI):**
```
<|im_start|>system
...<|im_end|>
<|im_start|>user
...<|im_end|>
<|im_start|>assistant
...<|im_end|>
```

**Llama (Meta):**
```
<s>[INST] <<SYS>>
...
<</SYS>>

... [/INST] ... </s>
```

这些格式是训练时定义的，API 服务必须按照训练格式拼接。这就是为什么不同模型的 API 不兼容。

### Tool use 也是拼接出来的

**你发送的 tool result:**
```python
{"type": "tool_result", "tool_use_id": "toolu_123", "content": "file1.py\nfile2.py"}
```

**API 拼成（Claude 格式）:**
```xml
<tool_result id="toolu_123">
file1.py
file2.py
</tool_result>
```

**模型输出 tool use:**
```xml
<tool_use id="toolu_456" name="bash">
{"command": "ls -la"}
</tool_use>
```

API 服务识别 `<tool_use>` 标签，解析后转成 JSON 返回。

### API 服务的作用

### API 服务的作用

| 功能 | 说明 |
|------|------|
| **格式转换** | 把你的 JSON 转成模型训练时的格式 |
| **Token 管理** | 确保不超过 context window（如 200K tokens） |
| **流式输出** | 把模型生成的 token 实时返回给你 |
| **工具调用解析** | 识别模型输出的 `<tool_use>` 标签，转成 JSON |
| **错误处理** | 如果模型输出格式错误，重试或返回错误 |

### 完整流程示例

**1. 你的代码:**
```python
response = client.messages.create(
    model="claude-opus-4-6",
    system="You are a Python expert",
    messages=[{"role": "user", "content": "Write a hello world script"}],
    tools=[{"name": "bash", "description": "Run bash commands", ...}]
)
```

**2. API 服务拼接:**
```
<|system|>You are a Python expert
Available tools: bash...<|end|>
<|user|>Write a hello world script<|end|>
<|assistant|>
```

**3. 模型生成 → 4. API 解析返回 → 5. 你执行工具 → 6. 再次拼接 → 7. 最终响应**

（完整流程见 s01-the-agent-loop.md）

---

## 总结

| 层级 | 你看到的 | 实际发生的 |
|------|---------|-----------|
| **你的代码** | `messages` 列表（JSON） | 高层抽象 |
| **API 服务** | 拼接 + 格式化 | `<|user|>...<|assistant|>...` |
| **模型输入** | Token 序列 | `[1234, 5678, 9012, ...]` |
| **模型处理** | Transformer 推理 | 自回归生成（预测下一个 token） |
| **模型输出** | Token 序列 | `[3456, 7890, 2345, ...]` |
| **API 服务** | 解析 + 转换 | 识别 `<tool_use>` 等标签 |
| **你收到的** | JSON 响应 | `{"role": "assistant", "content": [...]}` |

**核心理解：**
1. **模型是续写引擎** — 通过 Causal Attention 学会根据前文预测下一个 token
2. **KV 缓存省算力** — 避免重复计算已处理 token 的 Key 和 Value，节省 50 倍计算量
3. **训练分两阶段** — 预训练学语言（海量无标注数据），指令微调学对话（少量高质量对话）
4. **API 做格式转换** — 把 JSON 转成模型训练时的特殊格式，把模型输出转回 JSON
