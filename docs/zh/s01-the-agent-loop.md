# s01: The Agent Loop (智能体循环)

`[ s01 ] s02 > s03 > s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`

> *"One loop & Bash is all you need"* -- 一个工具 + 一个循环 = 一个智能体。

## 问题

语言模型能推理代码, 但碰不到真实世界 -- 不能读文件、跑测试、看报错。没有循环, 每次工具调用你都得手动把结果粘回去。你自己就是那个循环。

## 解决方案

```
+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> |  Tool   |
| prompt |      |       |      | execute |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                    (loop until stop_reason != "tool_use")
```

一个退出条件控制整个流程。循环持续运行, 直到模型不再调用工具。

## 工作原理

1. 用户 prompt 作为第一条消息。

```python
messages.append({"role": "user", "content": query})
```

2. 将消息和工具定义一起发给 LLM。

```python
response = client.messages.create(
    model=MODEL, system=SYSTEM, messages=messages,
    tools=TOOLS, max_tokens=8000,
)
```

3. 追加助手响应。检查 `stop_reason` -- 如果模型没有调用工具, 结束。

```python
messages.append({"role": "assistant", "content": response.content})
if response.stop_reason != "tool_use":
    return
```

4. 执行每个工具调用, 收集结果, 作为 user 消息追加。回到第 2 步。

```python
results = []
for block in response.content:
    if block.type == "tool_use":
        output = run_bash(block.input["command"])
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
messages.append({"role": "user", "content": results})
```

组装为一个完整函数:

```python
def agent_loop(query):
    messages = [{"role": "user", "content": query}]
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_bash(block.input["command"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

不到 30 行, 这就是整个智能体。后面 11 个章节都在这个循环上叠加机制 -- 循环本身始终不变。

## Claude vs OpenAI 协议差异

Claude (Anthropic) 和 OpenAI 的 tool use 协议有显著不同。理解的关键是区分**接收**（模型返回什么）和**发送**（你如何返回工具结果）。

### 一次完整的工具调用流程

#### 阶段 1: 接收模型响应（模型想调用工具）

**Claude 返回的结构:**

```json
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "让我帮你查一下"},
    {"type": "tool_use", "id": "toolu_abc", "name": "bash", "input": {"command": "ls"}},
    {"type": "tool_use", "id": "toolu_xyz", "name": "bash", "input": {"command": "pwd"}}
  ],
  "stop_reason": "tool_use"
}
```

**OpenAI 返回的结构:**

```json
{
  "role": "assistant",
  "content": "让我帮你查一下",
  "tool_calls": [
    {"id": "call_abc", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}},
    {"id": "call_xyz", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"pwd\"}"}}
  ],
  "finish_reason": "tool_calls"
}
```

**接收阶段的差异:**
- **Claude**: `content` 是列表，文本和工具调用都是 block
- **OpenAI**: `content` 和 `tool_calls` 是两个独立字段

#### 阶段 2: 发送工具结果（你执行完工具后）

**Claude 发送的消息:**

```python
# 工具结果作为 user 消息发送
{
    "role": "user",
    "content": [
        {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "file1.py\nfile2.py"},
        {"type": "tool_result", "tool_use_id": "toolu_xyz", "content": "/home/user"}
    ]
}
```

**OpenAI 发送的消息:**

```python
# 工具结果作为独立的 tool 角色消息发送
[
    {"role": "tool", "tool_call_id": "call_abc", "content": "file1.py\nfile2.py"},
    {"role": "tool", "tool_call_id": "call_xyz", "content": "/home/user"}
]
```

**发送阶段的差异（核心）:**
- **Claude**: 工具结果放在 `role: "user"` 消息里，用 `tool_result` block
- **OpenAI**: 工具结果用专门的 `role: "tool"` 消息

### 完整对话示例

**Claude 的消息历史:**

```python
messages = [
    # 1. 用户请求
    {"role": "user", "content": "列出所有 Python 文件"},

    # 2. 模型响应（想调用工具）
    {"role": "assistant", "content": [
        {"type": "text", "text": "让我帮你查一下"},
        {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "find . -name '*.py'"}}
    ]},

    # 3. 你发送工具结果（role: user）
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file1.py\nfile2.py\nfile3.py"}
    ]},

    # 4. 模型最终响应
    {"role": "assistant", "content": [
        {"type": "text", "text": "找到 3 个 Python 文件: file1.py, file2.py, file3.py"}
    ]}
]
```

**OpenAI 的消息历史:**

```python
messages = [
    # 1. 用户请求
    {"role": "user", "content": "列出所有 Python 文件"},

    # 2. 模型响应（想调用工具）
    {"role": "assistant", "content": "让我帮你查一下", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\":\"find . -name '*.py'\"}"}}
    ]},

    # 3. 你发送工具结果（role: tool）
    {"role": "tool", "tool_call_id": "call_1", "content": "file1.py\nfile2.py\nfile3.py"},

    # 4. 模型最终响应
    {"role": "assistant", "content": "找到 3 个 Python 文件: file1.py, file2.py, file3.py"}
]
```

### 核心差异总结

| 维度 | Claude | OpenAI |
|------|--------|--------|
| **接收：模型响应结构** | `content` 是列表，包含 `text` 和 `tool_use` block | `content` + `tool_calls` 两个字段 |
| **发送：工具结果角色** | `role: "user"` + `tool_result` block | `role: "tool"` 独立消息 |
| **并行工具调用** | `content` 列表里多个 `tool_use` | `tool_calls` 数组里多个调用 |
| **停止原因** | `stop_reason: "tool_use"` | `finish_reason: "tool_calls"` |

### 代码实现对比

**Claude 的 agent loop:**

```python
while True:
    response = client.messages.create(model=MODEL, messages=messages, tools=TOOLS)
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":
        break

    # 执行工具，收集结果
    results = []
    for block in response.content:
        if block.type == "tool_use":
            output = execute_tool(block.name, block.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output
            })

    # 工具结果作为 user 消息发送
    messages.append({"role": "user", "content": results})
```

**OpenAI 的 agent loop:**

```python
while True:
    response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    message = response.choices[0].message
    messages.append(message)

    if message.finish_reason != "tool_calls":
        break

    # 执行工具，每个结果作为独立的 tool 消息
    for tool_call in message.tool_calls:
        output = execute_tool(tool_call.function.name, json.loads(tool_call.function.arguments))
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": output
        })
```

**关键差异:**
- **Claude**: 所有工具结果打包成一个 `user` 消息
- **OpenAI**: 每个工具结果是一个独立的 `tool` 消息

### 消息角色对比

| Role | Claude | OpenAI |
|------|--------|--------|
| `system` | ❌ (通过 API 参数传递) | ✅ 系统指令 |
| `user` | ✅ 用户消息 + 工具结果 | ✅ 仅用户消息 |
| `assistant` | ✅ 助手回复 | ✅ 助手回复 |
| `tool` | ❌ (不存在) | ✅ 工具执行结果 |

**设计哲学:**
- **Claude**: 工具结果是"环境反馈"，从模型视角看就是用户提供的信息，所以用 `user` role
- **OpenAI**: 工具结果是独立的执行层，用专门的 `tool` role 区分

### 其他关键差异

### 其他关键差异

| 特性 | Claude (Anthropic) | OpenAI |
|------|-------------------|--------|
| 内容结构 | `content` 是数组，可混合 text 和 tool_use | `content` 是字符串，`tool_calls` 是独立字段 |
| 工具参数 | `input` 是原生 JSON 对象 | `arguments` 是 JSON 字符串（需要解析） |
| 停止标识 | `stop_reason: "tool_use"` | `finish_reason: "tool_calls"` |
| 工具 ID | `tool_use_id` | `tool_call_id` |
| 工具结果消息数 | 所有结果打包成 1 个 `user` 消息 | 每个结果是 1 个 `tool` 消息 |

### 并行执行工具

两个协议都支持在一个响应中返回多个工具调用：

**Claude 示例:**
```python
# 模型返回
response.content = [
    {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "find . -name '*.py'"}},
    {"type": "tool_use", "id": "toolu_2", "name": "bash", "input": {"command": "ls -la"}}
]

# 你可以并行执行这两个命令
```

**OpenAI 示例:**
```python
# 模型返回
message.tool_calls = [
    {"id": "call_1", "function": {"name": "bash", "arguments": "{\"command\":\"find . -name '*.py'\"}"}},
    {"id": "call_2", "function": {"name": "bash", "arguments": "{\"command\":\"ls -la\"}"}}
]

# 你可以并行执行这两个命令
```

生产级 agent（如 Claude Code）会并行执行这些独立的工具调用以提高效率。

## 变更内容

| 组件          | 之前       | 之后                           |
|---------------|------------|--------------------------------|
| Agent loop    | (无)       | `while True` + stop_reason     |
| Tools         | (无)       | `bash` (单一工具)              |
| Messages      | (无)       | 累积式消息列表                 |
| Control flow  | (无)       | `stop_reason != "tool_use"`    |

## 试一试

```sh
cd learn-claude-code
python agents/s01_agent_loop.py
```

试试这些 prompt (英文 prompt 对 LLM 效果更好, 也可以用中文):

1. `Create a file called hello.py that prints "Hello, World!"`
2. `List all Python files in this directory`
3. `What is the current git branch?`
4. `Create a directory called test_output and write 3 files in it`
