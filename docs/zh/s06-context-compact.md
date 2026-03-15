# s06: Context Compact (上下文压缩)

`s01 > s02 > s03 > s04 > s05 > [ s06 ] | s07 > s08 > s09 > s10 > s11 > s12`

> *"上下文总会满, 要有办法腾地方"* -- 三层压缩策略, 换来无限会话。

## 问题

上下文窗口是有限的。读一个 1000 行的文件就吃掉 ~4000 token; 读 30 个文件、跑 20 条命令, 轻松突破 100k token。不压缩, 智能体根本没法在大项目里干活。

## 解决方案

三层压缩, 激进程度递增:

```
Every turn:
+------------------+
| Tool call result |
+------------------+
        |
        v
[Layer 1: micro_compact]        (silent, every turn)
  Replace tool_result > 3 turns old
  with "[Previous: used {tool_name}]"
        |
        v
[Check: tokens > 50000?]
   |               |
   no              yes
   |               |
   v               v
continue    [Layer 2: auto_compact]
              Save transcript to .transcripts/
              LLM summarizes conversation.
              Replace all messages with [summary].
                    |
                    v
            [Layer 3: compact tool]
              Model calls compact explicitly.
              Same summarization as auto_compact.
```

## 工作原理

### 第一层 -- micro_compact

每次 LLM 调用前, 将旧的 tool result 替换为占位符 (保留最近 3 个)。

**压缩前的 messages:**

```json
[
  {"role": "user", "content": "Read file1.py, file2.py, file3.py, file4.py"},
  {"role": "assistant", "content": [
    {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "file1.py"}}
  ]},
  {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "call_1", "content": "def foo():\n    return 42\n# ... 500 lines ..."}
  ]},
  {"role": "assistant", "content": [
    {"type": "tool_use", "id": "call_2", "name": "read_file", "input": {"path": "file2.py"}}
  ]},
  {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "call_2", "content": "class Bar:\n    pass\n# ... 800 lines ..."}
  ]},
  {"role": "assistant", "content": [
    {"type": "tool_use", "id": "call_3", "name": "read_file", "input": {"path": "file3.py"}}
  ]},
  {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "call_3", "content": "import os\n# ... 300 lines ..."}
  ]},
  {"role": "assistant", "content": [
    {"type": "tool_use", "id": "call_4", "name": "read_file", "input": {"path": "file4.py"}}
  ]},
  {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "call_4", "content": "# Current file content..."}
  ]}
]
```

**micro_compact 后 (保留最近 3 个 tool_result, 压缩旧的):**

```json
[
  {"role": "user", "content": "Read file1.py, file2.py, file3.py, file4.py"},
  {"role": "assistant", "content": [
    {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "file1.py"}}
  ]},
  {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "call_1", "content": "[Previous: used read_file]"}
  ]},
  {"role": "assistant", "content": [
    {"type": "tool_use", "id": "call_2", "name": "read_file", "input": {"path": "file2.py"}}
  ]},
  {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "call_2", "content": "class Bar:\n    pass\n# ... 800 lines ..."}
  ]},
  {"role": "assistant", "content": [
    {"type": "tool_use", "id": "call_3", "name": "read_file", "input": {"path": "file3.py"}}
  ]},
  {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "call_3", "content": "import os\n# ... 300 lines ..."}
  ]},
  {"role": "assistant", "content": [
    {"type": "tool_use", "id": "call_4", "name": "read_file", "input": {"path": "file4.py"}}
  ]},
  {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "call_4", "content": "# Current file content..."}
  ]}
]
```

**关键点:** file1.py 的完整内容被替换为占位符, 节省了 ~2000 tokens。

---

### 第二层 -- auto_compact

当 token 总数超过阈值 (如 50000), 保存完整对话到磁盘, 让 LLM 生成摘要, 用摘要替换所有历史消息。

**压缩前 (假设已经有 20 轮对话, 50000+ tokens):**

```json
[
  {"role": "user", "content": "Read all Python files in agents/"},
  {"role": "assistant", "content": "I'll read them one by one..."},
  // ... 18 more turns of tool calls and results ...
  {"role": "user", "content": "Now analyze the architecture"},
  {"role": "assistant", "content": [
    {"type": "tool_use", "id": "call_20", "name": "bash", "input": {"command": "tree agents/"}}
  ]},
  {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "call_20", "content": "agents/\n├── s01_hello.py\n├── s02_tools.py\n..."}
  ]}
]
```

**auto_compact 后:**

```json
[
  {"role": "user", "content": "[Conversation compressed. Transcript: .transcripts/transcript_1710489234.jsonl]\n\n**Summary:**\n\n1. **Accomplished:** Read 15 Python files in agents/ directory (s01-s06 series). Analyzed code structure.\n\n2. **Current state:** All files use similar pattern: tool definitions → agent_loop → main. Key insight: each stage builds on previous (s01=hello, s02=tools, s03=loop, s04=streaming, s05=skills, s06=compact).\n\n3. **Key decisions:** Identified that s06_context_compact.py implements 3-layer compression (micro/auto/manual). THRESHOLD=5000 tokens, KEEP_RECENT=3 tool results."},
  {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing."}
]
```

**关键点:** 50000 tokens 压缩到 ~500 tokens, 但核心信息保留。完整历史在 `.transcripts/` 可恢复。

---

### 第三层 -- manual compact

模型主动调用 `compact` 工具, 触发与 auto_compact 相同的摘要机制。

**使用场景:** 模型判断当前对话已经完成一个阶段, 主动压缩以准备下一阶段。

---

### 完整流程整合

```python
def agent_loop(messages: list):
    while True:
        micro_compact(messages)                        # Layer 1: 每次调用前
        if estimate_tokens(messages) > THRESHOLD:
            messages[:] = auto_compact(messages)       # Layer 2: 超阈值时
        response = client.messages.create(...)
        # ... tool execution ...
        if manual_compact:
            messages[:] = auto_compact(messages)       # Layer 3: 模型主动触发
```

完整历史通过 transcript 保存在磁盘上。信息没有真正丢失, 只是移出了活跃上下文。

---

## 常见疑问

### Q1: micro_compact 会不会丢失重要信息?

**不会。** 原因:

1. **模型已经提取了关键信息** - 模型读完文件后, 会在自己的 assistant 回复中总结重要内容 (这部分不会被 micro_compact 清除)
2. **按需重新读取** - 如果后续真的需要某个文件内容, 模型会再次调用 `read_file`
3. **只压缩旧结果** - 保留最近 3 个 tool_result, 确保当前工作上下文完整

**案例:** "熟悉项目架构" 的正确做法

```
User: 熟悉一下这个项目的架构

Assistant (turn 1):
  [调用 bash tool: ls -la]
  [调用 read_file: README.md]

  "这是一个 Python 项目, 主要目录:
   - agents/: 包含 s01-s06 系列教程代码
   - docs/: 中文文档
   - 核心模式: 每个 stage 都实现 agent_loop + tools
   我需要读几个关键文件来理解架构..."

Assistant (turn 2):
  [调用 read_file: agents/s01_hello.py]  # tool_result_1
  [调用 read_file: agents/s06_context_compact.py]  # tool_result_2

  "架构总结:
   - s01: 最简 agent loop (system + user message)
   - s06: 完整实现 (tools + streaming + compression)
   - 共同模式: SYSTEM prompt + agent_loop + tool handlers
   - 关键设计: 三层压缩 (micro/auto/manual)"

User (turn 3): 那 s03 是做什么的?

Assistant (turn 3):
  [此时 tool_result_1 已被 micro_compact 压缩]
  [但 assistant 的架构总结还在, 可以基于总结回答]
  [如果需要 s03 细节, 会重新调用 read_file]
```

**关键:** 模型不会盲目读取所有文件, 而是:
1. 先读目录结构和 README
2. 读几个代表性文件 (如 s01, s06)
3. **在 assistant 回复中主动总结架构** (这个总结不会被压缩)
4. 后续问题基于总结回答, 需要时再读具体文件

### Q2: 为什么不直接用 summarize, 而要先 micro_compact?

**效率。** micro_compact 是轻量级操作 (字符串替换), 不需要调用 LLM。只有在 token 真正超限时才触发昂贵的 summarize。

### Q3: transcript 文件有什么用?

1. **调试** - 出问题时可以回溯完整对话
2. **恢复** - 理论上可以从 transcript 恢复被压缩的上下文 (虽然当前代码没实现)
3. **分析** - 可以分析模型的工具使用模式, 优化 prompt

## 相对 s05 的变更

| 组件           | 之前 (s05)       | 之后 (s06)                     |
|----------------|------------------|--------------------------------|
| Tools          | 5                | 5 (基础 + compact)             |
| 上下文管理     | 无               | 三层压缩                       |
| Micro-compact  | 无               | 旧结果 -> 占位符               |
| Auto-compact   | 无               | token 阈值触发                 |
| Transcripts    | 无               | 保存到 .transcripts/           |

## 试一试

```sh
cd learn-claude-code
python agents/s06_context_compact.py
```

试试这些 prompt (英文 prompt 对 LLM 效果更好, 也可以用中文):

1. `Read every Python file in the agents/ directory one by one` (观察 micro-compact 替换旧结果)
2. `Keep reading files until compression triggers automatically`
3. `Use the compact tool to manually compress the conversation`
