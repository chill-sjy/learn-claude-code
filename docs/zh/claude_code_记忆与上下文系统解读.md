# Claude Code 记忆与上下文系统解读

> 覆盖 `agents/s06_context_compact.py`、`agents/s_full.py`，以及逆向工程分析中的 `AU2`、`wU2`、`qH1` 等核心函数

---

## 0. 先回答：Claude Code 的"记忆"是什么？

与 DeerFlow 的双系统设计不同，Claude Code 的记忆机制专注于解决一个核心问题：**如何在上下文窗口有限的情况下，让 Agent 能够持续处理长对话和大型代码库**。

```
Claude Code 记忆系统
├── 上下文压缩（三层递进式压缩）    ← 当前会话的 context window 管理
│   ├── Layer 1: micro_compact        ← 每轮自动执行，轻量级
│   ├── Layer 2: auto_compact         ← 阈值触发，深度压缩
│   └── Layer 3: manual compact       ← 用户/Agent 主动触发
│
├── Transcript 持久化                ← 完整历史保存到磁盘
│   └── .transcripts/transcript_{timestamp}.jsonl
│
└── CLAUDE.md 长期记忆              ← 项目级别的持久化知识（可选）
    └── 存储项目上下文、编码规范、重要决策
```

| 维度 | Claude Code 上下文压缩 | DeerFlow 上下文压缩 | DeerFlow 长期记忆 |
|---|---|---|---|
| **目标** | 防止 context window 溢出 | 同左 | 记住"用户是谁" |
| **存储位置** | messages 列表（内存）+ .transcripts（磁盘） | State["messages"]（内存） | memory.json（磁盘文件） |
| **跨会话** | ✅ Transcript 可恢复 | ❌ 否 | ✅ 是 |
| **压缩层级** | 三层递进式 | 单层 | 不适用 |
| **触发机制** | 每轮 + 阈值 + 手动 | before_model（阈值触发） | after_agent（防抖） |
| **LLM 作用** | 8段式结构化摘要 | 对话内容压缩摘要 | 事实提取 + 用户画像 |

**类比 Java**：
- Claude Code 的三层压缩 ≈ JVM 的三级缓存（L1/L2/L3），从快速轻量到深度处理递进
- micro_compact ≈ `SoftReference` 软引用，温和地释放不紧急的资源
- auto_compact ≈ Major GC，深度清理腾出大量空间
- Transcript 持久化 ≈ 数据库事务日志（WAL），保证信息可恢复

---

## 1. 为什么需要上下文压缩？

### 1.1 问题背景

大模型的上下文窗口是有限的（如 Claude 3.5 Sonnet 是 200k tokens）。在实际的软件工程任务中，上下文消耗非常快：

```
一个真实的开发场景：
├── read_file 读取 1000 行代码           → ~4,000 tokens
├── read_file 读取另一个文件             → ~3,500 tokens
├── bash 执行命令查看目录结构             → ~500 tokens
├── grep_search 搜索代码引用              → ~2,000 tokens
├── ... 重复 20 轮操作 ...
└── 总计                                → 轻松突破 100,000+ tokens
```

如果不做任何处理，Agent 根本无法在大型代码库中持续工作。

### 1.2 核心设计思想

Claude Code 的解决方案是**三层递进式压缩**——从轻量到激进，按需启动：

```
                        压缩激进程度
                              ↑
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
     micro_compact      auto_compact      manual compact
     (每轮自动)         (92%阈值触发)      (主动调用)
           │                  │                  │
           ↓                  ↓                  ↓
     替换旧工具结果      LLM生成摘要        同auto_compact
     "[Previous:...]"   替换全部消息        用户控制时机
           │                  │                  │
     ─────┴──────────────────┴──────────────────┴─────→ Token节省量
```

---

## 2. 三层压缩机制详解

### 2.1 Layer 1: micro_compact（微观压缩）

**触发时机**：每次 LLM 调用前自动执行（无感知）

**工作原理**：把超过 3 轮之前的工具执行结果替换为占位符

```python
# agents/s06_context_compact.py

KEEP_RECENT = 3  # 保留最近 3 个工具结果

def micro_compact(messages: list) -> list:
    # 1. 收集所有 tool_result 条目
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
    
    # 2. 如果结果数 <= 3，不做处理
    if len(tool_results) <= KEEP_RECENT:
        return messages
    
    # 3. 找到每个 tool_result 对应的工具名称
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
    
    # 4. 清理旧结果（只保留最后 KEEP_RECENT 个）
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, result in to_clear:
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            tool_id = result.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            result["content"] = f"[Previous: used {tool_name}]"
    
    return messages
```

**压缩效果示例**：

```
压缩前：
├── tool_result: read_file → "import os\nimport sys\n..." (5000字符)
├── tool_result: bash → "total 24\ndrwxr-xr-x  5 user..." (2000字符)
├── tool_result: read_file → "class MyClass:\n..." (8000字符)
├── tool_result: grep_search → "file1.py:10: import\n..." (3000字符)  ← 保留
├── tool_result: bash → "BUILD SUCCESS" (100字符)                    ← 保留
└── tool_result: read_file → "def main():\n..." (4000字符)           ← 保留

压缩后：
├── tool_result: "[Previous: used read_file]"
├── tool_result: "[Previous: used bash]"
├── tool_result: "[Previous: used read_file]"
├── tool_result: grep_search → "file1.py:10: import\n..." (保留原内容)
├── tool_result: bash → "BUILD SUCCESS" (保留原内容)
└── tool_result: read_file → "def main():\n..." (保留原内容)
```

**为什么只保留最近 3 个？**

Agent 的工作模式是：基于前几步的结果做决策。3 轮之前的工具结果大概率已经被 Agent "消化"过了（体现在后续的思考和输出中），原始内容不再需要。但工具调用本身的**存在性**需要保留，所以用 `[Previous: used xxx]` 占位。

**类比 Java**：就像 `WeakHashMap` 的弱引用机制——当内存紧张时，允许 GC 回收不再强引用的对象，但你知道那个 key 曾经存在过。

---

### 2.2 Layer 2: auto_compact（自动压缩）

**触发条件**：Token 估计值超过阈值（默认 50,000）

**工作原理**：
1. 把完整对话保存到磁盘（Transcript）
2. 让 LLM 生成对话摘要
3. 用摘要替换所有消息

```python
THRESHOLD = 50000  # Token 阈值
TRANSCRIPT_DIR = WORKDIR / ".transcripts"

def estimate_tokens(messages: list) -> int:
    """粗略的 Token 估计：~4 字符 = 1 token"""
    return len(str(messages)) // 4

def auto_compact(messages: list) -> list:
    # 1. 保存完整对话到磁盘（保证信息不丢失）
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    
    # 2. 截取对话内容（防止摘要请求本身太长）
    conversation_text = json.dumps(messages, default=str)[:80000]
    
    # 3. 调用 LLM 生成摘要
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        max_tokens=2000,
    )
    summary = response.content[0].text
    
    # 4. 用摘要替换所有消息
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing."},
    ]
```

**压缩后的消息格式**：

```
User: [Conversation compressed. Transcript: .transcripts/transcript_1710234567.jsonl]

用户正在调试一个 Python Web 应用的登录模块。已经完成：
1) 定位到 auth/login.py 中的 validate_token() 函数有 bug
2) 读取了相关的 3 个文件：login.py, token_utils.py, config.py
3) 发现问题：token 过期检查使用了错误的时区

当前状态：
- 已确认 bug 根因
- 待修复 validate_token() 函数

��键决策：
- 使用 pytz 库处理时区问题
- 保持向后兼容，不改变 API 签名

Assistant: Understood. I have the context from the summary. Continuing.
```

**零信息丢失设计**：Transcript 文件保存了完整的对话历史，即使内存中只剩下摘要，也可以通过 Transcript 恢复完整上下文（虽然通常不需要）。

---

### 2.3 Layer 3: manual compact（手动压缩）

**触发方式**：通过 `compact` 工具调用，或用户输入 `/compact` 命令

**使用场景**：
- Agent 主动判断需要压缩（如即将执行大规模文件读取）
- 用户觉得上下文太长，主动要求压缩
- 在关键节点做"存档点"

```python
# 工具定义
TOOLS = [
    # ... 其他工具 ...
    {"name": "compact", 
     "description": "Trigger manual conversation compression.",
     "input_schema": {
         "type": "object", 
         "properties": {
             "focus": {"type": "string", "description": "What to preserve in the summary"}
         }
     }},
]

# Agent 循环中的处理
def agent_loop(messages: list):
    while True:
        # ... LLM 调用和工具执行 ...
        
        manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    manual_compact = True
                    output = "Compressing..."
                # ... 处理其他工具 ...
        
        # Layer 3: 如果触发了手动压缩
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
```

**CLI 命令支持**：

```python
# agents/s_full.py
if query.strip() == "/compact":
    if history:
        print("[manual compact via /compact]")
        history[:] = auto_compact(history)
```

---

### 2.4 三层压缩的整合流程

```python
def agent_loop(messages: list):
    while True:
        # ╔═══════════════════════════════════════════════════════╗
        # ║ Layer 1: micro_compact                                 ║
        # ║ 每轮自动执行，替换 3 轮前的工具结果为占位符              ║
        # ╚═══════════════════════════════════════════════════════╝
        micro_compact(messages)
        
        # ╔═══════════════════════════════════════════════════════╗
        # ║ Layer 2: auto_compact                                  ║
        # ║ 当 token 估计值超过阈值时，保存 transcript + LLM 摘要   ║
        # ╚═══════════════════════════════════════════════════════╝
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        
        # 调用 LLM
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        
        # 如果没有工具调用，对话结束
        if response.stop_reason != "tool_use":
            return
        
        # 处理工具调用
        results = []
        manual_compact = False
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    manual_compact = True
                    output = "Compressing..."
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    output = handler(**block.input) if handler else f"Unknown tool"
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        
        messages.append({"role": "user", "content": results})
        
        # ╔═══════════════════════════════════════════════════════╗
        # ║ Layer 3: manual compact                                ║
        # ║ Agent 主动调用 compact 工具时触发                       ║
        # ╚═══════════════════════════════════════════════════════╝
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
```

**可视化流程图**：

```
每轮 Agent 循环:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────┐                                        │
│  │ micro_compact() │  ← 每轮必执行，清理旧工具结果            │
│  └────────┬────────┘                                        │
│           ↓                                                 │
│  ┌─────────────────┐    ┌──────────────────┐                │
│  │ tokens > 50000? │───→│ auto_compact()   │  ← 阈值触发     │
│  └────────┬────────┘ yes└──────────────────┘                │
│           │ no                                              │
│           ↓                                                 │
│  ┌─────────────────┐                                        │
│  │ LLM 调用        │                                        │
│  └────────┬────────┘                                        │
│           ↓                                                 │
│  ┌─────────────────┐                                        │
│  │ 工具执行        │                                        │
│  └────────┬────────┘                                        │
│           ↓                                                 │
│  ┌─────────────────┐    ┌──────────────────┐                │
│  │ compact 工具?   │───→│ auto_compact()   │  ← 手动触发     │
│  └────────┬────────┘ yes└──────────────────┘                │
│           │ no                                              │
│           ↓                                                 │
│       下一轮循环                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 原始 Claude Code 的 8 段式压缩（AU2）

通过逆向工程分析，我们发现真实的 Claude Code 使用了更精细的**8 段式结构化压缩**（`AU2` 函数）。这比教学实现中的简单摘要更加系统化。

### 3.1 8 段结构

```javascript
// 混淆前的函数名: AU2
// 功能: 生成 8 段式结构化摘要 Prompt

function AU2(conversationHistory) {
  const compressionPrompt = `
## Conversation Summary Request

Please provide a comprehensive summary of this conversation in the following 8 sections:

### 1. Primary Request and Intent
${extractPrimaryRequest(conversationHistory)}
// 用户最初的请求是什么？核心意图是什么？

### 2. Key Technical Concepts  
${extractTechnicalConcepts(conversationHistory)}
// 涉及了哪些技术概念、框架、API？

### 3. Files and Code Sections
${extractFileReferences(conversationHistory)}
// 读取/修改了哪些文件？关键代码片段是什么？

### 4. Errors and Fixes
${extractErrorsAndFixes(conversationHistory)}
// 遇到了哪些错误？如何解决的？

### 5. Problem Solving
${extractProblemSolving(conversationHistory)}
// 解决问题的思路和方法是什么？

### 6. All User Messages
${extractUserMessages(conversationHistory)}
// 用户的所有消息摘要（保证不丢失用户意图）

### 7. Pending Tasks
${extractPendingTasks(conversationHistory)}
// 还有哪些待完成的任务？

### 8. Current Work
${extractCurrentWork(conversationHistory)}
// 当前正在做什么？下一步是什么？
`;
  
  return compressionPrompt;
}
```

### 3.2 各段含义详解

| 段落 | 作用 | 为什么重要 |
|---|---|---|
| **1. Primary Request** | 记住用户最初的目标 | 防止长对话中偏离主题 |
| **2. Technical Concepts** | 保留技术上下文 | Agent 需要知道在什么技术栈下工作 |
| **3. Files and Code** | 记录文件引用 | 代码修改需要知道改了哪些文件 |
| **4. Errors and Fixes** | 避免重复犯错 | 不要再次引入已修复的 bug |
| **5. Problem Solving** | 保留解决思路 | 维持解决问题的连贯性 |
| **6. User Messages** | 完整的用户意图 | 用户说的每句话都可能包含重要约束 |
| **7. Pending Tasks** | 待办事项清单 | 确保不遗漏任务 |
| **8. Current Work** | 当前进度 | 无缝继续工作 |

### 3.3 压缩效果

根据逆向工程分析：
- **长度减少**：平均 70-80%
- **关键信息保留**：95% 以上
- **触发阈值**：92% 上下文窗口使用率

---

## 4. Token 管理和窗口控制

### 4.1 Token 估计方法

**教学实现（简化版）**：

```python
def estimate_tokens(messages: list) -> int:
    """粗略估计：~4 字符 = 1 token"""
    return len(str(messages)) // 4
```

**原始 Claude Code（精确版）**：

```javascript
// 混淆函数名: zY5
// 功能: 精确的 Token 统计（包含缓存 token）

function zY5(messages) {
  return {
    input_tokens: number,           // 输入 token 数
    cache_creation_input_tokens: number,  // 缓存创建消耗的 token
    cache_read_input_tokens: number,      // 缓存读取消耗的 token
    output_tokens: number,          // 输出 token 数
    total: sum(all_tokens)          // 总计
  };
}
```

**为什么需要缓存感知？**

Claude API 支持 Prompt Caching，相同的上下文前缀可以被缓存。Token 统计需要区分：
- 新输入的 token（全价计费）
- 缓存创建的 token（有额外成本）
- 缓存读取的 token（便宜很多）

### 4.2 渐进式预警系统

原始 Claude Code 不是一刀切地在 92% 触发压缩，而是有**多级预警**：

```javascript
// 上下文使用率预警级别
const CONTEXT_WARNING_LEVELS = {
  NORMAL:    0.75,  // 75% - 初始警告（内部日志）
  WARNING:   0.85,  // 85% - 紧急警告（可能显示给用户）
  CRITICAL:  0.92,  // 92% - 自动触发压缩（h11 常量）
  EMERGENCY: 0.95   // 95% - 紧急截断（最后手段）
};

function checkContextUsage(usage) {
  const ratio = usage.total / contextLimit;
  
  if (ratio > CONTEXT_WARNING_LEVELS.CRITICAL) {
    triggerAutoCompaction();  // wU2 函数
  } else if (ratio > CONTEXT_WARNING_LEVELS.WARNING) {
    showContextWarning();     // 提醒用户
  }
}
```

**类比 Java**：就像 JVM 的 GC 日志级别——`-XX:+PrintGC` 只打印 Minor GC，`-XX:+PrintGCDetails` 打印详细信息，`-XX:GCTimeRatio=19` 控制 GC 时间占比。不同级别对应不同的响应策略。

### 4.3 配置参数对照

| 参数 | 教学实现值 | 原始 Claude Code 值 | 说明 |
|---|---|---|---|
| `THRESHOLD` | 50,000 tokens | 92% 上下文窗口 | 触发 auto_compact 的阈值 |
| `KEEP_RECENT` | 3 | ~3 | micro_compact 保留的工具结果数 |
| `max_tokens (摘要)` | 2,000 | ~2,000 | LLM 生成摘要的最大长度 |
| `截取长度` | 80,000 字符 | 动态计算 | 喂给摘要 LLM 的最大输入 |

---

## 5. Transcript 持久化与恢复

### 5.1 存储格式

**位置**：`.transcripts/` 目录

**文件命名**：`transcript_{unix_timestamp}.jsonl`

**格式**：JSONL（每行一个 JSON 对象）

```jsonl
{"role": "user", "content": "帮我调试登录功能"}
{"role": "assistant", "content": [{"type": "text", "text": "好的，让我先看一下..."}]}
{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "abc123", "content": "文件内容..."}]}
...
```

**为什么用 JSONL 而不是单个 JSON 文件？**

- **流式写入**：每条消息追加一行，不需要读取-修改-写入整个文件
- **容错性**：即使进程崩溃，已写入的行不会丢失
- **大文件友好**：可以逐行读取，不需要一次性加载到内存

**类比 Java**：就像 Kafka 的日志段（Log Segment）——追加写入，顺序读取，高吞吐。

### 5.2 恢复机制

虽然压缩后的消息中包含了 Transcript 路径，但当前实现并没有自动恢复功能。这是一个**手动恢复**的设计：

```python
# 理论上可以这样恢复（需要自行实现）
def recover_from_transcript(transcript_path: str) -> list:
    messages = []
    with open(transcript_path, "r") as f:
        for line in f:
            messages.append(json.loads(line))
    return messages
```

**设计哲学**：Transcript 主要是**审计日志**，而不是实时恢复点。压缩后的摘要已经包含了继续工作所需的所有信息，Transcript 是用来事后查看"到底发生了什么"的。

---

## 6. 与 DeerFlow 的设计对比

### 6.1 设计哲学差异

| 维度 | Claude Code | DeerFlow |
|---|---|---|
| **核心问题** | 单会话内的上下文管理 | 跨会话的用户档案 + 单会话上下文 |
| **压缩触发** | 阈值触发 + 手动触发 | 阈值触发（before_model） |
| **压缩层级** | 三层递进（轻 → 重） | 单层摘要 |
| **信息持久化** | Transcript 文件（可选） | memory.json（必选） |
| **摘要策略** | 8 段式结构化 | 简单摘要 |
| **用户画像** | 无（或依赖 CLAUDE.md） | 有（事实提取 + 置信度） |

### 6.2 为什么 Claude Code 不需要长期记忆？

1. **定位不同**：Claude Code 是 CLI 工具，主要场景是短期的开发任务，不是长期的对话助手
2. **项目上下文**：通过 CLAUDE.md 文件实现项目级别的"长期记忆"，而不是用户级别
3. **简化设计**：减少复杂度，专注于做好上下文压缩这一件事

### 6.3 可借鉴的设计

**从 Claude Code 借鉴到 DeerFlow**：
- 三层递进式压缩（避免一刀切）
- 8 段式结构化摘要（保留更多关键信息）
- Transcript 持久化（审计和恢复）

**从 DeerFlow 借鉴到 Claude Code**：
- 长期用户档案（个性化体验）
- 防抖队列（减少不必要的 LLM 调用）
- 置信度过滤（提高记忆质量）

---

## 7. 设计模式总结

| 模式 | 体现位置 | 说明 |
|---|---|---|
| **分层压缩模式** | micro_compact → auto_compact → manual | 从轻量到激进，按需启动，避免过度压缩 |
| **零丢失设计** | Transcript 持久化 | 完整历史保存到磁盘，压缩只是"移出活跃上下文" |
| **渐进式预警** | 75% → 85% → 92% → 95% | 多级阈值，不同响应策略 |
| **结构化摘要** | 8 段式 AU2 Prompt | 比自由格式摘要保留更多关键信息 |
| **惰性计算** | estimate_tokens 按需调用 | Token 计算只在需要时执行 |
| **占位符替换** | micro_compact 的 `[Previous: used xxx]` | 保留存在性，释放内容空间 |
| **追加写入** | JSONL 格式 Transcript | 流式写入，容错性好，适合大文件 |

---

## 8. 实战：运行示例

### 8.1 启动 Agent

```bash
cd learn-claude-code
python agents/s06_context_compact.py
```

### 8.2 观察 micro_compact

```
> Read every Python file in the agents/ directory one by one

# 观察输出：
> read_file: import anthropic...  (第1个文件，完整内容)
> read_file: import anthropic...  (第2个文件，完整内容)
> read_file: import anthropic...  (第3个文件，完整内容)
> read_file: import anthropic...  (第4个文件，完整内容)
# 此时第1个文件的 tool_result 已被替换为 "[Previous: used read_file]"
```

### 8.3 触发 auto_compact

```
> Keep reading files until compression triggers automatically

# 持续读取文件...
[auto_compact triggered]
[transcript saved: .transcripts/transcript_1710234567.jsonl]

# 所有消息被替换为摘要
```

### 8.4 手动压缩

```
> Use the compact tool to manually compress the conversation

# 或者直接输入命令：
/compact

[manual compact]
[transcript saved: .transcripts/transcript_1710234890.jsonl]
```

---

## 9. 常见问题

### Q1: 为什么 THRESHOLD 设置为 50000 而不是更高？

这是一个保守的阈值。Claude 3.5 Sonnet 的上下文窗口是 200k tokens，但：
- 需要为输出预留空间（最多 8000 tokens）
- 需要为系统提示预留空间
- Token 估计是近似值，有误差
- 过高的上下文使用率会影响模型性能

50000 tokens 大约是 25%，提供了充足的安全边际。生产环境中，原始 Claude Code 使用 92%（约 184k tokens）。

### Q2: micro_compact 为什么保留 3 个而不是更多？

基于经验观察：
- Agent 的决策主要依赖最近 2-3 轮的工具结果
- 更早的结果通常已经被"消化"到后续的思考中
- 3 是一个平衡点：既能保留足够上下文，又能有效减少 token 消耗

### Q3: Transcript 文件会无限增长吗？

当前实现没有自动清理机制。在生产环境中，应该添加：
- 按时间清理（如保留最近 7 天）
- 按数量清理（如保留最近 100 个）
- 按大小清理（如总大小不超过 1GB）

### Q4: 压缩后 Agent 会"忘记"之前的工作吗？

理论上不会。8 段式摘要设计专门考虑了这个问题：
- **Pending Tasks** 段落保留待办事项
- **Current Work** 段落记录当前进度
- **Files and Code** 段落记录文件引用

但如果摘要生成不完整，可能会丢失一些上下文。这是 LLM 摘要的固有风险。

---

## 10. 参考文件

| 文件 | 说明 |
|---|---|
| `agents/s06_context_compact.py` | 教学实现：三层压缩完整代码 |
| `agents/s_full.py` | 完整 Agent 参考实现 |
| `docs/zh/s06-context-compact.md` | 中文文档（简版） |
| `analysis_claude_code/work_doc_for_this/system_design_analysis_stage1.md` | 逆向工程分析文档 |
| `analysis_claude_code/.../H2_CONTEXT_MEMORY.md` | 上下文管理深度分析 |