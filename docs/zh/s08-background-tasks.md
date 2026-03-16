# s08: Background Tasks (后台任务)

`s01 > s02 > s03 > s04 > s05 > s06 | s07 > [ s08 ] s09 > s10 > s11 > s12`

> *"慢操作丢后台, agent 继续想下一步"* -- 后台线程跑命令, 完成后注入通知。

## 问题

有些命令要跑好几分钟: `npm install`、`pytest`、`docker build`。阻塞式循环下模型只能干等。用户说 "装依赖, 顺便建个配置文件", 智能体却只能一个一个来。

## 解决方案

```
Main thread                Background thread
+-----------------+        +-----------------+
| agent loop      |        | subprocess runs |
| ...             |        | ...             |
| [LLM call] <---+------- | enqueue(result) |
|  ^drain queue   |        +-----------------+
+-----------------+

Timeline:
Agent --[spawn A]--[spawn B]--[other work]----
             |          |
             v          v
          [A runs]   [B runs]      (parallel)
             |          |
             +-- results injected before next LLM call --+
```

## 工作原理

1. BackgroundManager 用线程安全的通知队列追踪任务。

```python
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self._notification_queue = []
        self._lock = threading.Lock()
```

2. `run()` 启动守护线程, 立即返回。

```python
def run(self, command: str) -> str:
    task_id = str(uuid.uuid4())[:8]
    self.tasks[task_id] = {"status": "running", "command": command}
    thread = threading.Thread(
        target=self._execute, args=(task_id, command), daemon=True)
    thread.start()
    return f"Background task {task_id} started"
```

3. 子进程完成后, 结果进入通知队列。

```python
def _execute(self, task_id, command):
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=300)
        output = (r.stdout + r.stderr).strip()[:50000]
    except subprocess.TimeoutExpired:
        output = "Error: Timeout (300s)"
    with self._lock:
        self._notification_queue.append({
            "task_id": task_id, "result": output[:500]})
```

4. 每次 LLM 调用前排空通知队列。

```python
def agent_loop(messages: list):
    while True:
        notifs = BG.drain_notifications()
        if notifs:
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['result']}" for n in notifs)
            messages.append({"role": "user",
                "content": f"<background-results>\n{notif_text}\n"
                           f"</background-results>"})
            messages.append({"role": "assistant",
                "content": "Noted background results."})
        response = client.messages.create(...)
```

循环保持单线程。只有子进程 I/O 被并行化。

## 详细执行流程

下面是一个完整的执行流程图，展示主线程、后台线程和子进程之间的协作：

```
主线程（agent loop）                 后台线程（_execute）              子进程
        |
        |-- BG.run("git log")
        |   创建 task_id: "a1b2c3d4"
        |   tasks[id] = {status: "running"}
        |
        |-- thread.start() -----------> 线程启动
        |   立刻返回 task_id             |
        |   "Background task a1b2c3d4    |
        |    started: git log"           |
        |                               |-- subprocess.run("git log") -----> 执行 shell 命令
        |-- 继续处理其他工具调用          |                                      |
        |   (不阻塞！)                   |   (线程阻塞等待)                     |
        |                               |                                      | 运行中...
        |-- 可能调用其他 tools            |                                      |
        |   write_file, read_file...    |                                      |
        |                               |                                      |
        |                               |<-- 命令完成，返回 stdout/stderr <-----+
        |                               |
        |                               |-- 捕获输出 (最多 50000 字符)
        |                               |-- 更新 tasks[id].status = "completed"
        |                               |-- 更新 tasks[id].result = output
        |                               |
        |                               |-- with self._lock:
        |                               |     _notification_queue.append({
        |                               |       "task_id": "a1b2c3d4",
        |                               |       "status": "completed",
        |                               |       "result": output[:500]
        |                               |     })
        |                               |
        |                               |-- 线程结束 (daemon=True 自动清理)
        |
        |-- 下一轮 agent_loop 开始
        |   (可能是用户新输入，也可能是工具调用完成)
        |
        |-- notifs = BG.drain_notifications()
        |   <-------------------------- 从 _notification_queue 取出结果
        |   返回: [{"task_id": "a1b2c3d4", "status": "completed", ...}]
        |
        |-- if notifs:
        |     messages.append({
        |       "role": "user",
        |       "content": "<background-results>
        |                   [bg:a1b2c3d4] completed: ...
        |                   </background-results>"
        |     })
        |     messages.append({
        |       "role": "assistant",
        |       "content": "Noted background results."
        |     })
        |
        |-- response = client.messages.create(...)
        |   LLM 看到后台任务结果，可以基于此继续工作
        |
        v
```

### 关键时间点

1. **T0**: 用户输入 "Run git log in background, then create config.json"
2. **T1**: LLM 调用 `background_run(command="git log")`，立刻返回 task_id
3. **T2**: LLM 继续调用 `write_file(path="config.json", ...)`，不等待 git log
4. **T3**: 后台线程中 git log 执行完成，结果进入 notification_queue
5. **T4**: 下次 LLM 调用前，drain_notifications() 把结果注入到 messages
6. **T5**: LLM 看到 `<background-results>` 标签，知道 git log 已完成

## 核心设计要点

### 1. 线程 vs 进程的选择

**为什么用 Thread + subprocess？**

```python
# 架构层次：
主进程 (Python agent)
  └─ 主线程 (agent_loop)
       └─ 后台线程 (Thread, daemon=True)
            └─ 子进程 (subprocess.run, shell=True)
```

- **Thread**: 轻量级，共享内存（可以直接访问 `self.tasks` 和 `_notification_queue`）
- **subprocess**: 隔离执行环境，可以运行任意 shell 命令，不受 Python GIL 限制
- **daemon=True**: 主程序退出时自动清理后台线程，避免僵尸线程

**Python 的 GIL 影响吗？**

不影响！因为：
- 后台线程大部分时间在等待 I/O（subprocess 执行命令）
- 等待 I/O 时会释放 GIL，主线程可以继续运行
- 即使有 GIL，I/O 密集型任务依然可以并发

### 2. 线程安全的通知队列

```python
self._lock = threading.Lock()  # 保护共享数据

# 写入（后台线程）
with self._lock:
    self._notification_queue.append(result)

# 读取（主线程）
with self._lock:
    notifs = list(self._notification_queue)
    self._notification_queue.clear()
```

**为什么需要锁？**
- 主线程和后台线程同时访问 `_notification_queue`
- 没有锁可能导致数据竞争（race condition）
- Python 的 list 操作不是原子的，需要显式加锁

### 3. 结果注入机制

```python
# 关键：在 LLM 调用前注入，而不是调用后
notifs = BG.drain_notifications()
if notifs:
    # 伪造一个 user 消息 + assistant 确认
    messages.append({"role": "user", "content": "<background-results>..."})
    messages.append({"role": "assistant", "content": "Noted background results."})
# 然后才调用 LLM
response = client.messages.create(...)
```

**为什么这样设计？**
- LLM 需要在上下文中看到后台任务的结果
- 通过 `<background-results>` 标签明确标识这是系统注入的消息
- assistant 的 "Noted" 消息保持对话连贯性，避免 LLM 困惑

### 4. 主动通知的可能性

虽然当前实现是"被动拉取"（下次 LLM 调用时才检查），但可以扩展为"主动推送"：

```python
# 扩展思路：轮询 + 主动触发
def background_monitor():
    while True:
        time.sleep(5)  # 每 5 秒检查一次
        notifs = BG.drain_notifications()
        if notifs:
            # 主动触发一次 agent_loop
            trigger_agent_with_notification(notifs)
```

这样就能实现：
- 后台任务完成后，agent 主动告诉用户 "git log 执行完了，发现了 3 个新提交"
- 不需要用户再问一次才能看到结果

## 实际应用场景

### 场景 1: 并行安装依赖

```
用户: "Install npm packages and run tests"

传统方式 (阻塞):
  npm install (3 分钟) → pytest (2 分钟) = 总共 5 分钟

后台任务方式:
  background_run("npm install") → 立刻返回
  background_run("pytest") → 立刻返回
  两个任务并行执行 = 总共 3 分钟 (取最长的)
```

### 场景 2: 长时间构建 + 其他工作

```
用户: "Build Docker image and update README"

Agent 执行:
  1. background_run("docker build -t myapp .")  # 5 分钟
  2. write_file("README.md", ...)               # 立刻完成
  3. 下次循环时收到 Docker 构建结果
  4. 告诉用户: "README 已更新，Docker 镜像构建完成"
```

### 场景 3: 监控任务状态

```
用户: "Start 3 background tasks and check their status"

Agent 执行:
  1. background_run("sleep 2") → task_id: abc123
  2. background_run("sleep 4") → task_id: def456
  3. background_run("sleep 6") → task_id: ghi789
  4. check_background() → 列出所有任务状态
     abc123: [completed] sleep 2
     def456: [running] sleep 4
     ghi789: [running] sleep 6
```

## 相对 s07 的变更

| 组件           | 之前 (s07)       | 之后 (s08)                         |
|----------------|------------------|------------------------------------|
| Tools          | 8                | 6 (基础 + background_run + check)  |
| 执行方式       | 仅阻塞           | 阻塞 + 后台线程                    |
| 通知机制       | 无               | 每轮排空的队列                     |
| 并发           | 无               | 守护线程                           |

## 代码实现细节

### BackgroundManager 类结构

```python
class BackgroundManager:
    def __init__(self):
        self.tasks = {}                    # task_id -> {status, result, command}
        self._notification_queue = []      # 完成的任务通知
        self._lock = threading.Lock()      # 线程安全锁

    def run(self, command: str) -> str:
        """启动后台任务，立刻返回 task_id"""
        # 1. 生成唯一 ID
        task_id = str(uuid.uuid4())[:8]

        # 2. 记录任务状态
        self.tasks[task_id] = {
            "status": "running",
            "result": None,
            "command": command
        }

        # 3. 启动守护线程
        thread = threading.Thread(
            target=self._execute,
            args=(task_id, command),
            daemon=True  # 主程序退出时自动清理
        )
        thread.start()

        # 4. 立刻返回，不等待
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str):
        """线程目标函数：执行命令，捕获结果，推送通知"""
        try:
            # 阻塞等待子进程完成（但不阻塞主线程）
            r = subprocess.run(
                command,
                shell=True,
                cwd=WORKDIR,
                capture_output=True,
                text=True,
                timeout=300  # 5 分钟超时
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"

        # 更新任务状态
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"

        # 推送通知到队列（线程安全）
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],
                "result": (output or "(no output)")[:500],
            })

    def drain_notifications(self) -> list:
        """取出并清空所有待处理通知"""
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs
```

### agent_loop 中的注入逻辑

```python
def agent_loop(messages: list):
    while True:
        # ===== 关键步骤 1: 检查后台任务 =====
        notifs = BG.drain_notifications()
        if notifs:
            # 构造通知消息
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['result']}"
                for n in notifs
            )

            # 注入到对话历史（伪造 user 消息）
            messages.append({
                "role": "user",
                "content": f"<background-results>\n{notif_text}\n</background-results>"
            })

            # 添加 assistant 确认（保持对话连贯）
            messages.append({
                "role": "assistant",
                "content": "Noted background results."
            })

        # ===== 关键步骤 2: 调用 LLM =====
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,  # 包含了注入的后台结果
            tools=TOOLS,
            max_tokens=8000,
        )

        # ... 处理工具调用 ...
```

**注入时机的重要性**：
- 必须在 `client.messages.create()` **之前**注入
- 这样 LLM 才能在当前轮次看到后台任务结果
- 如果在之后注入，LLM 要等到下一轮才能看到

## 与传统阻塞方式的对比

### 传统阻塞方式 (s07)

```python
# 用户: "Install dependencies and create config"

# 执行流程：
1. run_bash("npm install")  # 阻塞 3 分钟
   └─ agent 等待...
   └─ LLM 无法思考其他事情
   └─ 3 分钟后返回

2. run_write("config.json", ...)  # 1 秒
   └─ 立刻完成

总耗时: 3 分 1 秒
```

### 后台任务方式 (s08)

```python
# 用户: "Install dependencies and create config"

# 执行流程：
1. background_run("npm install")  # 立刻返回 task_id
   └─ 后台线程启动
   └─ agent 继续执行

2. run_write("config.json", ...)  # 1 秒
   └─ 立刻完成

3. 下次循环时收到通知
   └─ "npm install completed"

总耗时: 1 秒 (用户感知) + 3 分钟 (后台)
```

**用户体验提升**：
- 阻塞方式：用户等待 3 分钟，什么都不能做
- 后台方式：用户 1 秒后看到 config 创建完成，3 分钟后收到安装完成通知

## 潜在的扩展方向

### 1. 进度回调

```python
def _execute_with_progress(self, task_id, command):
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )

    # 实时读取输出
    for line in process.stdout:
        # 可以推送进度通知
        self._push_progress(task_id, line.decode())

    process.wait()
```

### 2. 任务依赖

```python
# 任务 B 依赖任务 A
task_a = BG.run("npm install")
task_b = BG.run_after(task_a, "npm test")  # A 完成后才执行 B
```

### 3. 任务取消

```python
def cancel(self, task_id: str):
    """终止正在运行的后台任务"""
    task = self.tasks.get(task_id)
    if task and task["process"]:
        task["process"].terminate()
```

### 4. 主动通知（轮询模式）

```python
# 在主线程中启动监控
def start_monitor():
    def monitor_loop():
        while True:
            time.sleep(2)  # 每 2 秒检查
            notifs = BG.drain_notifications()
            if notifs:
                # 触发一次 agent_loop，主动告诉用户
                inject_and_respond(notifs)

    threading.Thread(target=monitor_loop, daemon=True).start()
```

## 试一试

```sh
cd learn-claude-code
python agents/s08_background_tasks.py
```

试试这些 prompt (英文 prompt 对 LLM 效果更好, 也可以用中文):

1. `Run "sleep 5 && echo done" in the background, then create a file while it runs`
   - 观察：文件会立刻创建，5 秒后收到 sleep 完成通知

2. `Start 3 background tasks: "sleep 2", "sleep 4", "sleep 6". Check their status.`
   - 观察：三个任务并行执行，check_background 显示实时状态

3. `Run pytest in the background and keep working on other things`
   - 观察：pytest 在后台跑，agent 可以继续处理其他请求

4. `Run "git log --oneline -10" in background, then read README.md`
   - 观察：两个操作几乎同时完成，不需要等待 git log

## 当前实现的关键限制

### 问题：agent_loop 会提前退出

这是当前实现的一个**重要限制**，需要特别注意：

```python
# 实际执行流程：
用户: "Install npm packages and run tests"

agent_loop(messages):
    1. LLM 决策 → background_run("npm install")  # 启动后台任务
    2. LLM 决策 → background_run("pytest")       # 启动后台任务
    3. LLM 返回文本: "已启动两个后台任务 abc123 和 def456"
    4. stop_reason != "tool_use" → return  # ❌ agent_loop 结束！

    # 此时两个任务还在后台运行
    # 但 agent_loop 已经退出了
    # 结果只是静静地躺在 _notification_queue 里

# 下次循环必须等用户主动询问：
用户: "做完了吗？"  # 必须有这一步！

agent_loop(messages):
    1. notifs = BG.drain_notifications()  # 这时才取出结果
    2. 注入到 messages
    3. LLM 看到结果: "npm install 完成了，pytest 也完成了"
```

**为什么会这样？**

因为 `agent_loop` 的退出条件是：

```python
if response.stop_reason != "tool_use":
    return  # 一旦 LLM 不再调用工具，循环就结束
```

一旦 LLM 只返回文本（不调用工具），循环就结束了。后台任务完成后，**没有任何机制能重新启动 agent_loop**。

### 适合和不适合的场景

**✅ 适合当前实现的场景**：

```
用户: "Run pytest in background"
Agent: "已启动后台任务 abc123"
[用户去喝咖啡，3 分钟后回来]
用户: "pytest 跑完了吗？"  # 主动询问
Agent: [检查队列] "完成了，所有测试通过"
```

**❌ 不适合的场景**：

```
用户: "Install dependencies and then deploy"
Agent: "已启动 npm install"
[3 分钟后 npm install 完成]
# 用户期望 agent 自动继续执行 deploy
# 但实际上 agent_loop 已经停止，什么都不会发生 ❌
```

### 解决方案对比

#### 方案 1: 阻塞等待所有后台任务

```python
def agent_loop(messages: list):
    while True:
        # ... LLM 调用，工具执行 ...

        # 在每次循环末尾检查
        if has_running_background_tasks():
            print("等待后台任务完成...")
            wait_for_all_background_tasks()  # 阻塞等待
            # 强制再调用一次 LLM
            continue
```

**优点**：
- 简单，不需要改架构
- 保证 LLM 能看到所有结果

**缺点**：
- 失去了"非阻塞"的优势
- 如果任务很长（10 分钟），还是要等
- 用户体验回到了阻塞模式

#### 方案 2: 轮询监控 + 主动触发

```python
class Agent:
    def __init__(self):
        self.messages = []
        self.start_background_monitor()

    def start_background_monitor(self):
        def monitor_loop():
            while True:
                time.sleep(2)  # 每 2 秒检查
                notifs = BG.drain_notifications()
                if notifs:
                    # 主动触发一次 agent_loop
                    self.on_background_complete(notifs)

        threading.Thread(target=monitor_loop, daemon=True).start()

    def on_background_complete(self, notifs):
        # 注入通知到对话历史
        notif_text = "\n".join(
            f"[bg:{n['task_id']}] {n['status']}: {n['result']}"
            for n in notifs
        )
        self.messages.append({
            "role": "user",
            "content": f"<background-results>\n{notif_text}\n</background-results>"
        })

        # 主动再跑一次 agent_loop
        self.agent_loop()

        # 推送到 UI（需要 UI 支持）
        self.ui.display_message("后台任务完成")
```

**优点**：
- 真正的主动通知
- 后台任务完成后自动继续工作
- 用户体验好

**缺点**：
- 需要重构架构（messages 要变成实例变量）
- 需要 UI 支持异步消息推送
- 复杂度显著增加

#### 方案 3: 事件驱动架构（最彻底）

```python
class Agent:
    def __init__(self):
        self.messages = []
        self.event_queue = queue.Queue()

    def run(self):
        # 主事件循环
        while True:
            event = self.event_queue.get()  # 阻塞等待事件

            if event.type == "user_input":
                self.messages.append({"role": "user", "content": event.data})
                self.agent_loop()

            elif event.type == "background_complete":
                # 后台任务完成，主动触发
                self.inject_notification(event.data)
                self.agent_loop()  # 再跑一次！

    def agent_loop(self):
        # 不再是 while True，只跑一轮
        notifs = BG.drain_notifications()
        # ... 注入，调用 LLM ...

# 后台任务完成时推送事件
def _execute(self, task_id, command):
    # ... 执行命令 ...
    agent.event_queue.put(Event("background_complete", result))
```

**优点**：
- 架构清晰，易于扩展
- 支持多种事件类型（用户输入、后台完成、定时任务等）
- 真正的异步响应

**缺点**：
- 需要完全重写 agent 架构
- 学习曲线陡峭
- 需要配套的 UI 和消息系统

### 生产级实现的考虑

像 Claude Code 这样的生产级工具，可能采用类似方案 2 或方案 3 的混合架构：

```python
# 伪代码：Claude Code 可能的实现
class ClaudeCodeSession:
    def __init__(self):
        self.messages = []
        self.background_monitor = BackgroundMonitor(self)
        self.ui = UIChannel()

    def handle_user_input(self, text):
        self.messages.append({"role": "user", "content": text})
        self.agent_loop()

    def on_background_complete(self, notifs):
        # 监控线程回调
        self.inject_notifications(notifs)

        # 主动再跑一次 agent_loop
        self.agent_loop()

        # 推送到 UI（用户看到实时通知）
        self.ui.push_notification("后台任务完成", notifs)

class BackgroundMonitor:
    def __init__(self, session):
        self.session = session
        threading.Thread(target=self.monitor, daemon=True).start()

    def monitor(self):
        while True:
            time.sleep(2)
            notifs = BG.drain_notifications()
            if notifs:
                # 回调到主 session
                self.session.on_background_complete(notifs)
```

### 当前实现的定位

当前的 s08 实现是一个**教学示例**，它展示了：

1. ✅ 如何用线程实现非阻塞执行
2. ✅ 如何用队列传递结果
3. ✅ 如何在下次循环时注入结果
4. ✅ 线程安全的基本实践

但它**不是一个完整的生产级方案**，因为：

1. ❌ 依赖用户主动询问才能看到结果
2. ❌ 没有主动通知机制
3. ❌ 不支持任务完成后自动继续工作
4. ❌ 没有 UI 集成

**适用场景**：
- 学习后台任务的基本原理
- 用户明确知道任务在跑，会主动询问结果
- 简单的脚本工具，不需要复杂的交互

**不适用场景**：
- 需要任务完成后自动继续工作
- 需要实时通知用户
- 复杂的多步骤工作流

## 常见问题

**Q: 后台任务失败了怎么办？**
A: 异常会被捕获并记录在 `result` 中，通过 notification_queue 通知 agent。

**Q: 如果后台任务太多会怎样？**
A: 当前实现没有限制，可以添加最大并发数：
```python
if len([t for t in self.tasks.values() if t["status"] == "running"]) >= 5:
    return "Error: Too many background tasks"
```

**Q: daemon=True 会导致任务被强制终止吗？**
A: 是的。如果主程序退出，后台线程会被强制终止。可以改为 daemon=False 并在退出前等待所有任务完成。

**Q: 为什么不用 asyncio？**
A: asyncio 更适合纯 Python 的异步 I/O。这里需要执行外部命令（subprocess），Thread + subprocess 更直观且兼容性更好。

**Q: 为什么后台任务完成后 agent 不会自动继续工作？**
A: 因为 `agent_loop` 在 LLM 不再调用工具时就退出了。要实现自动继续，需要采用方案 2 或方案 3 的架构，这超出了当前教学示例的范围。

**Q: 如何改造成生产级实现？**
A: 需要：
1. 将 `messages` 提升为类成员变量
2. 实现后台监控线程 + 回调机制
3. 集成 UI 的异步消息推送
4. 添加事件队列和状态管理
5. 处理并发、错误恢复、任务取消等边界情况
