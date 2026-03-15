# MCP (Model Context Protocol) 详细讲解

## 目录
- [什么是 MCP](#什么是-mcp)
- [核心概念](#核心概念)
- [MCP 的核心价值](#mcp-的核心价值)
- [通信协议与方式](#通信协议与方式)
- [NPM vs NPX](#npm-vs-npx)
- [完整流程详解](#完整流程详解)
- [实际案例](#实际案例)

---

## 什么是 MCP

MCP (Model Context Protocol) 是一个**标准化协议**，让 AI 应用能够以统一的方式连接各种工具和数据源。

**类比理解**：
- 就像浏览器的**插件系统**（Chrome Extensions）
- 或者 VS Code 的**插件市场**
- 让 AI 应用也有了"插件生态"

---

## 核心概念

### 1. Client-Server 架构

```
┌─────────────┐
│   AI Model  │  (Claude, GPT 等)
└──────┬──────┘
       │
┌──────▼──────────┐
│   MCP Client    │  ← 你的 AI 应用（Claude Desktop, Cursor 等）
│   (聚合层)      │     负责：启动 Server、聚合 tools、路由请求
└─────┬───────────┘
      │
      ├─────────┬─────────┬─────────┐
      │         │         │         │
┌─────▼───┐ ┌──▼────┐ ┌──▼────┐ ┌──▼────┐
│ Server1 │ │Server2│ │Server3│ │Server4│
│ (文件)  │ │(数据库)│ │(Git)  │ │(PDF)  │
└─────────┘ └───────┘ └───────┘ └───────┘
```

**关键点**：
- **一个 MCP Client** 可以连接**多个 MCP Server**
- 每个 Server 提供一组相关的工具（tools）
- Client 负责聚合所有 tools 并提供给 AI 模型

### 2. MCP Server 的两种部署模式

#### 模式 1：本地进程模式（最常见）

```
┌─────────────────────────────────────┐
│      你的机器 (localhost)            │
│                                     │
│  ┌──────────────┐                  │
│  │ MCP Client   │  (主进程)        │
│  │ 比如：Claude │                  │
│  └───┬──────────┘                  │
│      │                             │
│      │ fork 子进程 + stdio 通信     │
│      │                             │
│  ┌───▼──────────┐  ┌─────────────┐ │
│  │ Server1      │  │ Server2     │ │
│  │ (子进程)     │  │ (子进程)    │ │
│  └──────────────┘  └─────────────┘ │
└─────────────────────────────────────┘
```

**特点**：
- Server 是本地可执行程序（Node.js 脚本、Python 程序、二进制文件等）
- Client **启动和管理** Server 进程
- 通过 **stdio**（标准输入输出）通信
- Server 生命周期 = Client 生命周期

#### 模式 2：远程服务模式（较少见）

```
┌──────────────┐           网络           ┌──────────────┐
│ MCP Client   │ ◄────── SSE/HTTP ──────► │ MCP Server   │
│ (本地)       │                          │ (云端)       │
└──────────────┘                          └──────────────┘
```

**特点**：
- Server 是独立部署的远程服务
- Client 通过 **SSE** (Server-Sent Events) 连接
- Server 生命周期**独立**于 Client

---

## MCP 的核心价值

### 问题：没有 MCP 之前

如果你想给 AI 应用添加功能，需要：

```javascript
// 你的 AI 应用代码
const tools = [
  {
    name: "read_file",
    handler: (path) => {
      // 自己写文件读取逻辑
      return fs.readFileSync(path);
    }
  },
  {
    name: "generate_pdf",
    handler: (content) => {
      // 自己写 PDF 生成逻辑（很复杂！）
      // 需要学习 PDF 库、处理各种边界情况...
    }
  },
  {
    name: "query_database",
    handler: (sql) => {
      // 自己写数据库连接逻辑
    }
  }
];
```

**痛点**：
- ❌ 每个功能都要自己写代码
- ❌ 第三方工具无法"插入"
- ❌ 想加新功能 = 改代码 + 重新部署
- ❌ 无法复用社区的工作

### 解决方案：有了 MCP 之后

```json
// 只需要配置文件，不用写代码！
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"]
    },
    "pdf": {
      "command": "npx",
      "args": ["-y", "awesome-pdf-mcp-server"]
    },
    "database": {
      "command": "npx",
      "args": ["-y", "postgres-mcp-server"]
    }
  }
}
```

**优势**：
- ✅ **即插即用**：只需修改配置文件
- ✅ **社区生态**：任何人都可以发布 MCP Server 到 npm
- ✅ **零代码集成**：用户不需要懂编程
- ✅ **标准化**：所有 Server 遵循同一协议

### 类比：插件市场

| 传统方式 | MCP 方式 |
|---------|---------|
| 自己造轮子 | 从"插件市场"安装 |
| 改代码 + 重新部署 | 改配置文件 + 重启应用 |
| 功能固定 | 功能可扩展 |
| 无法复用社区工作 | 复用社区 MCP Server |

**实际例子**：

假设有人做了一个 "Notion MCP Server" 并发布到 npm：

```bash
npm publish @someone/mcp-notion-server
```

**所有支持 MCP 的应用（Claude Desktop、Cursor、你的自定义应用）都能立即使用**：

```json
{
  "mcpServers": {
    "notion": {
      "command": "npx",
      "args": ["-y", "@someone/mcp-notion-server"],
      "env": {
        "NOTION_API_KEY": "your-api-key"
      }
    }
  }
}
```

不需要：
- ❌ 等官方支持 Notion
- ❌ 自己写 Notion API 集成代码
- ❌ 修改应用源码

---

## 通信协议与方式

### 1. 协议：JSON-RPC 2.0

MCP 使用 **JSON-RPC 2.0** 协议进行通信。

**什么是 JSON-RPC？**
- 一种轻量级的远程过程调用协议
- 使用 JSON 格式传输数据
- 简单、易于实现

**请求示例**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

**响应示例**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "read_file",
        "description": "读取文件内容",
        "inputSchema": {
          "type": "object",
          "properties": {
            "path": { "type": "string" }
          }
        }
      }
    ]
  }
}
```

### 2. 传输方式

#### 方式 1：stdio（标准输入输出）- 本地模式

```
┌─────────────────┐
│   MCP Client    │
│                 │
│  stdin  ──────┐ │
│  stdout ◄─────┤ │
└───────────────┼─┘
                │ JSON-RPC 消息
┌───────────────▼─┐
│   MCP Server    │
│                 │
│  stdin  ◄─────  │ (接收请求)
│  stdout ──────► │ (返回响应)
└─────────────────┘
```

**工作原理**：
- Client 和 Server 在**同一台机器**上，但是**不同进程**
- Client 通过 Server 的 **stdin** 发送 JSON-RPC 请求
- Server 通过 **stdout** 返回 JSON-RPC 响应
- 就像两个程序通过"管道"对话

**优点**：
- 简单、高效
- 不需要网络配置
- 安全（本地通信）

#### 方式 2：SSE（Server-Sent Events）- 远程模式

```
┌─────────────────┐      HTTP/SSE      ┌─────────────────┐
│   MCP Client    │ ◄─────────────────► │   MCP Server    │
│   (本地)        │                     │   (远程)        │
└─────────────────┘                     └─────────────────┘
```

**工作原理**：
- Server 是独立的 Web 服务
- Client 通过 HTTP 连接
- 使用 SSE 保持长连接

**使用场景**：
- Server 部署在云端
- 多个 Client 共享同一个 Server
- 需要跨网络访问

### 3. 对比总结

| 特性 | stdio (本地) | SSE (远程) |
|------|-------------|-----------|
| 部署位置 | 同一台机器 | 可以跨网络 |
| 启动方式 | Client 启动 Server 进程 | Server 独立运行 |
| 通信方式 | 标准输入输出 | HTTP/SSE |
| 生命周期 | 跟随 Client | 独立 |
| 使用场景 | 本地工具（文件、数据库） | 远程服务（API、云服务） |
| 常见程度 | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## NPM vs NPX

### NPM（Node Package Manager）

**作用**：包管理器，用于安装和管理 Node.js 包

```bash
# 全局安装
npm install -g @modelcontextprotocol/server-filesystem

# 本地安装
npm install @modelcontextprotocol/server-filesystem
```

**特点**：
- 下载包到本地磁盘（`node_modules` 或全局目录）
- 需要手动管理版本和更新
- 占用磁盘空间
- 安装后可以直接运行

**使用场景**：
- 项目依赖
- 需要长期使用的工具
- 需要离线使用

### NPX（Node Package Execute）

**作用**：临时执行 npm 包，不需要安装

```bash
# 临时执行（推荐用于 MCP）
npx -y @modelcontextprotocol/server-filesystem
```

**特点**：
- **不安装**到本地（或只缓存临时文件）
- 自动下载最新版本
- 执行完后可能保留在缓存中（下次更快）
- `-y` 参数：自动确认，不询问用户

**工作流程**：
1. 检查本地是否有这个包
2. 没有则从 npm registry 下载到临时缓存
3. 执行包的入口文件
4. 执行完毕（进程可能继续运行）

**使用场景**：
- 一次性执行的工具
- 总是想用最新版本
- 不想污染全局环境
- **MCP Server（推荐）**

### 为什么 MCP 推荐用 NPX？

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",  // 👈 用 npx，不是 npm
      "args": ["-y", "@modelcontextprotocol/server-filesystem"]
    }
  }
}
```

**原因**：
1. **自动更新**：每次启动都用最新版本
2. **无需安装**：用户不需要手动 `npm install`
3. **简单配置**：只需要包名，不需要路径
4. **跨平台**：不用担心全局安装路径问题

### 对比总结

| 特性 | NPM | NPX |
|------|-----|-----|
| 安装到本地 | ✅ 是 | ❌ 否（或临时缓存） |
| 占用空间 | 较多 | 较少 |
| 版本管理 | 手动更新 | 自动最新 |
| 使用前提 | 需要先安装 | 直接执行 |
| MCP 推荐 | ❌ | ✅ |

---

## 完整流程详解

### 阶段 1：用户配置（手动操作）

用户编辑配置文件，添加 MCP Server：

**Claude Desktop 配置文件位置**：
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Cursor 配置**：
- 在设置界面中配置

**配置示例**：
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/you/Documents"]
    },
    "database": {
      "command": "npx",
      "args": ["-y", "postgres-mcp-server"],
      "env": {
        "DATABASE_URL": "postgresql://localhost/mydb"
      }
    }
  }
}
```

**配置说明**：
- `command`: 要执行的命令（通常是 `npx`）
- `args`: 传给命令的参数数组
- `env`: 环境变量（可选，用于传递 API Key 等）

### 阶段 2：应用启动（自动）

用户启动 Claude Desktop 或 Cursor。

**MCP Client 做的事情**：

1. **读取配置文件**
   ```
   读取 claude_desktop_config.json
   解析 mcpServers 配置
   ```

2. **启动所有 MCP Server 进程**
   ```
   对于每个 Server 配置：
     - 执行 command + args
     - 创建子进程
     - 建立 stdio 连接
   ```

3. **初始化握手**
   ```
   Client → Server: initialize 请求
   Server → Client: 返回 Server 信息
   ```

4. **获取 tools 列表**
   ```
   Client → Server: tools/list 请求
   Server → Client: 返回所有 tools
   ```

5. **聚合所有 tools**
   ```
   合并所有 Server 的 tools
   提供给 AI 模型
   ```

### 阶段 3：Server 启动（自动）

以 `npx -y @modelcontextprotocol/server-filesystem /Users/you/Documents` 为例：

**NPX 做的事情**：

1. **检查缓存**
   ```
   查找 ~/.npm/_npx/ 缓存
   如果有缓存且版本匹配，跳到步骤 3
   ```

2. **下载包**
   ```
   从 npm registry 下载包
   解压到临时目录
   ```

3. **执行包**
   ```
   读取 package.json 的 bin 字段
   执行入口文件（通常是 Node.js 脚本）
   ```

**MCP Server 做的事情**：

1. **初始化**
   ```
   创建 Server 实例
   注册 tools
   设置请求处理器
   ```

2. **监听 stdin**
   ```
   等待来自 Client 的 JSON-RPC 请求
   ```

3. **响应请求**
   ```
   接收请求 → 处理 → 返回结果
   ```

### 阶段 4：运行时交互（自动）

**场景**：用户问 AI "帮我读取 /path/to/file.txt"

**完整流程**：

```
1. 用户输入
   └─> "帮我读取 /path/to/file.txt"

2. AI 模型分析
   └─> 决定使用 read_file tool

3. AI 返回 tool call
   └─> {
         "name": "read_file",
         "arguments": { "path": "/path/to/file.txt" }
       }

4. MCP Client 接收 tool call
   └─> 识别这个 tool 属于哪个 Server（filesystem）

5. Client 发送 JSON-RPC 请求
   └─> 通过 stdin 发送给 filesystem Server
   └─> {
         "jsonrpc": "2.0",
         "method": "tools/call",
         "params": {
           "name": "read_file",
           "arguments": { "path": "/path/to/file.txt" }
         }
       }

6. MCP Server 接收请求
   └─> 从 stdin 读取 JSON-RPC

7. Server 执行操作
   └─> 读取文件内容

8. Server 返回结果
   └─> 通过 stdout 返回 JSON-RPC 响应
   └─> {
         "jsonrpc": "2.0",
         "result": {
           "content": [
             { "type": "text", "text": "文件内容..." }
           ]
         }
       }

9. Client 接收结果
   └─> 从 stdout 读取响应

10. Client 返回给 AI 模型
    └─> 传递文件内容

11. AI 模型继续推理
    └─> 基于文件内容生成回复

12. 返回给用户
    └─> "文件内容是：..."
```

### 阶段 5：应用关闭（自动）

用户关闭 Claude Desktop 或 Cursor。

**MCP Client 做的事情**：

1. **终止所有 Server 进程**
   ```
   发送 SIGTERM 信号
   等待进程退出
   如果超时，发送 SIGKILL 强制终止
   ```

2. **清理资源**
   ```
   关闭 stdio 连接
   释放内存
   ```

---

## 实际案例

### 案例 1：文件系统 MCP Server

**配置**：
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/you/Documents"
      ]
    }
  }
}
```

**提供的 tools**：
- `read_file`: 读取文件
- `write_file`: 写入文件
- `list_directory`: 列出目录
- `create_directory`: 创建目录
- `move_file`: 移动文件
- `search_files`: 搜索文件

**使用场景**：
- "帮我读取项目的 README.md"
- "在 Documents 文件夹创建一个新文件"
- "搜索所有包含 'TODO' 的文件"

### 案例 2：数据库 MCP Server

**配置**：
```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "postgres-mcp-server"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost/mydb"
      }
    }
  }
}
```

**提供的 tools**：
- `query`: 执行 SQL 查询
- `list_tables`: 列出所有表
- `describe_table`: 查看表结构

**使用场景**：
- "查询用户表中的所有数据"
- "帮我创建一个新表"
- "统计订单表中的记录数"

### 案例 3：自定义 MCP Server

假设你想创建一个"天气查询" MCP Server：

**1. 创建 npm 包**：
```bash
mkdir weather-mcp-server
cd weather-mcp-server
npm init -y
```

**2. 实现 Server**（简化示例）：
```javascript
// index.js
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server({
  name: 'weather-server',
  version: '1.0.0'
});

// 注册 tool
server.setRequestHandler('tools/list', async () => ({
  tools: [{
    name: 'get_weather',
    description: '获取指定城市的天气',
    inputSchema: {
      type: 'object',
      properties: {
        city: { type: 'string', description: '城市名称' }
      },
      required: ['city']
    }
  }]
}));

// 处理 tool 调用
server.setRequestHandler('tools/call', async (request) => {
  if (request.params.name === 'get_weather') {
    const city = request.params.arguments.city;
    // 调用天气 API
    const weather = await fetchWeather(city);
    return {
      content: [{
        type: 'text',
        text: `${city}的天气：${weather}`
      }]
    };
  }
});

// 启动
const transport = new StdioServerTransport();
await server.connect(transport);
```

**3. 发布到 npm**：
```bash
npm publish
```

**4. 用户使用**：
```json
{
  "mcpServers": {
    "weather": {
      "command": "npx",
      "args": ["-y", "weather-mcp-server"]
    }
  }
}
```

**5. AI 可以调用**：
- "北京今天天气怎么样？"
- "上海明天会下雨吗？"

---

## 总结

### MCP 的本质

**MCP = AI 应用的插件系统**

- **标准化协议**：统一的通信方式（JSON-RPC）
- **插件生态**：任何人都可以发布 MCP Server
- **即插即用**：用户只需修改配置文件
- **社区驱动**：复用社区的工作成果

### 核心优势

1. **扩展性**：轻松添加新功能
2. **复用性**：使用社区的 MCP Server
3. **标准化**：统一的接口和协议
4. **安全性**：进程隔离，权限可控
5. **简单性**：用户无需编程知识

### 关键技术点

- **架构**：Client-Server
- **协议**：JSON-RPC 2.0
- **传输**：stdio（本地）/ SSE（远程）
- **部署**：npx 临时执行
- **生命周期**：Client 管理 Server 进程

### 适用场景

- ✅ 需要访问本地资源（文件、数据库）
- ✅ 需要集成第三方服务（API、工具）
- ✅ 需要扩展 AI 应用的能力
- ✅ 需要复用社区的工作
- ✅ 需要标准化的工具接口

---

## 参考资源

- [MCP 官方文档](https://modelcontextprotocol.io/)
- [MCP GitHub](https://github.com/modelcontextprotocol)
- [MCP Server 列表](https://github.com/modelcontextprotocol/servers)
