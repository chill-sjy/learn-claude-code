# Skills 深度解析：以 PPTX 为例

> 本文档深入讲解 Claude Code 的 Skills 机制，通过 Anthropic 官方的 pptx skill 实例，展示 skills 如何在真实场景中工作。

## 目录

- [核心概念](#核心概念)
- [Skill 的完整结构](#skill-的完整结构)
- [PPTX Skill 实战解析](#pptx-skill-实战解析)
- [工作流程详解](#工作流程详解)
- [设计哲学](#设计哲学)

---

## 核心概念

### Skill 不是 Tool

这是最容易混淆的地方：

```
❌ 错误理解：Skill = 一个新的 tool
✅ 正确理解：Skill = 知识库 + 辅助脚本，通过现有 tools 使用
```

**Agent 的 Tools（永远只有这几个）**：
- `bash` - 执行命令
- `read_file` - 读文件
- `write_file` - 写文件
- `edit_file` - 编辑文件
- `load_skill` - 加载技能知识

**Skill 提供的是**：
- 📚 领域知识（SKILL.md）
- 🛠️ 辅助脚本（scripts/）
- 📖 参考资料（references/）

### 两层加载机制

```
┌─────────────────────────────────────────┐
│ System Prompt (Layer 1 - 始终存在)      │
│                                         │
│ Skills available:                       │
│   - pptx: Process PowerPoint files      │  ← 只有名称和简短描述
│   - pdf: Process PDF files              │     (~100 tokens/skill)
│   - code-review: Review code            │
└─────────────────────────────────────────┘

当模型调用 load_skill("pptx"):

┌─────────────────────────────────────────┐
│ tool_result (Layer 2 - 按需加载)        │
│                                         │
│ <skill name="pptx">                     │
│   # PPTX Skill                          │  ← 完整的知识内容
│   ## Reading Content                    │     (~2000+ tokens)
│   ```bash                               │
│   python scripts/thumbnail.py file.pptx │
│   ```                                   │
│   ...                                   │
│ </skill>                                │
└─────────────────────────────────────────┘
```

**为什么这样设计？**
- 如果把所有 skill 内容都放进 system prompt：10 个 skills × 2000 tokens = 20,000 tokens 浪费
- 按需加载：只有用到时才加载，节省上下文

---

## Skill 的完整结构

以 Anthropic 官方的 pptx skill 为例：

```
skills/pptx/
├── SKILL.md              # 核心：技能知识库
├── editing.md            # 补充文档：编辑工作流详解
├── pptxgenjs.md          # 补充文档：从零创建 PPT
├── LICENSE.txt           # 许可证
└── scripts/              # 辅助脚本目录
    ├── __init__.py
    ├── add_slide.py      # 添加幻灯片
    ├── clean.py          # 清理 XML
    ├── thumbnail.py      # 生成缩略图
    └── office/           # Office 文件处理工具
        ├── unpack.py     # 解压 .pptx
        ├── pack.py       # 打包 .pptx
        └── soffice.py    # LibreOffice 包装器
```

### SKILL.md 的作用

这是 skill 的"大脑"，告诉模型：
1. **什么时候用这个 skill**（description）
2. **有哪些能力**（Quick Reference）
3. **怎么使用这些能力**（具体命令和工作流）
4. **有哪些辅助资源**（scripts、补充文档）

### scripts/ 的作用

**不是独立的 tool**，而是：
- 可以通过 `bash` 工具执行的 Python 脚本
- 封装了复杂的领域逻辑（如 XML 操作）
- 模型可以灵活组合使用

### 补充文档的作用

当 SKILL.md 太长时，拆分成多个文档：
- `editing.md` - 详细的编辑工作流
- `pptxgenjs.md` - 从零创建的详细指南

模型会在需要时 `read_file("skills/pptx/editing.md")`

---

## PPTX Skill 实战解析

### 场景 1：读取 PPT 内容

**用户请求**：
```
"帮我看看 presentation.pptx 里都讲了什么"
```

**模型的思考过程**：
1. 识别关键词 "pptx" → 需要 pptx skill
2. 调用 `load_skill("pptx")`

**系统返回 SKILL.md 内容**（部分）：
```markdown
## Reading Content

```bash
# Text extraction
python -m markitdown presentation.pptx

# Visual overview
python scripts/thumbnail.py presentation.pptx
```
```

**模型执行**：
```python
# 模型看到 SKILL.md 里推荐用 markitdown
bash("python -m markitdown presentation.pptx")
```

**返回结果**：
```
Slide 1: Introduction
- Welcome to our Q4 review
- Key metrics overview

Slide 2: Revenue Growth
- Q4 revenue: $2.5M
- 25% increase YoY
...
```

**模型回复用户**：
```
这个 PPT 包含 5 张幻灯片，主要内容是 Q4 业务回顾：
1. 介绍页
2. 收入增长（Q4 收入 250 万美元，同比增长 25%）
3. ...
```

---

### 场景 2：编辑现有 PPT

**用户请求**：
```
"把 template.pptx 的第 2 页标题改成 'Revenue Highlights'，
然后复制这一页，在副本里把数字改成 Q3 的数据"
```

这是一个**复杂的多步骤任务**，让我们看看模型如何处理。

#### 第 1 步：加载 Skill

```python
load_skill("pptx")
```

**SKILL.md 告诉模型**：
```markdown
## Editing Workflow

**Read [editing.md](editing.md) for full details.**

1. Analyze template with `thumbnail.py`
2. Unpack → manipulate slides → edit content → clean → pack
```

#### 第 2 步：读取详细工作流

模型看到 "Read editing.md"，于是：

```python
read_file("skills/pptx/editing.md")
```

**editing.md 内容**（简化版）：
```markdown
# PPTX Editing Workflow

## Step 1: Visual Analysis
```bash
python scripts/thumbnail.py template.pptx
```
This generates a grid showing all slides.

## Step 2: Unpack
```bash
python scripts/office/unpack.py template.pptx unpacked/
```
Extracts the .pptx into XML files.

## Step 3: Locate Content
Slides are in `unpacked/ppt/slides/slide2.xml`

## Step 4: Edit XML
Modify the XML directly or use text replacement.

## Step 5: Add Slides
```bash
python scripts/add_slide.py unpacked/ slide2.xml
```

## Step 6: Clean & Pack
```bash
python scripts/clean.py unpacked/
python scripts/office/pack.py unpacked/ output.pptx
```
```

#### 第 3 步：执行工作流

现在模型知道完整流程了，开始执行：

**3.1 分析模板**
```python
bash("python scripts/thumbnail.py template.pptx")
```
返回：缩略图网格，模型可以看到每页的布局

**3.2 解压 PPT**
```python
bash("python scripts/office/unpack.py template.pptx unpacked/")
```
返回：`Unpacked to unpacked/`

现在文件结构是：
```
unpacked/
├── [Content_Types].xml
├── ppt/
│   ├── presentation.xml
│   ├── slides/
│   │   ├── slide1.xml
│   │   ├── slide2.xml    ← 我们要编辑这个
│   │   └── slide3.xml
│   └── slideLayouts/
└── _rels/
```

**3.3 读取第 2 页内容**
```python
read_file("unpacked/ppt/slides/slide2.xml")
```

返回（简化）：
```xml
<?xml version="1.0"?>
<p:sld>
  <p:cSld>
    <p:spTree>
      <p:sp>
        <p:txBody>
          <a:p>
            <a:r>
              <a:t>Q4 Revenue</a:t>  ← 找到了标题
            </a:r>
          </a:p>
        </p:txBody>
      </p:sp>
      <p:sp>
        <p:txBody>
          <a:p>
            <a:r>
              <a:t>$2.5M</a:t>  ← 找到了数据
            </a:r>
          </a:p>
        </p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
</p:sld>
```

**3.4 修改标题**
```python
edit_file(
    path="unpacked/ppt/slides/slide2.xml",
    old_text="<a:t>Q4 Revenue</a:t>",
    new_text="<a:t>Revenue Highlights</a:t>"
)
```

**3.5 复制幻灯片**
```python
bash("python scripts/add_slide.py unpacked/ slide2.xml")
```

返回：
```
Created slide4.xml from slide2.xml
Add to presentation.xml <p:sldIdLst>: <p:sldId id="257" r:id="rId5"/>
```

**3.6 编辑新幻灯片的数据**
```python
read_file("unpacked/ppt/slides/slide4.xml")
# 找到 $2.5M

edit_file(
    path="unpacked/ppt/slides/slide4.xml",
    old_text="<a:t>$2.5M</a:t>",
    new_text="<a:t>$2.1M</a:t>"  # Q3 数据
)

edit_file(
    path="unpacked/ppt/slides/slide4.xml",
    old_text="<a:t>Q4</a:t>",
    new_text="<a:t>Q3</a:t>"
)
```

**3.7 更新 presentation.xml**

模型需要把新幻灯片注册到主文件：

```python
read_file("unpacked/ppt/presentation.xml")
# 找到 <p:sldIdLst>

edit_file(
    path="unpacked/ppt/presentation.xml",
    old_text="</p:sldIdLst>",
    new_text='  <p:sldId id="257" r:id="rId5"/>\n</p:sldIdLst>'
)
```

**3.8 清理和打包**
```python
bash("python scripts/clean.py unpacked/")
# 返回：Cleaned XML formatting

bash("python scripts/office/pack.py unpacked/ output.pptx")
# 返回：Packed to output.pptx
```

**3.9 验证结果**
```python
bash("python scripts/thumbnail.py output.pptx")
# 生成缩略图，模型检查是否正确
```

#### 模型回复用户

```
已完成修改：
1. ✅ 将第 2 页标题改为 "Revenue Highlights"
2. ✅ 复制了第 2 页，创建为第 4 页
3. ✅ 将第 4 页的数据改为 Q3（$2.1M）

生成的文件：output.pptx

我已经验证了结果，4 张幻灯片都正确显示。
```

---

### 场景 3：从零创建 PPT

**用户请求**：
```
"创建一个产品介绍 PPT，包括：
1. 封面（产品名：CloudSync）
2. 功能介绍（3 个要点）
3. 定价页"
```

#### 第 1 步：加载 Skill 并选择方法

```python
load_skill("pptx")
```

**SKILL.md 的 Quick Reference**：
```markdown
| Task | Guide |
|------|-------|
| Edit or create from template | Read [editing.md](editing.md) |
| Create from scratch | Read [pptxgenjs.md](pptxgenjs.md) |
```

模型判断：没有模板，需要从零创建 → 读取 pptxgenjs.md

```python
read_file("skills/pptx/pptxgenjs.md")
```

#### 第 2 步：学习 pptxgenjs 用法

**pptxgenjs.md 内容**（简化）：
```markdown
# Creating PPTX from Scratch with pptxgenjs

## Installation
```bash
npm install -g pptxgenjs
```

## Basic Usage
```javascript
const pptxgen = require("pptxgenjs");
let pres = new pptxgen();

// Add slide
let slide = pres.addSlide();
slide.addText("Hello World", {
  x: 1, y: 1, w: 8, h: 1,
  fontSize: 44, bold: true
});

pres.writeFile({ fileName: "output.pptx" });
```

## Layout Guidelines
- Title slide: centered text, large font (44pt)
- Content slide: title at top (36pt), bullets below (18pt)
- Use color palette from SKILL.md
```

#### 第 3 步：创建 JavaScript 脚本

模型现在知道怎么做了，开始写代码：

```python
write_file(
    path="create_presentation.js",
    content="""
const pptxgen = require("pptxgenjs");
let pres = new pptxgen();

// 设置主题颜色（使用 SKILL.md 推荐的 Midnight Executive）
const colors = {
  primary: "1E2761",   // navy
  secondary: "CADCFC", // ice blue
  accent: "FFFFFF"     // white
};

// Slide 1: 封面
let slide1 = pres.addSlide();
slide1.background = { color: colors.primary };
slide1.addText("CloudSync", {
  x: 0.5, y: 2.5, w: 9, h: 1.5,
  fontSize: 60, bold: true, color: colors.accent,
  align: "center"
});
slide1.addText("Seamless Cloud Storage for Teams", {
  x: 0.5, y: 4, w: 9, h: 0.5,
  fontSize: 24, color: colors.secondary,
  align: "center"
});

// Slide 2: 功能介绍
let slide2 = pres.addSlide();
slide2.addText("Key Features", {
  x: 0.5, y: 0.5, w: 9, h: 0.8,
  fontSize: 36, bold: true, color: colors.primary
});

const features = [
  { text: "Real-time Sync", desc: "Instant file synchronization across all devices" },
  { text: "Team Collaboration", desc: "Share and co-edit documents seamlessly" },
  { text: "Enterprise Security", desc: "Bank-level encryption and compliance" }
];

let yPos = 1.8;
features.forEach((feature, i) => {
  // 图标圆圈
  slide2.addShape(pres.ShapeType.ellipse, {
    x: 0.8, y: yPos - 0.1, w: 0.4, h: 0.4,
    fill: { color: colors.primary }
  });

  // 标题
  slide2.addText(feature.text, {
    x: 1.5, y: yPos, w: 7.5, h: 0.4,
    fontSize: 20, bold: true, color: colors.primary
  });

  // 描述
  slide2.addText(feature.desc, {
    x: 1.5, y: yPos + 0.4, w: 7.5, h: 0.3,
    fontSize: 14, color: "666666"
  });

  yPos += 1.2;
});

// Slide 3: 定价
let slide3 = pres.addSlide();
slide3.addText("Pricing", {
  x: 0.5, y: 0.5, w: 9, h: 0.8,
  fontSize: 36, bold: true, color: colors.primary
});

const plans = [
  { name: "Starter", price: "$9/mo", features: "100GB storage" },
  { name: "Pro", price: "$29/mo", features: "1TB storage + Priority support" },
  { name: "Enterprise", price: "Custom", features: "Unlimited + Dedicated account manager" }
];

let xPos = 0.8;
plans.forEach(plan => {
  // 卡片背景
  slide3.addShape(pres.ShapeType.rect, {
    x: xPos, y: 1.5, w: 2.6, h: 3,
    fill: { color: "F5F5F5" },
    line: { color: colors.primary, width: 2 }
  });

  // 方案名
  slide3.addText(plan.name, {
    x: xPos, y: 1.8, w: 2.6, h: 0.5,
    fontSize: 24, bold: true, color: colors.primary,
    align: "center"
  });

  // 价格
  slide3.addText(plan.price, {
    x: xPos, y: 2.5, w: 2.6, h: 0.8,
    fontSize: 36, bold: true, color: colors.primary,
    align: "center"
  });

  // 功能
  slide3.addText(plan.features, {
    x: xPos + 0.2, y: 3.5, w: 2.2, h: 0.6,
    fontSize: 12, color: "666666",
    align: "center"
  });

  xPos += 3;
});

// 保存
pres.writeFile({ fileName: "CloudSync_Presentation.pptx" });
console.log("Created CloudSync_Presentation.pptx");
"""
)
```

#### 第 4 步：执行脚本

```python
bash("node create_presentation.js")
```

返回：
```
Created CloudSync_Presentation.pptx
```

#### 第 5 步：验证结果

```python
bash("python scripts/thumbnail.py CloudSync_Presentation.pptx")
```

模型检查缩略图，确认：
- ✅ 封面：深蓝背景，白色大标题
- ✅ 功能页：3 个要点，带图标和描述
- ✅ 定价页：3 个卡片式布局

#### 模型回复用户

```
已创建 CloudSync 产品介绍 PPT，包含 3 张幻灯片：

1. 封面 - 使用深蓝色主题，突出产品名称
2. 功能介绍 - 3 个核心功能，每个带图标和详细说明
3. 定价页 - 3 个方案的卡片式对比

文件：CloudSync_Presentation.pptx

我使用了 SKILL.md 推荐的 "Midnight Executive" 配色方案（深蓝 + 冰蓝 + 白色），
并遵循了设计指南中的布局原则（图标 + 文字、卡片式对比）。
```

---

## 工作流程详解

### references/ 的使用场景

在 agent-builder skill 中：

```
skills/agent-builder/
├── SKILL.md
├── scripts/
│   └── init_agent.py       # 脚手架生成器
└── references/
    ├── minimal-agent.py    # 最小化 agent 示例
    ├── tool-templates.py   # Tool 定义模板
    └── subagent-pattern.py # Subagent 模式示例
```

**使用场景**：

**场景：用户想学习如何构建 agent**

```
用户："教我怎么写一个最简单的 agent"

模型：
1. load_skill("agent-builder")
2. 看到 SKILL.md 提到：
   "references/minimal-agent.py - Complete working agent (~80 lines)"
3. read_file("skills/agent-builder/references/minimal-agent.py")
4. 向用户展示并解释代码
```

**references/ 的作用**：
- 📖 **教学材料**：完整的可运行示例
- 🎯 **最佳实践**：展示推荐的实现方式
- 🔍 **参考代码**：模型可以学习并改编

**与 scripts/ 的区别**：
- `scripts/` - 直接执行的工具（`bash("python scripts/xxx.py")`）
- `references/` - 学习和参考的代码（`read_file("references/xxx.py")`）

---

## 设计哲学

### 为什么不把每个脚本做成独立 Tool？

**如果采用传统方式**：

```python
TOOLS = [
    {"name": "thumbnail_pptx", "description": "Generate PPTX thumbnails", ...},
    {"name": "unpack_pptx", "description": "Unpack PPTX to XML", ...},
    {"name": "pack_pptx", "description": "Pack XML to PPTX", ...},
    {"name": "add_slide_pptx", "description": "Add slide to PPTX", ...},
    {"name": "clean_pptx", "description": "Clean PPTX XML", ...},
    # ... pptx skill 就需要 5+ 个 tools

    {"name": "merge_pdf", ...},
    {"name": "split_pdf", ...},
    # ... pdf skill 又需要 N 个 tools

    # 100 个 skills × 平均 5 个工具 = 500 个 tools？
]
```

**问题**：
1. **Tool 爆炸**：模型需要在 500 个 tool 中选择
2. **不灵活**：如果需要修改脚本参数？如果需要组合使用？
3. **维护困难**：每个脚本都要定义 tool schema
4. **上下文浪费**：所有 tool 定义都占用 system prompt

**Claude Code 的方式**：

```python
TOOLS = [
    bash,         # 执行任何命令
    read_file,    # 读任何文件
    write_file,   # 写任何文件
    edit_file,    # 编辑任何文件
    load_skill    # 加载任何技能
]
# 永远只有 5 个 tools
```

**优势**：
1. **工具最小化**：5 个通用工具覆盖所有场景
2. **知识模块化**：每个 skill 是独立的知识包
3. **灵活组合**：模型可以创造性地使用脚本
4. **易于扩展**：添加新 skill 不需要改代码

### 模型的智能体现在哪里？

**传统方式**：
```
用户："编辑 PPT"
模型：调用 edit_pptx_tool(file="x.pptx", changes=[...])
系统：执行预定义的编辑逻辑
```
模型只是"调用工具"，没有理解业务。

**Claude Code 方式**：
```
用户："编辑 PPT"
模型：
  1. 我需要 pptx 知识 → load_skill("pptx")
  2. 学习：原来要先 unpack，再编辑 XML，再 pack
  3. 执行：bash("unpack...") → read_file(...) → edit_file(...) → bash("pack...")
  4. 验证：bash("thumbnail...") 检查结果
```
模型**理解了业务流程**，可以灵活应对变化。

**例子：用户临时改需求**

```
用户："等等，在打包之前，帮我把所有幻灯片的字体都改成 Arial"

传统方式：
  ❌ edit_pptx_tool 没有"改字体"参数 → 无法完成

Claude Code 方式：
  ✅ 模型理解了 XML 结构，可以：
     1. 遍历所有 slideN.xml
     2. 用正则替换字体定义
     3. 然后继续 pack
```

### 信任模型的能力

这是 Claude Code 最核心的哲学：

> **The model already knows how to be an agent. Your job is to get out of the way.**

**不要过度设计**：
- ❌ 不要预定义 50 个专用工具
- ❌ 不要写死工作流程
- ❌ 不要限制模型的创造力

**给模型自由**：
- ✅ 提供通用能力（bash, read, write, edit）
- ✅ 提供领域知识（skills）
- ✅ 让模型自己组合和创新

---

## 总结

### Skill 的本质

```
Skill ≠ Tool
Skill = 知识库 + 工具箱 + 教学材料
```

**三个组成部分**：
1. **SKILL.md** - 核心知识，告诉模型"怎么做"
2. **scripts/** - 辅助脚本，通过 `bash` 执行
3. **references/** - 参考代码，通过 `read_file` 学习

### 工作流程

```
用户请求
    ↓
模型识别需要的 skill
    ↓
load_skill("xxx") ← 加载知识
    ↓
模型学习 SKILL.md
    ↓
模型制定执行计划
    ↓
组合使用 bash/read/write/edit
    ↓
调用 scripts/ 中的脚本
    ↓
读取 references/ 学习
    ↓
完成任务
```

### 设计原则

1. **最小化 Tools**：5 个通用工具足够
2. **模块化知识**：每个 skill 独立维护
3. **按需加载**：用到才加载，节省上下文
4. **信任模型**：给知识，不给限制

### PPTX Skill 的启示

通过 pptx skill，我们看到：
- **复杂任务可以分解**：unpack → edit → pack
- **脚本封装复杂逻辑**：XML 操作不需要模型手写
- **模型理解业务流程**：不是机械调用，而是灵活组合
- **验证闭环**：thumbnail 检查结果

这就是 Claude Code Skills 的精髓：**给模型知识和工具，让它自己思考如何完成任务**。
