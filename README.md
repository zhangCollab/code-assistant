# AI Agent - 智能编程助手

交互式命令行编程助手，支持通过自然语言执行各种开发任务。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 进入当前目录的交互模式
python main.py

# 指定工作目录
python main.py ./project
```

## 功能特性

- **多引擎支持**：支持 OpenAI、Qwen（阿里千问）、BigModel（智谱） 等多种大语言模型
- **会话管理**：支持创建、切换、删除多个会话，上下文自动关联
- **智能工具调用**：内置文件读写、命令执行、代码搜索等开发工具
- **Agent 模式**：自主理解需求、分解任务、渐进式执行

## 环境变量配置

需要配置以下环境变量：

| 变量名 | 说明             | 默认值        |
|--------|----------------|------------|
| `api_key` | OpenAI API Key | -          |
| `producer` | 引擎类型           | `bigmodel` |
| `model` | 模型名称           | `glm-4.7`  |
| `enable_thinking` | 思考模式           | `True`     |
| `api_key` | API 密钥         | 必需         |

## 使用方法

### 交互模式

启动后直接进入交互模式，输入任务描述即可。

```bash
python main.py
```

### 会话管理命令

| 命令 | 说明 |
|------|------|
| `/new` | 创建新会话 |
| `/show` | 列出所有会话 |
| `/switch <id>` | 切换到指定会话 |
| `/delete <id>` | 删除会话 |

### 示例任务

- 创建一个 HTTP 服务器
- 帮我优化这段代码的性能
- 解释这个函数的作用
- 添加用户认证功能

## 支持的 LLM 引擎

## 内置工具

| 工具 | 说明 |
|------|------|
| `read` | 读取文件内容 |
| `write` | 创建/覆盖文件 |
| `edit` | 编辑文件内容 |
| `glob` | 文件路径搜索 |
| `grep` | 内容搜索 |
| `bash` | 执行命令 |
| `todowrite` | 任务清单管理 |
| `question` | 用户交互 |
| `webfetch` | 获取网页内容 |

## 项目结构

```
code-assistant/
├── main.py                 # 主入口
├── config.py              # 配置文件
├── requirements.txt        # 依赖列表
├── core/
│   ├── agent_engine.py    # Agent 引擎
│   ├── llm_engine.py      # LLM 引擎
│   ├── tool_executor.py   # 工具执行器
│   ├── tool_definitions.py # 工具定义
│   ├── session_manager.py # 会话管理
│   ├── file_manager.py    # 文件管理
└── workdir/               # 工作目录
```

## License

MIT
