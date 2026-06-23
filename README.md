# bom-agent

基于 LangChain 的结构分解 agent。输入一个对象名，agent 会尽量搜索公开网页信息，对其进行通用的结构化拆解，并输出结构化 JSON。

## 设计目标

- 以公开信息为主：优先搜索拆机、规格页、维修指南、专利、FCC/监管材料、供应链资料、材料说明。
- 强制结构化输出：不是自由文本，而是可消费的分解 JSON。
- 支持不完全信息：对无法直接证实的节点，允许行业常识推断，但必须降低置信度并写明依据。

## 当前实现

- Agent 框架：`LangChain`
- 模型接入：`langchain_openai.ChatOpenAI`
- 工具：
  - `web_search`：公网搜索产品/组件/材料线索
  - `read_web_page`：通过 Jina Reader 抓取网页可读正文
- 输出模型：
  - `subject -> top level nodes -> optional child nodes`

## 安装

要求：

- Python `3.11+`
- 可用的模型服务 API key

安装依赖：

```bash
pip install -e .
cp config.example.toml config.local.toml
```

然后编辑 `config.local.toml`，把真实 key 写进去。这个文件已被 Git 忽略，不会默认提交。

如果你用的是 DeepSeek，推荐这样配：

```toml
api_key = "你的 DeepSeek key"
base_url = "https://api.deepseek.com"
model = "deepseek-v4-flash"
search_provider = "serper"
tavily_api_key = ""
serper_api_key = "你的 Serper key"
brave_api_key = ""
searxng_base_url = ""
thinking_enabled = false
model_timeout_seconds = 30
model_max_retries = 0
max_tool_calls = 12
recursion_limit = 24
```

## 使用

```bash
bom-agent "iPhone 15 Pro"
```

带补充上下文：

```bash
bom-agent "MacBook Air" --context "13-inch M3 model released in 2024"
```

输出可读文本：

```bash
bom-agent "锂电产业链" --format text
```

## 输出示例

```json
{
  "subject_name": "Example Device",
  "subject_kind": "product",
  "subject_description": "Short description of the device and the framing used for this analysis.",
  "decomposition_goal": "Explain the main functional structure of the product.",
  "decomposition_basis": "functional modules",
  "depth_policy": "Only direct downstream items, no further decomposition.",
  "scope_notes": [
    "Battery cathode chemistry inferred from teardown and supplier context."
  ],
  "top_level_nodes": [
    {
      "name": "显示屏",
      "description": "6.1 英寸 OLED 显示屏，支持高刷新率",
      "supplier_market": "约 80% 由三星供应，约 20% 由京东方供应",
      "cost_share": "30-40%"
    },
    {
      "name": "处理器",
      "description": "A 系列 SoC，含 CPU/GPU/神经引擎",
      "supplier_market": "主要供应商为台积电代工，苹果自研设计",
      "cost_share": "15-25%"
    },
    {
      "name": "电池",
      "description": "锂离子电池，容量约 3300mAh",
      "supplier_market": "主要供应商包括德赛电池、欣旺达，具体份额未公开",
      "cost_share": "5-10%"
    }
  ]
}
```

## 项目结构

```text
src/
  agent.py
  config.py
  main.py
  models.py
  prompts.py
  web_tools.py
```

## 限制

- 公开网页的结构和反爬策略会影响搜索质量。
- 当前 `web_search` 使用公开搜索结果页解析，稳定性一般，后续建议替换成正式搜索 API。
- 当前 `web_search` 支持可切换 provider：`tavily`、`serper`、`brave`、`searxng`。`read_web_page` 仍通过 Jina Reader 抓取正文。
- 用户要求的 `agent-reach` skill 在本环境中声明存在，但 `agent-reach` 可执行程序不可用，因此当前实现未能直接接入该工具链。

## 配置优先级

程序按以下顺序读取配置：

1. 环境变量
2. `config.local.toml`
3. 代码默认值

可配置项：

- `api_key`
- `base_url`
- `model`
- `search_provider`
- `tavily_api_key`
- `serper_api_key`
- `brave_api_key`
- `searxng_base_url`
- `thinking_enabled`
- `temperature`
- `model_timeout_seconds`
- `model_max_retries`
- `max_pages`
- `max_search_results`
- `max_tool_calls`
- `recursion_limit`

兼容字段：

- `deepseek_api_key`
- `openai_api_key`

当前默认值已经切到 DeepSeek：

- `base_url = "https://api.deepseek.com"`
- `model = "deepseek-v4-flash"`
- `thinking_enabled = false`

## 下一步建议

- 接入正式搜索 API，例如 Exa、SerpAPI 或 Tavily。
- 为不同对象类型增加更明确的分解策略模板，例如产品结构、工艺流程、供应链、产业链。
- 增加结果去重、证据打分、术语词典和单独的分解树校验器。
