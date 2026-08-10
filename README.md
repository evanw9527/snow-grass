# Snow Grass

Snow Grass is the Python/FastAPI backend for the Sedum Agent interface. It provides a
multi-model catalog, persistent chat sessions, declarative Skills, a controlled tool registry,
and Server-Sent Events for model output and execution traces.

## Local development

```bash
conda create -n snow-grass python=3.12 -y
conda activate snow-grass
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
uvicorn snow_grass.main:app --host 127.0.0.1 --port 8000 --reload
```

The API is served at `http://127.0.0.1:8000`; health is available at
`GET /api/v1/health`. Model keys are optional for startup. Without a configured key the model
catalog marks that model as unavailable and chat returns a structured SSE failure event.

Conda manages the Python runtime, while pip installs the exact package versions recorded in
`requirements.lock.txt`. After changing dependencies in `pyproject.toml`, refresh the lock file
from a clean `snow-grass` Conda environment:

```bash
python -m pip install -e . --group dev
python -m pip freeze --exclude-editable > requirements.lock.txt
```

Quality checks:

```bash
ruff check .
mypy src tests
pytest
```

## Memory

Snow Grass keeps the full chat transcript in SQLite while sending only a bounded context to the
model. The context is assembled in this order:

1. Agent, model identity, and active Skill instructions.
2. Active workspace/session memories relevant to the latest question.
3. The current rolling session summary.
4. Recent raw messages within the configured count and estimated Token budget.

Messages covered by the current summary are logical cold data: they remain available from the
history API but are excluded from normal model requests. When compaction fails, chat degrades to
the recent-message window instead of failing. Token counts used before a request are conservative
estimates; provider-returned usage remains the source of truth for billing analytics.

Memory endpoints:

```text
GET    /api/v1/sessions/{session_id}/memory
POST   /api/v1/sessions/{session_id}/memory/compact
GET    /api/v1/memories?scope_type=session&scope_id={session_id}
POST   /api/v1/memories
PATCH  /api/v1/memories/{memory_id}
DELETE /api/v1/memories/{memory_id}
```

Long-term memories support `session` and `workspace` scopes. Only active, unexpired, non-deleted
items are eligible for retrieval. Credential-like content is rejected and is never persisted as
long-term memory. Optional automatic extraction is disabled by default; when enabled with
`SNOW_GRASS_MEMORY_AUTO_EXTRACT_ENABLED=true`, extracted facts are stored as `candidate` and do
not affect model context until explicitly activated through the PATCH API.

## Skill management and publishing

Sedum 的“技能管理”页面连接 Snow Grass 的 `/api/v1/admin/skills` 管理接口。技能有两种来源：

- `builtin`：从 `skills/` 目录加载，只读；如需修改，先克隆为自定义技能。
- `managed`：保存在 SQLite，依次经过草稿保存、校验、SemVer 发布和版本激活。

每个草稿和 Release 都是完整的 UTF-8 标准技能包：

```text
skill-id/
├── SKILL.md                 # 必需：name/description frontmatter + 执行说明
├── manifest.yaml            # 必需：ID、版本、选择规则、模型、工具和运行限制
├── agents/
│   └── openai.yaml          # 推荐：列表展示和默认提示词元数据
├── scripts/                 # 可选：Agent 可调用的 Python 脚本
├── references/              # 可选：按需读取的领域资料
├── assets/                  # 可选：输出模板和静态资源（当前管理 API 限 UTF-8 文本）
└── tests/
    └── selection.yaml       # 可选：自动选择规则测试
```

`SKILL.md` frontmatter 只允许 `name` 和 `description`，其中 `name` 必须与技能 ID 一致；用户看到的
名称由 `agents/openai.yaml` 的 `display_name` 提供。包内路径必须是安全的 POSIX 相对路径，
禁止绝对路径和 `..`。`scripts/` 会随 Release 完整版本化；当前只支持 `.py` 文件。Skill 被选中
不会自动执行脚本，模型会根据 `SKILL.md` 的流程，通过 `run_skill_script` 函数调用选择脚本和参数，
Snow Grass 执行后再把结构化结果返回模型。

例如，Skill 包包含 `scripts/query.py` 时，可以在 `SKILL.md` 中明确写出：

```markdown
用户请求实时数据时，调用 `scripts/query.py`。第一个参数是城市名。必须根据脚本返回结果回答，
脚本失败时说明具体失败原因，不得编造实时数据。
```

脚本通过当前 Snow Grass Conda 环境的 Python 解释器运行，参数直接以字符串数组传入，不经过 Shell。
每次执行使用临时工作目录，并受 `manifest.yaml.limits.timeout_seconds`、全局超时和输出大小限制。
相关配置为：

```text
SNOW_GRASS_SKILL_SCRIPT_EXECUTION_ENABLED=true
SNOW_GRASS_SKILL_SCRIPT_TIMEOUT_SECONDS=30
SNOW_GRASS_SKILL_SCRIPT_MAX_OUTPUT_CHARS=65536
```

这套限制不是容器或虚拟机级强隔离：Python 进程仍以 Snow Grass 服务进程账号运行。管理端只能发布
经过审核的可信脚本；生产环境若允许不受信任用户编辑 Skill，应关闭脚本执行，或在外部容器/沙箱服务中
实现执行器。Snow Grass 不执行 Skill 中的 JavaScript、Java、二进制文件或任意 Shell 命令字符串。

自动选择以 `SKILL.md` 的 `name + description` 为主要匹配元数据，同时支持中英文；
`manifest.yaml.selection.keywords` 是可选的高权重信号，不配置关键词也可以发布和自动命中。
`selection.mode=manual` 的技能只允许用户显式选择。

草稿使用递增 `revision` 做乐观锁；提交旧 revision 时返回 HTTP 409，避免覆盖其他修改。每次发布
会生成不可变 Release，激活历史 Release 即完成回滚。启停状态、当前激活版本和操作审计都会持久化，
服务重启后继续生效。运行时只读取已发布且已激活的快照，未发布草稿不会影响对话。

主要管理接口：

```text
GET    /api/v1/admin/skills
POST   /api/v1/admin/skills
GET    /api/v1/admin/skills/{skill_id}
PUT    /api/v1/admin/skills/{skill_id}/draft
POST   /api/v1/admin/skills/{skill_id}/validate
POST   /api/v1/admin/skills/{skill_id}/publish
POST   /api/v1/admin/skills/{skill_id}/releases/{release_id}/activate
PUT    /api/v1/admin/skills/{skill_id}/enabled
POST   /api/v1/admin/skills/{skill_id}/clone
POST   /api/v1/admin/skills/{skill_id}/archive
```

管理接口面向本地开发环境。任意文件中疑似密钥的内容会被校验器拒绝；模型和工具引用必须存在于
后端注册表。已发布 Skill 的 Python 脚本可按上述受控流程执行，因此生产部署前必须增加身份认证、
发布权限和审核控制；也可以设置 `SNOW_GRASS_SKILL_ADMIN_ENABLED=false` 关闭管理接口，或设置
`SNOW_GRASS_SKILL_SCRIPT_EXECUTION_ENABLED=false` 完全关闭脚本执行。

Sedum 本地启动：

```bash
cd /Users/tb/Documents/sedum
npm install
npm run dev
```

Vite 开发服务器通过现有代理访问 `http://127.0.0.1:8000`；如需直连其他后端地址，设置
`VITE_API_BASE_URL`。
