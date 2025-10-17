## RAG项目：基于RAG的本地QA系统

## 🧭 一、总体结构概览

这个项目是一个完整的 **本地 RAG 智能问答系统**。
 它包含以下五个层次 👇

| 层级                 | 模块                                         | 功能                                                         |
| -------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| **1️⃣ 数据输入层**     | PDF文件上传、提取、切分                      | 将原始文档转为小段文本块（chunks）                           |
| **2️⃣ 向量化与索引层** | `SentenceTransformer` + `FAISS` + `BM25`     | 将文本块转为向量、建立索引、支持语义检索与关键词检索         |
| **3️⃣ 检索与重排序层** | `recursive_retrieval()` + CrossEncoder / LLM | 结合语义搜索与关键词搜索，递归生成更优查询，提升召回率与准确性 |
| **4️⃣ 答案生成层**     | `Ollama` / `SiliconFlow API`                 | 调用大模型生成最终回答                                       |
| **5️⃣ 前端展示层**     | `Gradio`                                     | 文件上传、进度可视化、问答界面、分块查看                     |

🧩 模块之间的逻辑关系如下图（与 readme.md 一致）：

```
PDF → 提取 → 切分 → 向量化 → FAISS + BM25 建索引
用户提问 → 向量化查询 → 混合检索 + 重排序 → 上下文拼接
上下文 + Prompt → LLM 生成答案 → 输出至前端
```

------

## ⚙️ 二、核心逻辑流程

下面我们按实际运行流程分解系统核心逻辑👇

------

### **1️⃣ 文档处理阶段 — process_multiple_pdfs()**

📍文件位置：

```python
def process_multiple_pdfs(files: List[Any], progress=gr.Progress()):
```

🔹 主要任务：

- 提取 PDF 文本（`pdfminer`）
- 切分文本块（`RecursiveCharacterTextSplitter`）
- 向量化嵌入（`SentenceTransformer`）
- 存入 `FAISS` 索引
- 构建 `BM25` 索引用于关键词检索

🔹 处理逻辑图：

```python
for 每个上传的PDF:
    text = extract_text(file)
    chunks = split_text(text, chunk_size=400)
    embeddings = EMBED_MODEL.encode(chunks)
    faiss_index.add(embeddings)
    记录 {id -> content, metadata}
BM25_MANAGER.build_index(all_texts)
```

🧠 你可以理解为：

> “我把每个PDF拆成很多小段话，并把这些小段变成向量存进FAISS数据库里，用于后续相似度检索。”

------

### **2️⃣ 混合检索阶段 — recursive_retrieval()**

📍函数：

```python
def recursive_retrieval(initial_query, max_iterations=3, enable_web_search=False, model_choice="ollama"):
```

🔹 功能：

- 根据用户问题执行 **多轮迭代检索**；
- 每轮结合两种检索结果：
  - **语义检索**（FAISS 向量相似度）
  - **关键词检索**（BM25）
- 用 **交叉编码器（CrossEncoder）** 或 **LLM** 进行重排序；
- 通过 **LLM 判断是否需要继续查询**（递归检索）。

🔹 流程图：

```python
query = 用户问题
for i in range(3):   # 最多3轮
    faiss_results = 向量搜索(query)
    bm25_results = 关键词搜索(query)
    hybrid_results = hybrid_merge(faiss_results, bm25_results)
    reranked_results = rerank_results(query, hybrid_results)
    
    if LLM判断信息足够:
        break
    else:
        query = LLM生成的新查询（例如“发动机冒蓝烟原因”→“涡轮增压器故障是否相关？”）
```

🧠 面试讲法：

> “我实现了一个递归式检索机制，系统会让 LLM 分析当前结果是否充分，不够就自动生成新查询，再次检索。”

------

### **3️⃣ 重排序阶段 — rerank_results()**

📍核心逻辑：

```python
if method == "cross_encoder":
    return rerank_with_cross_encoder(...)
elif method == "llm":
    return rerank_with_llm(...)
```

🔹 两种方式：

1. **CrossEncoder（句对匹配模型）**
   - 输入 `[query, doc]`
   - 输出相关性分数（0~1）
   - 更快，离线可运行
2. **LLM 打分**
   - 提问 LLM “这段话与问题相关性几分？”
   - 精度高但开销大

🔹 意义：

> “这一步是 RAG 系统的精髓。它能让最终送入大模型生成的上下文更精准、更贴近问题。”

------

### **4️⃣ 答案生成阶段 — query_answer() / stream_answer()**

📍核心逻辑：

```python
prompt = f"""
作为一个专业的问答助手，你需要基于以下参考内容回答问题：
{context}
问题：{question}
请遵循以下原则：
1. 仅基于参考内容回答
2. 不够信息要明确说明
3. 用中文，结构清晰
"""
```

🔹 核心任务：

- 整合上下文（local + web）
- 拼接 Prompt
- 调用模型生成回答：
  - 本地 `Ollama (deepseek-r1)`
  - 云端 `SiliconFlow API`

🔹 同时还包含：

- `<think>` 思维链内容解析
- 矛盾检测（`detect_conflicts()`）
- 来源可信度评估（`evaluate_source_credibility()`）

🧠 面试讲法：

> “生成阶段不仅做简单回答，还会检测来源间的矛盾，评估可靠性，并把 LLM 的推理过程以 `<think>` 折叠展示。”

------

### **5️⃣ 前端交互层 — Gradio 界面**

📍部分定义：

```
with gr.Blocks(title="本地RAG问答系统"):
```

🔹 包含：

- 文件上传组件
- 文档处理进度条
- 提问输入框 + 联网搜索按钮
- 文档分块可视化表格（`get_document_chunks()`）

🧠 面试讲法：

> “为了方便展示，我用 Gradio 构建了前端界面，用户可以上传PDF、查看分块、提问并实时看到回答生成。”

------

## 🔍 三、系统中的关键创新点（可重点讲）

| 功能                                | 技术亮点                                |
| ----------------------------------- | --------------------------------------- |
| **递归检索**                        | 自动生成新查询，使 RAG 更智能、更深入   |
| **混合检索 (Hybrid Search)**        | FAISS + BM25 融合，兼顾语义与关键词     |
| **交叉编码器重排序**                | 二阶段排序提升精准度                    |
| **矛盾检测与来源评估**              | 判断多来源内容冲突，提高可信度          |
| **思维链可视化**                    | 将 LLM 推理过程折叠展示，便于解释性分析 |
| **双模型支持 (Ollama/SiliconFlow)** | 支持本地部署与云端推理切换              |



------



# FAST API 接口层

## 🧭 一、快速总结：你目前的 API 层已经具备哪些功能

| 功能                                          | 实现情况                     | 说明                                                         |
| --------------------------------------------- | ---------------------------- | ------------------------------------------------------------ |
| **文档上传接口** `/api/upload`                | ✅ 已实现                     | 接收 PDF、调用 `rag_demo_pro.process_multiple_pdfs`，返回分块数量与状态 |
| **问答接口** `/api/ask`                       | ✅ 已实现                     | 支持流式生成（`stream_answer`）、自动提取引用来源、结构化返回 |
| **系统状态接口** `/api/status`                | ✅ 已实现                     | 检查 Ollama 服务、SERPAPI 配置、版本与模型                   |
| **联网搜索状态接口** `/api/web_search_status` | ✅ 已实现                     | 检查是否配置 `SERPAPI_KEY`                                   |
| **进度追踪**                                  | ✅ 基于 `ProgressCallback` 类 | 替代 Gradio 的进度条，打印日志                               |
| **日志系统**                                  | ✅ 已配置 `logging`           | 格式化日志输出                                               |
| **生命周期管理**                              | ✅ 使用 `lifespan`            | 启动时检查模型环境，关闭时清理                               |
| **自动端口检测**                              | ✅ 使用 `is_port_available`   | 避免端口冲突                                                 |

📌 小结：

> 你现在的 RAG 项目，已经是一个完整的 **后端服务 (API Server)**，支持上传、检索、问答和健康检查。
>  接下来我们不需要“重写”API，而是 **进一步优化 + 教你如何讲解与扩展它**。

------

## 🧩 二、逐模块讲解代码逻辑（让你彻底掌握）

我会分模块解释，帮你理解每个部分“为什么要这样写”。

------

### ① 启动结构与依赖导入

```python
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
import rag_demo_pro
```

🔹 **作用：**

- 使用 FastAPI 作为 HTTP 框架；
- 导入 `rag_demo_pro` 中的核心功能（PDF 处理 + 问答）；
- `asynccontextmanager` 用来定义启动与关闭时的逻辑（lifespan）。

------

### ② 生命周期管理（`lifespan`）

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not rag_demo_pro.check_environment():
        logger.error("环境检查失败！请确保Ollama服务已启动且所需模型已加载")
    yield
    logger.info("API服务已关闭")
```

🔹 **作用：**

- 在启动时检测 Ollama 是否运行、模型是否可用；
- 在关闭时执行清理（目前只写了日志，可拓展为释放向量索引等）。

📘 讲给面试官听的说法：

> “我在 API 层通过 lifespan 做了环境自检和优雅关闭，保证模型服务准备就绪后再接收请求。”

------

### ③ 进度回调类（代替 Gradio）

```python
class ProgressCallback:
    def __call__(self, progress: float, desc: str = None):
        logger.info(f"进度: {progress:.2f} - {desc}")
```

🔹 **作用：**

- 模拟 Gradio 的进度条；
- 在控制台实时打印处理状态。

🧠 这样做的意义：

> 让 `rag_demo_pro.process_multiple_pdfs` 能复用原来的“progress参数”，无需改内部代码。

------

### ④ PDF 上传接口 `/api/upload`

```python
@app.post("/api/upload", response_model=FileProcessResult)
async def upload_pdf(file: UploadFile = File(...)):
```

📘 **核心逻辑流程：**

1. 检查文件类型（只能是 PDF）；
2. 保存为临时文件；
3. 调用 `rag_demo_pro.process_multiple_pdfs` 异步执行；
4. 删除临时文件；
5. 返回处理结果（是否成功 + 文档块数量）。

📦 **进阶理解：**

- 这一步实际上建立了“知识库”：PDF → 文本 → 向量化 → 存入 FAISS；
- 返回的 `chunks` 可以被 `/api/ask` 调用时检索使用。

------

### ⑤ 问答接口 `/api/ask`

```python
@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
```

📘 **逻辑流程：**

1. 接收问题（`req.question`）；
2. 调用异步包装函数 `process_answer_stream()`；
3. 收集 LLM 生成的回答；
4. 从回答中提取引用来源；
5. 返回结构化结果（answer + sources + metadata）。

------

### ⑥ 异步包装函数 `process_answer_stream`

```python
async def process_answer_stream(question: str, enable_web_search: bool):
```

🔹 **核心功能：**

- 把 `rag_demo_pro.stream_answer()`（同步生成器）包装为异步任务；
- 允许 FastAPI 异步消费、等待最终结果。

🧠 这段是技术亮点：

> 这是“同步生成器异步化”的高级写法（使用 `loop.run_in_executor`），面试时可以讲你理解了 **事件循环** 和 **非阻塞式I/O**。

------

### ⑦ 状态接口 `/api/status`

```python
@app.get("/api/status")
```

🔹 **作用：**

- 检查 Ollama 模型服务是否在线；
- 检查是否配置 `SERPAPI_KEY`；
- 返回 API 版本与可用模型。

------

### ⑧ 联网搜索状态 `/api/web_search_status`

```python
@app.get("/api/web_search_status")
```

🔹 **作用：**

- 确认 `SERPAPI_KEY` 是否设置；
- 让前端知道是否能启用联网搜索选项。

------

### ⑨ 自动端口检测

```python
for p in [17995, 17996, 17997, 17998, 17999]:
    if rag_demo_pro.is_port_available(p):
        port = p
        break
```

🔹 **作用：**

- 避免端口被占用导致启动失败；
- 每次自动找一个可用端口（很聪明的做法 👍）。

------

## 🧠 三、如何回答“你是怎么设计 API 层的？”

这是面试时的讲解模板：

> “我在项目中使用 **FastAPI** 搭建了 REST API 层，将核心的 RAG 能力封装为标准接口。
>  上传接口 `/api/upload` 负责文档解析与向量化；问答接口 `/api/ask` 负责检索增强生成；
>  同时我实现了 `process_answer_stream` 来把同步生成器转为异步流式调用，让回答可实时输出。
>  API 启动时会自动检查模型服务状态（通过 lifespan），并通过日志记录进度。
>  这样做能让整个 RAG 系统不仅能在本地 Gradio 界面运行，也能直接被前端或其他服务调用。”

------

## 🚀 四、我建议的优化方向（适合你动手练习 + 简历加分）

| 优化目标                          | 改进思路                                   | 难度 |
| --------------------------------- | ------------------------------------------ | ---- |
| 🔹 支持多文件上传                  | 修改 `/api/upload` 支持 `List[UploadFile]` | ⭐    |
| 🔹 添加 `/api/delete` 删除指定文档 | 结合 `FAISS IndexIDMap`，删除向量          | ⭐⭐   |
| 🔹 添加 `/api/chunks`              | 返回当前向量库的摘要（chunk 数量、来源）   | ⭐    |
| 🔹 添加流式回答端点 `/api/stream`  | 用 `StreamingResponse` 实现实时输出        | ⭐⭐   |
| 🔹 添加日志文件记录                | 使用 `RotatingFileHandler` 保存交互日志    | ⭐⭐   |
| 🔹 Docker 部署                     | 打包为容器，暴露 17995 端口                | ⭐⭐⭐  |

------



